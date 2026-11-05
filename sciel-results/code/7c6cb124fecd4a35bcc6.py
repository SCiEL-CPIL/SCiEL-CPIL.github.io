"""
Product-formula design for Heisenberg-chain time evolution.
Strategy:
1. Build the Hamiltonians A and B for the 6-site XXZ chain.
2. Precompute eigen-decompositions of A, B, and H=A+B for fast evaluation.
3. Use a structured local search (Nelder-Mead) to optimize the 24 coefficients
   (12 for a, 12 for b) subject to sum(a)=1, sum(b)=1 and bounds [-50, 50].
   To handle constraints, we optimize 22 free variables and derive the 12th
   of each list to satisfy the sum constraint. We use a penalty for bounds violations.
4. Strict time management to ensure completion within the 150s limit.
"""

import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Builds A and B for the 6-site XXZ Heisenberg chain.
    Sites 0..5.
    Bonds:
    0: (0,1) -> Odd -> A
    1: (1,2) -> Even -> B
    2: (2,3) -> Odd -> A
    3: (3,4) -> Even -> B
    4: (4,5) -> Odd -> A
    
    Field: Uniform field 0.3.
    "each including half of the uniform field".
    A has field term 0.15 * sum S^z, B has field term 0.15 * sum S^z.
    Anisotropy lambda = 0.8.
    """
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    # Spin operators are 0.5 * Pauli
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    
    def op_on_site(op, site):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(np.eye(2, dtype=complex))
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res
    
    def bond_term(site1, site2):
        def two_site_op(op1, site1, op2, site2):
            ops = []
            for i in range(N):
                if i == site1:
                    ops.append(op1)
                elif i == site2:
                    ops.append(op2)
                else:
                    ops.append(np.eye(2, dtype=complex))
            res = ops[0]
            for i in range(1, N):
                res = np.kron(res, ops[i])
            return res

        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    # Field
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def _nelder_mead(f, x0, max_iter=500, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, tol=1e-6, time_limit=None, start_time=None):
    """
    Pure NumPy Nelder-Mead implementation.
    """
    if start_time is None:
        start_time = time.time()
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        # Small perturbation
        step = 0.1 if x_new[i] == 0 else 0.1 * x_new[i]
        x_new[i] += step
        simplex.append(x_new)
        
    values = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Sort simplex by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
            
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        
        # Centroid of all but worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    values[i] = f(simplex[i])
                    
    # Return best point
    order = np.argsort(values)
    return simplex[order[0]], values[order[0]]

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute eigen-decompositions for A, B, H
    # A and B are Hermitian, so use eigh
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    
    times = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    U_exact = []
    for t in times:
        phase = np.exp(-1j * t * eigvals_H)
        U_t = eigvecs_H @ np.diag(phase) @ eigvecs_H.conj().T
        U_exact.append(U_t)
    
    # Precompute A and B propagators for various coefficients? No, coefficients vary.
    # But we can precompute eigvecs and eigvals.
    
    def get_propagator(eigvals, eigvecs, coeff, t, H_matrix):
        """
        Compute exp(-i coeff * t * H_matrix)
        """
        if coeff == 0:
            return np.eye(H_matrix.shape[0], dtype=complex)
        phase = np.exp(-1j * coeff * t * eigvals)
        return eigvecs @ np.diag(phase) @ eigvecs.conj().T

    def objective(params):
        """
        params: 22 values.
        a_free = params[:11]
        b_free = params[11:22]
        a_12 = 1 - sum(a_free)
        b_12 = 1 - sum(b_free)
        """
        if len(params) != 22:
            return 1e10
            
        a_free = params[:11]
        b_free = params[11:22]
        
        a = list(a_free)
        a.append(1.0 - np.sum(a_free))
        
        b = list(b_free)
        b.append(1.0 - np.sum(b_free))
        
        # Check bounds [-50, 50]
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        # Compute error
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B)
                if b[k] != 0:
                    U_B = get_propagator(eigvals_B, eigvecs_B, b[k], t, B)
                    U_approx = U_B @ U_approx
                # exp(-i a_k t A)
                if a[k] != 0:
                    U_A = get_propagator(eigvals_A, eigvecs_A, a[k], t, A)
                    U_approx = U_A @ U_approx
            
            err = np.linalg.norm(U_approx - U_exact[i], 2)
            err_sum += err
            
        return err_sum / len(times)

    # Initial guess: Strang splitting
    # a = [0.5/6]*12, b = [1/6, 0]*6
    # But we need 22 free params.
    # Let's start with symmetric coefficients.
    # a_k = 1/12 for all k? No, sum(a)=1.
    # If we use Strang: a = [1/12]*12, b = [1/12]*12? No, that's not Strang.
    # Strang: a = [0.5/6]*12 = [1/12]*12. b = [1/6, 0]*6.
    # b sum = 6*(1/6) = 1.
    # a sum = 12*(1/12) = 1.
    
    a_init = [1.0/12.0] * 12
    b_init = []
    for _ in range(6):
        b_init += [1.0/6.0, 0.0]
        
    # Extract free params
    a_free_init = a_init[:11]
    b_free_init = b_init[:11]
    
    x0 = np.array(a_free_init + b_free_init, dtype=float)
    
    # Run Nelder-Mead
    x_best, f_best = _nelder_mead(objective, x0, max_iter=1000, time_limit=140.0, start_time=start_time)
    
    # Reconstruct a and b
    a_free = x_best[:11]
    b_free = x_best[11:22]
    
    a = list(a_free)
    a.append(1.0 - np.sum(a_free))
    
    b = list(b_free)
    b.append(1.0 - np.sum(b_free))
    
    # Ensure finite and within bounds (clipping just in case, though penalty should prevent)
    for i in range(12):
        if not np.isfinite(a[i]) or abs(a[i]) > 50:
            a[i] = 1.0/12.0
        if not np.isfinite(b[i]) or abs(b[i]) > 50:
            b[i] = 0.0 if i % 2 == 1 else 1.0/6.0
            
    # Final check on sums
    a_sum = sum(a)
    b_sum = sum(b)
    
    # Adjust last element to ensure sum=1 exactly if floating point errors occurred
    if abs(a_sum - 1.0) > 1e-12:
        a[11] += (1.0 - a_sum)
    if abs(b_sum - 1.0) > 1e-12:
        b[11] += (1.0 - b_sum)
        
    # Final bounds check
    for i in range(12):
        a[i] = np.clip(a[i], -50, 50)
        b[i] = np.clip(b[i], -50, 50)
        
    # Re-normalize after clipping?
    # If clipping happened, sums might be off.
    # But clipping is a safety net. If it triggers, the result is likely bad anyway.
    # Let's just return.
    
    return a, b

# EVOLVE-BLOCK-END
