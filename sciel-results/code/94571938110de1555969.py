import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import time

def _build_heisenberg_matrices():
    """
    Constructs A and B for a 6-site open XXZ Heisenberg chain.
    Anisotropy delta = 0.8, Field h = 0.3.
    H = sum_{j=1}^{5} [ Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + delta Sz_j Sz_{j+1} ] - h sum_{j=1}^6 Sz_j
    A corresponds to odd bonds (1-2, 3-4, 5-6) + half field?
    Task says: "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    So A = H_odd + 0.5 * H_field
       B = H_even + 0.5 * H_field
    """
    n_sites = 6
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    # Build full space operators
    # We need Sx_j, Sy_j, Sz_j acting on site j of 6 sites.
    # Use kron products.
    ops = {'x': sx, 'y': sy, 'z': sz, 'i': identity}
    
    def op_on_site(j, op_name):
        """Returns 2^6 x 2^6 matrix for op_name on site j (0-indexed)."""
        mats = []
        for i in range(n_sites):
            if i == j:
                mats.append(ops[op_name])
            else:
                mats.append(identity)
        # Order matters: site 0 is leftmost or rightmost?
        # Standard kron: A \otimes B.
        # If we index sites 0..5, usually site 0 is the first factor.
        # Let's assume site 0 is the most significant bit (leftmost in kron).
        for k in range(1, len(mats)):
            mats[k] = np.kron(mats[k-1], mats[k])
        return mats[-1]

    delta = 0.8
    h = 0.3
    
    H_odd = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    H_even = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    H_field = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    
    # Bonds: 0-1 (odd), 1-2 (even), 2-3 (odd), 3-4 (even), 4-5 (odd)
    # Indices 0..5.
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    def add_bond(H, j, k):
        # Sx_j Sx_k + Sy_j Sy_k + delta Sz_j Sz_k
        H += np.dot(op_on_site(j, 'x'), op_on_site(k, 'x'))
        H += np.dot(op_on_site(j, 'y'), op_on_site(k, 'y'))
        H += delta * np.dot(op_on_site(j, 'z'), op_on_site(k, 'z'))

    add_bond(H_odd, 0, 1)
    add_bond(H_odd, 2, 3)
    add_bond(H_odd, 4, 5)
    
    add_bond(H_even, 1, 2)
    add_bond(H_even, 3, 4)
    
    for j in range(n_sites):
        H_field += op_on_site(j, 'z')
        
    H_field *= -h # -h sum Sz
    
    A = H_odd + 0.5 * H_field
    B = H_even + 0.5 * H_field
    
    return A, B

def _compute_error(coeffs, A, B, times):
    """
    coeffs: (a, b) lists of length 12.
    Returns mean spectral norm error.
    """
    a, b = coeffs
    errors = []
    # Precompute exps for A and B? No, coefficients change.
    # But we can precompute eigendecomposition of A and B for speed?
    # exp(-i t A) = V diag(exp(-i t lambda)) V^-1.
    # Since A and B are Hermitian, V is unitary.
    
    # Let's precompute eigendecomposition once outside if possible, 
    # but here we are inside the function called by optimizer.
    # We can pass eigendata as closure or global.
    
    evals_A, evecs_A = sla.eigh(A)
    evals_B, evecs_B = sla.eigh(B)
    
    # Precompute exps for all t?
    # U_exact(t) = exp(-i t (A+B)).
    # We can diagonalize H = A+B.
    evals_H, evecs_H = sla.eigh(A + B)
    
    for t in times:
        # Construct U(t)
        U = np.eye(2**6, dtype=complex)
        # Order: k=1..12, exp(-i b_k t B) exp(-i a_k t A)
        # Note: Python lists are 0-indexed.
        for k in range(12):
            # exp(-i b_k t B)
            # B = V_B diag(evals_B) V_B^H
            # exp(-i b_k t B) = V_B diag(exp(-i b_k t evals_B)) V_B^H
            exp_B = evecs_B @ np.diag(np.exp(-1j * b[k] * t * evals_B)) @ evecs_B.conj().T
            U = U @ exp_B
            
            # exp(-i a_k t A)
            exp_A = evecs_A @ np.diag(np.exp(-1j * a[k] * t * evals_A)) @ evecs_A.conj().T
            U = U @ exp_A
        
        # U_exact
        U_exact = evecs_H @ np.diag(np.exp(-1j * t * evals_H)) @ evecs_H.conj().T
        
        # Error
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
        
    return np.mean(errors)

def coefficients(stages):
    """
    Optimizes coefficients for a 12-stage product formula.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Build Matrices
    A, B = _build_heisenberg_matrices()
    
    # Precompute eigendecompositions for speed in objective function
    evals_A, evecs_A = sla.eigh(A)
    evals_B, evecs_B = sla.eigh(B)
    evals_H, evecs_H = sla.eigh(A + B)
    
    times = [0.25, 0.5, 1.0]
    dim = 2**6
    
    # Objective function using precomputed eigs
    def objective(params):
        # params: 12 values for a, 12 values for b?
        # We enforce symmetry to reduce parameters and improve convergence.
        # a_k = a_{13-k}, b_k = b_{13-k}
        # Indices 0..11. Pairs: (0,11), (1,10), (2,9), (3,8), (4,7), (5,6)
        # Let params be [a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5]
        # a = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
        # b = [b0, b1, b2, b3, b4, b5, b5, b4, b3, b2, b1, b0]
        
        a_half = params[0:6]
        b_half = params[6:12]
        
        a = list(a_half) + list(a_half[::-1])
        b = list(b_half) + list(b_half[::-1])
        
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty to stop optimization
        
        # Calculate error
        U = np.eye(dim, dtype=complex)
        
        # We can speed this up by precomputing the diagonal exponentials
        # but the matrices V are fixed.
        # U = V_B D_B0 V_B^H V_A D_A0 V_A^H ...
        
        # To avoid repeated matrix mults, we can just do it.
        # 12 steps * 2 mults * 64x64. Fast enough.
        
        for k in range(12):
            # exp(-i b_k t B)
            # We need to do this for each t?
            # The objective function takes a single t? No, it takes mean over t.
            # So we must loop over t inside objective.
            
            # This might be slow if called many times.
            # Let's optimize the structure.
            
            pass # Placeholder, implemented below
            
        # Re-implementing objective with t-loop
        total_err = 0.0
        for t in times:
            U_t = np.eye(dim, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B)
                diag_B = np.exp(-1j * b[k] * t * evals_B)
                mat_B = evecs_B @ np.diag(diag_B) @ evecs_B.conj().T
                U_t = U_t @ mat_B
                
                # exp(-i a_k t A)
                diag_A = np.exp(-1j * a[k] * t * evals_A)
                mat_A = evecs_A @ np.diag(diag_A) @ evecs_A.conj().T
                U_t = U_t @ mat_A
            
            # U_exact
            diag_H = np.exp(-1j * t * evals_H)
            U_exact = evecs_H @ np.diag(diag_H) @ evecs_H.conj().T
            
            total_err += np.linalg.norm(U_t - U_exact, 2)
            
        return total_err / len(times)

    # Initial guess: Uniform 1/12
    # Or 4th order Suzuki coefficients?
    # Let's start with uniform symmetric
    x0 = np.array([1.0/12]*12)
    
    # Try Nelder-Mead
    res = opt.minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    # Check deadline
    if time.time() > deadline:
        # Return best found so far or initial
        best_x = res.x if hasattr(res, 'x') else x0
    else:
        best_x = res.x
        
    # Construct final a and b
    a_half = best_x[0:6]
    b_half = best_x[6:12]
    a = list(a_half) + list(a_half[::-1])
    b = list(b_half) + list(b_half[::-1])
    
    # Ensure sums are 1 (normalization)
    # The optimizer might drift slightly.
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-9:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-9:
        b = [x / sum_b for x in b]
        
    # Clip to [-50, 50]
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    # Re-normalize after clipping
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-9:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-9:
        b = [x / sum_b for x in b]

    return a, b
