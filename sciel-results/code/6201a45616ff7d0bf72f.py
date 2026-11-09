import numpy as np
import time
import sys
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) 
    for a 6-site open XXZ Heisenberg chain.
    H = sum_{j=1}^{N-1} (Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + delta Sz_j Sz_{j+1}) + h sum Sz_j
    N=6, delta=0.8, h=0.3
    """
    N = 6
    delta = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        ops = [I2] * N
        ops[site] = op
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result

    def bond_op(j, j_next):
        # Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + delta Sz_j Sz_{j+1}
        term_xx = np.kron(site_op(sx, j), site_op(sx, j_next))
        term_yy = np.kron(site_op(sy, j), site_op(sy, j_next))
        term_zz = delta * np.kron(site_op(sz, j), site_op(sz, j_next))
        return term_xx + term_yy + term_zz

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Half field for each
    half_field = field / 2.0
    field_term = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        field_term += half_field * site_op(sz, j)
        
    # Odd bonds: 0-1, 2-3, 4-5
    for j in range(0, N-1, 2):
        A += bond_op(j, j+1)
    A += field_term
    
    # Even bonds: 1-2, 3-4
    for j in range(1, N-1, 2):
        B += bond_op(j, j+1)
    B += field_term
        
    return A, B

def coefficients(stages):
    """
    Optimize coefficients for a 12-stage product formula.
    U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    """
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Precompute exact propagators at evaluation times
    t_evals = [0.25, 0.5, 1.0]
    U_exact = [np.exp(-1j * t * H_total) for t in t_evals]
    
    # Precompute exponentials of A and B for various coefficients?
    # No, coefficients change. We must compute exp(-i a_k t A) each time.
    # However, we can precompute the eigen-decomposition of A and B to speed up expm.
    eigvals_A, eigvecs_A = np.linalg.eig(A)
    eigvals_B, eigvecs_B = np.linalg.eig(B)
    
    def expm_A(a_k, t):
        # exp(-i a_k t A)
        # A = V_D V_A^{-1}
        # exp(A) = V_D exp(diag(eigvals)) V_A^{-1}
        # We want exp(-i a_k t A)
        # diag(exp(-i a_k t eigvals))
        exp_diag = np.exp(-1j * a_k * t * eigvals_A)
        return eigvecs_A @ np.diag(exp_diag) @ np.linalg.inv(eigvecs_A)
        
    def expm_B(b_k, t):
        exp_diag = np.exp(-1j * b_k * t * eigvals_B)
        return eigvecs_B @ np.diag(exp_diag) @ np.linalg.inv(eigvecs_B)

    # Optimization variables: a_1..a_11, b_1..b_11
    # a_12 = 1 - sum(a_1..a_11)
    # b_12 = 1 - sum(b_1..b_11)
    
    def objective(params):
        a_vars = params[:11]
        b_vars = params[11:22]
        
        a = np.array(a_vars)
        b = np.array(b_vars)
        
        a_full = np.append(a, 1.0 - np.sum(a))
        b_full = np.append(b, 1.0 - np.sum(b))
        
        # Check bounds [-50, 50]
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10
            
        total_error = 0.0
        for t, U_ex in zip(t_evals, U_exact):
            U_approx = np.eye(2**6, dtype=complex)
            # Product order: k=1..12
            # U = E_B1 E_A1 E_B2 E_A2 ... E_B12 E_A12
            # Note: Matrix multiplication is associative.
            # We build from right to left or left to right?
            # U(t) = prod_{k=1}^{12} ( ... ( (I * E_{B1} * E_{A1}) * E_{B2} ) ... )
            # Let's iterate k=0 to 11
            for k in range(12):
                # Apply E_B then E_A
                # U_approx = U_approx @ E_Bk @ E_Ak
                # To save memory/time, we can do:
                # U_approx = (U_approx @ expm_B(b_full[k], t)) @ expm_A(a_full[k], t)
                # This is 12 * 3 = 36 expm calls per evaluation point.
                # 3 points * 36 = 108 expm calls per objective function evaluation.
                # Nelder-Mead might need ~1000-5000 evals.
                # 1000 * 108 = 108,000 expm calls.
                # 64x64 matrix expm is fast.
                
                UB = expm_B(b_full[k], t)
                UA = expm_A(a_full[k], t)
                U_approx = U_approx @ UB @ UA
                
            err = np.linalg.norm(U_approx - U_ex, ord=2)
            total_error += err
            
        return total_error / 3.0

    # Initial guess: Strang splitting (2nd order)
    # a = [1/12]*12, b = [1/12]*12 is not Strang.
    # Strang for 12 stages:
    # We can use the simple 1/12 uniform as a start, or better, a 4th order symmetric.
    # Let's try a uniform start first, it's symmetric and stable.
    x0 = np.full(22, 1.0/12.0)
    
    # Run optimization
    result = None
    best_x = x0
    best_val = objective(x0)
    
    # We can run multiple restarts or just one long run.
    # Given the time budget, one robust Nelder-Mead run is good.
    # We can also try a few different initial guesses.
    
    # Guess 1: Uniform
    # Guess 2: 4th order symmetric coefficients scaled?
    # Suzuki 4th order: a = [1/2, 1/(4-2^(1/3)), 1/2 - 1/(4-2^(1/3)), ...]
    # Too complex to map to 12 stages easily without search.
    
    # Let's just run Nelder-Mead with a generous maxiter.
    # Time check
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception as e:
            pass
            
    # If time remains, try a second run with a different start
    if time.time() - start_time < deadline - 10:
        # Perturbed uniform
        x0_2 = x0 + np.random.uniform(-0.05, 0.05, 22)
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception as e:
            pass
            
    a_vars = best_x[:11]
    b_vars = best_x[11:22]
    
    a = np.append(a_vars, 1.0 - np.sum(a_vars))
    b = np.append(b_vars, 1.0 - np.sum(b_vars))
    
    # Final check on constraints
    # Clip to [-50, 50] just in case, though optimizer should keep them sane
    # If clipping breaks sum=1, we renormalize? No, sum=1 is hard constraint.
    # The optimizer respects the parameterization which ensures sum=1.
    
    return list(a), list(b)

# EVOLVE-BLOCK-END
