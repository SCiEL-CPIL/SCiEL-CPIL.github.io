import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

_DEADLINE = 135.0  # Safety margin before 150s kill

def _build_operators(n_sites=6, anisotropy=0.8, field=0.3):
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain.
    """
    dim = 2 ** n_sites
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def kron_all(ops_list):
        result = np.array([[1.0]], dtype=complex)
        for op in ops_list:
            result = np.kron(result, op)
        return result

    def bond_operator(j, gamma, h_half):
        ops = [identity] * n_sites
        ops[j] = sx
        ops[j + 1] = sx
        term_xx = kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sy
        ops[j + 1] = sy
        term_yy = kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sz
        ops[j + 1] = sz
        term_zz = gamma * kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sz
        term_field_j = (-h_half) * kron_all(ops)

        ops = [identity] * n_sites
        ops[j + 1] = sz
        term_field_j1 = (-h_half) * kron_all(ops)

        return term_xx + term_yy + term_zz + term_field_j + term_field_j1

    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        A += bond_operator(j, anisotropy, field / 2)

    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        B += bond_operator(j, anisotropy, field / 2)

    return A, B

def _compute_exact_propagators(A, B, t_values):
    """Compute exact propagators exp(-i t (A+B)) for given t values."""
    H = A + B
    eigvals, eigvecs = np.linalg.eig(H)
    exacts = []
    for t in t_values:
        # U = V diag(exp(-i t lambda)) V^-1
        # Using V @ diag @ V^-1 is safer than V @ diag @ V^H if V is not unitary, 
        # but H is Hermitian so V is unitary.
        exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ np.linalg.inv(eigvecs)
        exacts.append(exact)
    return exacts

def _eval_error(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 22 floats.
    First 11: a_1..a_6 and a_7..a_11 (a_12 is determined by sum constraint)
    Actually, let's parameterize the first 6 a's and first 6 b's.
    a_full: [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    Sum(a_full) = 2 * sum(a1..a6) = 1 => sum(a1..a6) = 0.5
    
    To handle the constraint sum=1 in the optimizer, we can just pass 11 free variables 
    and compute the 12th, OR pass 12 variables and ignore the constraint in the 
    objective, adding a penalty. 
    Better: Pass 11 variables for a, compute the 12th to satisfy symmetry and sum.
    
    Let's use 11 variables for a: a1..a11.
    a12 = 1 - sum(a1..a11).
    But we want symmetry: a12=a1, a11=a2, etc.
    So a1..a6 are free. a7..a12 are fixed by symmetry.
    Constraint: 2*sum(a1..a6) = 1.
    This is 1 constraint on 6 variables. We can use 5 free variables and compute the 6th?
    Or just use 6 variables and add a penalty term if sum != 0.5.
    
    Let's use 6 variables for a_half and 6 for b_half. Total 12 variables.
    Objective: Error + lambda * (sum(a_half)-0.5)^2 + lambda * (sum(b_half)-0.5)^2.
    This is robust.
    """
    a_half = params[:6]
    b_half = params[6:12]
    
    # Penalty for constraint violation
    sum_a = np.sum(a_half)
    sum_b = np.sum(b_half)
    penalty = 100.0 * ((sum_a - 0.5)**2 + (sum_b - 0.5)**2)
    
    # Construct full lists
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Clip to valid range to prevent numerical explosion in exp
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    total_error = 0.0
    dim = A.shape[0]
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        # Pre-computed eigendecomps:
        # A = V_A diag(A_vals) V_A^-1
        # exp(-i a t A) = V_A diag(exp(-i a t A_vals)) V_A^-1
        # We can pre-compute V_A and V_A_inv.
        
        # To speed up, we can pre-compute the diagonal matrices for each t and coeff?
        # No, coeffs change. But we can pre-compute V_A @ diag(exp(...)) @ V_A_inv.
        
        # Let's do the multiplications.
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = eig_B[0] @ np.diag(np.exp(-1j * b_full[k] * t * eig_B[1])) @ eig_B[2]
            # exp(-i a_k t A)
            exp_A = eig_A[0] @ np.diag(np.exp(-1j * a_full[k] * t * eig_A[1])) @ eig_A[2]
            
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        # Spectral norm
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values) + penalty

def coefficients(stages):
    start_time = time.time()
    
    # 1. Build Operators
    A, B = _build_operators()
    
    # 2. Pre-compute Eigendecompositions for A and B
    # A is Hermitian, so eigenvectors are unitary.
    eigvals_A, eigvecs_A = np.linalg.eig(A)
    eigvecs_A_inv = np.linalg.inv(eigvecs_A)
    eig_data_A = (eigvecs_A, eigvals_A, eigvecs_A_inv)
    
    eigvals_B, eigvecs_B = np.linalg.eig(B)
    eigvecs_B_inv = np.linalg.inv(eigvecs_B)
    eig_data_B = (eigvecs_B, eigvals_B, eigvecs_B_inv)
    
    # 3. Compute Exact Propagators
    t_values = [0.25, 0.5, 1.0]
    exacts = _compute_exact_propagators(A, B, t_values)
    
    # 4. Initial Guess
    # Start with uniform 2nd-order symmetric splitting
    # a_half sum = 0.5, so each a_i = 0.5/6 = 1/12
    a_init = np.full(6, 1.0/12.0)
    b_init = np.full(6, 1.0/12.0)
    x0 = np.concatenate([a_init, b_init])
    
    # 5. Optimization
    best_params = x0
    best_err = _eval_error(x0, A, B, eig_data_A, eig_data_B, exacts, t_values)
    
    try:
        # Nelder-Mead is robust for this small dimension
        res = minimize(
            _eval_error, 
            x0, 
            args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            
        # Check time
        if time.time() < start_time + _DEADLINE - 10:
            # Try a second run with a different start if time permits?
            # Or just refine.
            # Let's try a second run starting from a 4th-order guess if we have time.
            # Yoshida 4th order 2-pair coefficients scaled?
            # Let's just do a local refinement from the current best.
            res2 = minimize(
                _eval_error, 
                best_params, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            if res2.fun < best_err:
                best_err = res2.fun
                best_params = res2.x
                
    except Exception as e:
        # If optimization fails, keep the initial guess
        pass
    
    # 6. Extract Final Coefficients
    a_half = best_params[:6]
    b_half = best_params[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Ensure sums are exactly 1 (correct for any numerical drift)
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
        
    # Clip to bounds
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
        
    return list(a_full), list(b_full)
