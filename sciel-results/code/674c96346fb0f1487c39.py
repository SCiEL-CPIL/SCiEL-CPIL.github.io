import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 140.0  # Safety margin before 150s kill

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
    eigvals, eigvecs = np.linalg.eigh(H)
    exacts = []
    for t in t_values:
        exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        exacts.append(exact)
    return exacts

def _eval_error_frobenius(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 22 floats. First 11 are a_free, next 11 are b_free.
    a_12 = 1 - sum(a_free), b_12 = 1 - sum(b_free).
    Returns sum of squared Frobenius norms.
    """
    a_free = params[:11]
    b_free = params[11:22]
    
    a_full = np.empty(12)
    b_full = np.empty(12)
    
    a_full[:11] = a_free
    a_full[11] = 1.0 - np.sum(a_free)
    
    b_full[:11] = b_free
    b_full[11] = 1.0 - np.sum(b_free)
    
    # Note: We rely on the bounds of the optimizer to keep values in a safe range.
    # If values go out of [-50, 50], the evaluator will reject or clip.
    # For the surrogate, we just compute the error.
    
    total_error = 0.0
    dim = A.shape[0]
    
    V_A, lam_A = eig_A
    V_B, lam_B = eig_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            exp_B = V_B @ np.diag(np.exp(-1j * b_full[k] * t * lam_B)) @ V_B_h
            exp_A = V_A @ np.diag(np.exp(-1j * a_full[k] * t * lam_A)) @ V_A_h
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        # Frobenius norm squared: trace(diff^H diff)
        frob_sq = np.vdot(diff, diff).real
        total_error += frob_sq
        
    return total_error

def _eval_error_spectral(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 22 floats. First 11 are a_free, next 11 are b_free.
    Returns mean spectral norm error.
    """
    a_free = params[:11]
    b_free = params[11:22]
    
    a_full = np.empty(12)
    b_full = np.empty(12)
    
    a_full[:11] = a_free
    a_full[11] = 1.0 - np.sum(a_free)
    
    b_full[:11] = b_free
    b_full[11] = 1.0 - np.sum(b_free)
    
    # Clip to valid range for final evaluation
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping if necessary to maintain consistency
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b

    total_error = 0.0
    dim = A.shape[0]
    
    V_A, lam_A = eig_A
    V_B, lam_B = eig_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            exp_B = V_B @ np.diag(np.exp(-1j * b_full[k] * t * lam_B)) @ V_B_h
            exp_A = V_A @ np.diag(np.exp(-1j * a_full[k] * t * lam_A)) @ V_A_h
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)

def coefficients(stages):
    start_time = time.time()
    
    # 1. Build Operators
    A, B = _build_operators()
    
    # 2. Pre-compute Eigendecompositions for A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eig_data_A = (eigvecs_A, eigvals_A)
    
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eig_data_B = (eigvecs_B, eigvals_B)
    
    # 3. Compute Exact Propagators
    t_values = [0.25, 0.5, 1.0]
    exacts = _compute_exact_propagators(A, B, t_values)
    
    # 4. Initial Guess - Uniform coefficients
    # a_i = 1/12 for all i.
    # a_free = [1/12, ..., 1/12] (11 times)
    # a_12 = 1 - 11/12 = 1/12. Correct.
    x0 = np.full(22, 1.0/12.0)
    
    # Bounds for the 22 free variables.
    # We want to allow values that sum to 1, but keep individual values reasonable.
    # If we use [-0.5, 1.5], the last variable can be 1 - 11*(-0.5) = 6.5 or 1 - 11*1.5 = -15.5.
    # These are within [-50, 50].
    bounds = [(-0.5, 1.5)] * 22
    
    best_params = x0.copy()
    best_err_spectral = _eval_error_spectral(best_params, A, B, eig_data_A, eig_data_B, exacts, t_values)
    
    try:
        # Optimization using L-BFGS-B on the Frobenius norm surrogate
        # This is smooth and allows for efficient gradient-based optimization.
        
        def objective(params):
            return _eval_error_frobenius(params, A, B, eig_data_A, eig_data_B, exacts, t_values)
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 20:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < _eval_error_frobenius(x0, A, B, eig_data_A, eig_data_B, exacts, t_values):
                candidate_params = res.x
                # Evaluate with spectral norm to compare
                cand_err = _eval_error_spectral(candidate_params, A, B, eig_data_A, eig_data_B, exacts, t_values)
                if cand_err < best_err_spectral:
                    best_err_spectral = cand_err
                    best_params = candidate_params
        
        # Second round: Polish with a different method or tighter bounds if time permits
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 10:
            # Try Nelder-Mead on the spectral norm directly from the best found point
            # This can help escape local minima of the Frobenius surrogate if they differ
            def objective_spectral(params):
                return _eval_error_spectral(params, A, B, eig_data_A, eig_data_B, exacts, t_values)
            
            res2 = minimize(
                objective_spectral,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res2.fun < best_err_spectral:
                best_err_spectral = res2.fun
                best_params = res2.x

    except Exception as e:
        # If optimization fails, keep the initial guess or best found
        pass
    
    # 5. Extract Final Coefficients
    a_free = best_params[:11]
    b_free = best_params[11:22]
    
    a_full = np.empty(12)
    b_full = np.empty(12)
    
    a_full[:11] = a_free
    a_full[11] = 1.0 - np.sum(a_free)
    
    b_full[:11] = b_free
    b_full[11] = 1.0 - np.sum(b_free)
    
    # Final cleanup: Clip and Renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    else:
        a_full = np.full(12, 1.0/12.0)
        
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
    else:
        b_full = np.full(12, 1.0/12.0)
        
    return list(a_full), list(b_full)
