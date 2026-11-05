import time
import numpy as np

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

def _eval_error_symmetric(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 10 floats. 
    First 5 are x_0..x_4 for A symmetry.
    Last 5 are y_0..y_4 for B symmetry.
    
    A coefficients: [x0, x1, x2, x3, x4, x5, x5, x4, x3, x2, x1, x0]
    where x5 = 0.5 - sum(x0..x4)
    
    B coefficients: [y0, y1, y2, y3, y4, y5, y5, y4, y3, y2, y1, y0]
    where y5 = 0.5 - sum(y0..y4)
    """
    x_free = params[:5]
    y_free = params[5:]
    
    x5 = 0.5 - np.sum(x_free)
    y5 = 0.5 - np.sum(y_free)
    
    a_full = np.array([x_free[0], x_free[1], x_free[2], x_free[3], x_free[4], x5, 
                       x5, x_free[4], x_free[3], x_free[2], x_free[1], x_free[0]])
    b_full = np.array([y_free[0], y_free[1], y_free[2], y_free[3], y_free[4], y5, 
                       y5, y_free[4], y_free[3], y_free[2], y_free[1], y_free[0]])
    
    # Check bounds
    if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
        return 1e10

    total_error = 0.0
    dim = A.shape[0]
    
    V_A, lam_A = eig_A
    V_B, lam_B = eig_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # Pre-compute exp factors for this t and coefficient
            exp_B_vals = np.exp(-1j * b_full[k] * t * lam_B)
            exp_B = (V_B * exp_B_vals) @ V_B_h
            
            exp_A_vals = np.exp(-1j * a_full[k] * t * lam_A)
            exp_A = (V_A * exp_A_vals) @ V_A_h
            
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)

def _nelder_mead(func, x0, bounds, max_iter=500, tol=1e-8, time_limit=None, start_time=None):
    """
    A simple Nelder-Mead implementation with bounds handling via penalty.
    """
    n = len(x0)
    # Initial simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 if x0[i] == 0 else abs(x0[i]) * 0.1
        simplex[i + 1][i] += step
    
    # Evaluate initial simplex
    f_simplex = np.array([func(x) for x in simplex])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_x = x0.copy()
    best_f = np.min(f_simplex)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Sort simplex
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        if f_simplex[0] < best_f:
            best_f = f_simplex[0]
            best_x = simplex[0].copy()
            
        # Check convergence
        if np.max(f_simplex) - np.min(f_simplex) < tol:
            break
            
        centroid = np.mean(simplex[1:], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = func(x_r)
        
        if f_r < f_simplex[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = func(x_c)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = func(simplex[i])
                    
        # Handle bounds by pushing back into bounds if needed
        # Simple penalty approach: if out of bounds, func returns 1e10, so NM will naturally move away.
        # But to be safe, we can clip the simplex points.
        # However, clipping breaks the geometry. The penalty method is safer for this.
        
    return best_x, best_f

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
    
    # 4. Optimization Strategy
    # We use a multi-start Nelder-Mead approach.
    # The problem is 10-dimensional (5 for A, 5 for B).
    
    best_params = None
    best_err = np.inf
    
    # Define the objective function wrapper
    def objective(params):
        return _eval_error_symmetric(params, A, B, eig_data_A, eig_data_B, exacts, t_values)
    
    # Initial guesses
    # 1. Uniform
    x_uniform = np.full(10, 1.0/12.0)
    
    # 2. Slightly perturbed uniform
    rng = np.random.default_rng(42)
    
    starts = [x_uniform]
    for i in range(5):
        noise = rng.normal(0, 0.05, size=10)
        starts.append(x_uniform + noise)
    
    for x0 in starts:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        try:
            # Run Nelder-Mead
            # We don't use strict bounds in NM, but the objective function penalizes out-of-bounds.
            # To help NM, we can restrict the search space implicitly by the penalty.
            # However, standard NM can wander far. 
            # Let's add a simple time check inside the loop of NM? 
            # My custom NM has a time_limit parameter.
            
            x_opt, f_opt = _nelder_mead(
                objective, 
                x0, 
                bounds=None, # Handled by penalty in objective
                max_iter=300, 
                tol=1e-10,
                time_limit=_DEADLINE - (time.time() - start_time) - 5.0,
                start_time=start_time
            )
            
            if f_opt < best_err:
                best_err = f_opt
                best_params = x_opt
                
        except Exception as e:
            pass
            
    # Fallback if no good params found (should not happen)
    if best_params is None:
        best_params = x_uniform
        
    # 5. Extract Final Coefficients
    x_free = best_params[:5]
    y_free = best_params[5:]
    
    x5 = 0.5 - np.sum(x_free)
    y5 = 0.5 - np.sum(y_free)
    
    a_full = np.array([x_free[0], x_free[1], x_free[2], x_free[3], x_free[4], x5, 
                       x5, x_free[4], x_free[3], x_free[2], x_free[1], x_free[0]])
    b_full = np.array([y_free[0], y_free[1], y_free[2], y_free[3], y_free[4], y5, 
                       y5, y_free[4], y_free[3], y_free[2], y_free[1], y_free[0]])
    
    # Final sanity check: ensure finite and within bounds
    # If out of bounds, we might need to project back, but the penalty should have prevented this.
    # If we are still out of bounds, we return the uniform solution as a safe fallback.
    if not (np.all(np.isfinite(a_full)) and np.all(np.isfinite(b_full))):
        a_full = np.full(12, 1.0/12.0)
        b_full = np.full(12, 1.0/12.0)
        
    if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
        # Project to bounds? Or just use uniform?
        # Uniform is always valid.
        a_full = np.full(12, 1.0/12.0)
        b_full = np.full(12, 1.0/12.0)
        
    return list(a_full), list(b_full)
