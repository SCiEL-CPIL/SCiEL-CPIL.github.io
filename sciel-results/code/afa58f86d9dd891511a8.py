import time
import numpy as np

_DEADLINE = 130.0  # Safety margin before 150s kill

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

def _eval_error_fast(params, A, B, eig_A, eig_B, exacts, t_values, precomputed):
    """
    params: array of 24 floats. First 12 are a coefficients, last 12 are b coefficients.
    Uses precomputed eigenvectors for speed.
    """
    a_raw = params[:12]
    b_raw = params[12:24]
    
    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a_raw)
    sum_b = np.sum(b_raw)
    
    if np.abs(sum_a) < 1e-12:
        sum_a = 1.0
    if np.abs(sum_b) < 1e-12:
        sum_b = 1.0
        
    a_full = a_raw / sum_a
    b_full = b_raw / sum_b
    
    # Clip to valid range
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping if necessary
    sum_a_clip = np.sum(a_full)
    sum_b_clip = np.sum(b_full)
    if np.abs(sum_a_clip) > 1e-12:
        a_full = a_full / sum_a_clip
    if np.abs(sum_b_clip) > 1e-12:
        b_full = b_full / sum_b_clip
    
    total_error = 0.0
    dim = A.shape[0]
    
    V_A, lam_A = eig_A
    V_B, lam_B = eig_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # Pre-compute diagonal exponentials
            exp_B_diag = np.exp(-1j * b_full[k] * t * lam_B)
            exp_A_diag = np.exp(-1j * a_full[k] * t * lam_A)
            
            exp_B = (V_B * exp_B_diag) @ V_B_h
            exp_A = (V_A * exp_A_diag) @ V_A_h
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)



def _nelder_mead(func, x0, args=(), maxiter=1000, timeout=None, start_time=None):
    """
    Pure NumPy Nelder-Mead simplex optimization with adaptive step sizes.
    """
    n = len(x0)
    
    # Initialize simplex with adaptive step size
    simplex = [np.array(x0, dtype=float)]
    step = 0.1
    for i in range(n):
        x_i = x0.copy()
        x_i[i] += step * (1.0 if x0[i] == 0 else np.sign(x0[i]))
        simplex.append(x_i)
    
    # Evaluate initial simplex
    fvals = [func(s, *args) for s in simplex]
    
    # Sort by function value
    idx = np.argsort(fvals)
    simplex = [simplex[i] for i in idx]
    fvals = [fvals[i] for i in idx]
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for iteration in range(maxiter):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Current best
        best_x = simplex[0].copy()
        best_f = fvals[0]
        
        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = func(x_ref, *args)
        
        if f_ref < fvals[0]:
            # Try expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = func(x_exp, *args)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            x_con = centroid + rho * (simplex[-1] - centroid)
            f_con = func(x_con, *args)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i], *args)
        
        # Sort by function value
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < 1e-10:
            break
    
    return simplex[0], fvals[0]

def _build_comm_modes(n=12):
    """
    Build basis vectors for commutator-scaled parameterization.
    These modes are designed to cancel leading-order error terms.
    
    For a 12-stage product formula, we construct palindromic/symmetric modes
    that correspond to the structure of high-order splittings.
    
    The Suzuki-Yoshida 4th order 5-stage formula has coefficients:
    x1 = 1/(2*(2-2^(1/3)))
    x2 = -2^(1/3)/(2*(2-2^(1/3)))
    x3 = (2-2^(1/3))/(2*(2-2^(1/3)))
    
    We extend this to 12 stages with symmetric patterns and add
    commutator-inspired perturbation modes.
    """
    modes_a = []
    modes_b = []
    
    # Base: Suzuki-Yoshida 4th order pattern extended to 12 stages
    # SY4 5-stage: [x1, x2, x3, x2, x1] for A, [0, x1, x2, x1, 0] for B
    # Extend to 12 stages with symmetry
    
    cbrt2 = 2 ** (1.0 / 3.0)
    sy4_a = [1/(2*(2-cbrt2)), -cbrt2/(2*(2-cbrt2)), (2-cbrt2)/(2*(2-cbrt2)),
             -cbrt2/(2*(2-cbrt2)), 1/(2*(2-cbrt2))]
    sy4_b = [0, 1/(2*(2-cbrt2)), -cbrt2/(2*(2-cbrt2)), 1/(2*(2-cbrt2)), 0]
    
    # Scale SY4 to sum to 1
    sum_sy4_a = sum(sy4_a)
    sum_sy4_b = sum(sy4_b)
    sy4_a = [x / sum_sy4_a for x in sy4_a]
    sy4_b = [x / sum_sy4_b for x in sy4_b]
    
    # Mode 0: Base SY4 extended to 12 stages (palindromic)
    # Repeat and mirror: [a1,a2,a3,a4,a5,a5,a4,a3,a2,a1] -> 10, need 12
    # Use: [a1,a2,a3,a4,a5,a5,a4,a3,a2,a1,0,0] won't work well
    # Better: distribute SY4 over 12 with padding
    base_a = np.zeros(12)
    base_b = np.zeros(12)
    # Map 5 SY4 stages to 12 positions with symmetry
    # Positions: 0,1,2,3,4,5,6,7,8,9,10,11
    # Symmetric: pos i and pos 11-i
    # SY4 stages at positions: 0, 3, 6, 9, 11 (roughly)
    # Let's use a smoother mapping
    for i, (ai, bi) in enumerate(zip(sy4_a, sy4_b)):
        # Spread each SY4 stage over ~2.4 positions
        start_pos = int(i * 12 / 5)
        end_pos = int((i + 1) * 12 / 5)
        for p in range(start_pos, max(end_pos, start_pos + 1)):
            base_a[p] += ai / max(end_pos - start_pos, 1)
            base_b[p] += bi / max(end_pos - start_pos, 1)
    
    # Ensure sums are 1
    base_a /= np.sum(base_a)
    base_b /= np.sum(base_b)
    
    modes_a.append(base_a)
    modes_b.append(base_b)
    
    # Mode 1: Palindromic "edge emphasis" - stronger at edges
    # This helps with boundary effects in the open chain
    edge_a = np.array([1, 0.5, 0.3, 0.2, 0.15, 0.15, 0.15, 0.15, 0.2, 0.3, 0.5, 1.0])
    edge_a = (edge_a - np.mean(edge_a)) / np.linalg.norm(edge_a)
    modes_a.append(edge_a)
    
    edge_b = np.array([0.5, 1, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 1, 0.5])
    edge_b = (edge_b - np.mean(edge_b)) / np.linalg.norm(edge_b)
    modes_b.append(edge_b)
    
    # Mode 2: "Middle emphasis" - stronger in center
    mid_a = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.2])
    mid_a = (mid_a - np.mean(mid_a)) / np.linalg.norm(mid_a)
    modes_a.append(mid_a)
    
    mid_b = np.array([0.3, 0.5, 0.7, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 0.7, 0.5, 0.3])
    mid_b = (mid_b - np.mean(mid_b)) / np.linalg.norm(mid_b)
    modes_b.append(mid_b)
    
    # Mode 3: Alternating pattern (commutator-like)
    alt_a = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    alt_a = (alt_a - np.mean(alt_a)) / np.linalg.norm(alt_a)
    modes_a.append(alt_a)
    
    alt_b = np.array([0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1], dtype=float)
    alt_b = (alt_b - np.mean(alt_b)) / np.linalg.norm(alt_b)
    modes_b.append(alt_b)
    
    # Mode 4: Linear gradient
    lin_a = np.linspace(-1, 1, 12)
    lin_a = (lin_a - np.mean(lin_a)) / np.linalg.norm(lin_a)
    modes_a.append(lin_a)
    
    lin_b = np.linspace(-1, 1, 12)
    lin_b = (lin_b - np.mean(lin_b)) / np.linalg.norm(lin_b)
    modes_b.append(lin_b)
    
    # Mode 5: Quadratic
    quad_a = np.linspace(-1, 1, 12) ** 2
    quad_a = (quad_a - np.mean(quad_a)) / np.linalg.norm(quad_a)
    modes_a.append(quad_a)
    
    quad_b = np.linspace(-1, 1, 12) ** 2
    quad_b = (quad_b - np.mean(quad_b)) / np.linalg.norm(quad_b)
    modes_b.append(quad_b)
    
    return np.array(modes_a), np.array(modes_b)


def _comm_to_coeffs(weights_a, weights_b, base_a, base_b, modes_a, modes_b):
    """Convert commutator-mode weights to actual coefficients."""
    a = base_a.copy()
    b = base_b.copy()
    
    for i in range(len(weights_a)):
        a += weights_a[i] * modes_a[i]
    for i in range(len(weights_b)):
        b += weights_b[i] * modes_b[i]
    
    # Normalize to sum to 1
    if abs(np.sum(a)) > 1e-12:
        a = a / np.sum(a)
    else:
        a = np.ones(12) / 12
    
    if abs(np.sum(b)) > 1e-12:
        b = b / np.sum(b)
    else:
        b = np.ones(12) / 12
    
    return a, b


def _eval_error_comm(weights, A, B, eig_A, eig_B, exacts, t_values, base_a, base_b, modes_a, modes_b):
    """Evaluate error using commutator-mode parameterization."""
    weights_a = weights[:len(modes_a)]
    weights_b = weights[len(modes_a):]
    
    a, b = _comm_to_coeffs(weights_a, weights_b, base_a, base_b, modes_a, modes_b)
    
    params = np.concatenate([a, b])
    return _eval_error_fast(params, A, B, eig_A, eig_B, exacts, t_values, None)


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
    
    # 4. Build commutator modes
    modes_a, modes_b = _build_comm_modes()
    base_a = modes_a[0]  # First mode is the base SY4 pattern
    base_b = modes_b[0]
    
    n_modes = len(modes_a)
    n = 12
    
    # 5. Phase 1: Optimize in commutator-mode space (low-dimensional)
    # Start with zero weights (pure SY4 base)
    w0 = np.zeros(2 * n_modes)
    
    best_params = np.concatenate([base_a, base_b])
    best_err = _eval_error_fast(best_params, A, B, eig_data_A, eig_data_B, exacts, t_values, None)
    
    try:
        rng = np.random.RandomState(42)
        
        # Optimize in commutator mode space
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 30.0
        if time_remaining > 15.0:
            p_comm, v_comm = _nelder_mead(
                _eval_error_comm,
                w0,
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values, base_a, base_b, modes_a, modes_b),
                maxiter=200,
                timeout=time_remaining,
                start_time=start_time
            )
            
            if v_comm < best_err:
                best_err = v_comm
                a_opt, b_opt = _comm_to_coeffs(p_comm[:n_modes], p_comm[n_modes:], base_a, base_b, modes_a, modes_b)
                best_params = np.concatenate([a_opt, b_opt])
        
        # Phase 2: Refine in full 24D space from the commutator-optimized point
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 10.0
        if time_remaining > 10.0:
            # Multiple restarts in full space
            n_restarts = 3
            
            for restart in range(n_restarts):
                elapsed = time.time() - start_time
                time_remaining = _DEADLINE - elapsed - 8.0
                if time_remaining < 5.0:
                    break
                
                if restart == 0:
                    x_start = best_params.copy()
                else:
                    scale = 0.03 * (restart + 1)
                    x_start = best_params + rng.randn(2 * n) * scale
                    # Normalize sums
                    x_start[:n] = x_start[:n] / np.sum(x_start[:n])
                    x_start[n:] = x_start[n:] / np.sum(x_start[n:])
                
                p, v = _nelder_mead(
                    _eval_error_fast,
                    x_start,
                    args=(A, B, eig_data_A, eig_data_B, exacts, t_values, None),
                    maxiter=150,
                    timeout=time_remaining,
                    start_time=start_time
                )
                if v < best_err:
                    best_err = v
                    best_params = p.copy()
        
        # Phase 3: Final fine-tune
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 3.0
        if time_remaining > 3.0:
            p2, v2 = _nelder_mead(
                _eval_error_fast,
                best_params,
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values, None),
                maxiter=100,
                timeout=time_remaining,
                start_time=start_time
            )
            if v2 < best_err:
                best_err = v2
                best_params = p2.copy()
                
    except Exception:
        # If optimization fails, keep current best
        pass
    
    # 6. Extract Final Coefficients
    a_full = best_params[:n]
    b_full = best_params[n:]
    
    # Ensure sums are exactly 1 and within bounds
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    else:
        a_full = np.ones(n) / n
    
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
    else:
        b_full = np.ones(n) / n
        
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
