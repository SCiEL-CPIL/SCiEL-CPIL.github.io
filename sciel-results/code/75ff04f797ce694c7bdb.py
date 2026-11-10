import numpy as np
import time
import scipy.optimize as opt

# Pre-compute the index maps for full 100-dim projections
x_indices = np.arange(10)
y_indices = np.arange(10)
X_grid, Y_grid = np.meshgrid(x_indices, y_indices, indexing='xy')

D_vals = (X_grid - Y_grid)
H0_vals = X_grid
H1_vals = (X_grid + Y_grid)
H2_vals = (X_grid + 2 * Y_grid)

D_uniq, D_inv = np.unique(D_vals, return_inverse=True)
H0_uniq, H0_inv = np.unique(H0_vals, return_inverse=True)
H1_uniq, H1_inv = np.unique(H1_vals, return_inverse=True)
H2_uniq, H2_inv = np.unique(H2_vals, return_inverse=True)

D_inv_flat = D_inv.ravel()
H0_inv_flat = H0_inv.ravel()
H1_inv_flat = H1_inv.ravel()
H2_inv_flat = H2_inv.ravel()

D_bins = len(D_uniq)
H0_bins = len(H0_uniq)
H1_bins = len(H1_uniq)
H2_bins = len(H2_uniq)

def get_projection_entropy(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    if np.sum(mask) == 0:
        return 0.0
    q_nonzero = q[mask]
    return -np.sum(q_nonzero * np.log2(q_nonzero))

def get_projection_entropy_grad(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    if np.sum(mask) == 0:
        return 0.0, np.zeros_like(P)
    q_nonzero = q[mask]
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    grad_P = grad_q[inv_flat]
    return H, grad_P

def score_P(P):
    P_flat = P.ravel()
    D_val = get_projection_entropy(P_flat, D_inv_flat, D_bins)
    H0_val = get_projection_entropy(P_flat, H0_inv_flat, H0_bins)
    H1_val = get_projection_entropy(P_flat, H1_inv_flat, H1_bins)
    H2_val = get_projection_entropy(P_flat, H2_inv_flat, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def smooth_max_entropy(P_flat, temperature=0.1):
    """Compute smooth max of H0, H1, H2 using log-sum-exp."""
    H0_val, H0_grad = get_projection_entropy_grad(P_flat, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P_flat, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P_flat, H2_inv_flat, H2_bins)
    
    Hs = np.array([H0_val, H1_val, H2_val])
    grads = [H0_grad, H1_grad, H2_grad]
    
    H_max = np.max(Hs)
    if H_max < 1e-9:
        return 1e-9, np.zeros_like(P_flat)
    
    exps = np.exp((Hs - H_max) / temperature)
    exps /= exps.sum()
    M_smooth = H_max + temperature * np.log(exps.sum())
    
    M_grad = sum(exps[i] * grads[i] for i in range(3))
    return M_smooth, M_grad

def objective_and_grad_smooth(z, temperature=0.1):
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    M_val, M_grad = smooth_max_entropy(P, temperature)
    
    if M_val < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M_val
    grad_ratio = (M_val * D_grad - D_val * M_grad) / (M_val * M_val)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def objective_and_grad(z):
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    M = np.max(H012)
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M
    idx_max = np.argmax(H012)
    M_grad = grads_012[idx_max]
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

# --- Product-form helpers: P[y,x] = R[y] * C[x] ---
def _build_projection_maps(proj_vals):
    """For a 10x10 grid of projection values, build for each (y, v) the list of x."""
    maps = {}
    for y in range(10):
        for x in range(10):
            v = proj_vals[y, x]
            if v not in maps:
                maps[v] = {}
            if y not in maps[v]:
                maps[v][y] = []
            maps[v][y].append(x)
    return maps

D_map = _build_projection_maps(D_vals)
H0_map = _build_projection_maps(H0_vals)
H1_map = _build_projection_maps(H1_vals)
H2_map = _build_projection_maps(H2_vals)

def _proj_entropy_from_RC(R, C, pmap, n_bins):
    """Compute projection entropy given R (10,), C (10,), and precomputed map."""
    q = np.zeros(n_bins)
    for v in range(n_bins):
        if v not in pmap:
            continue
        for y, xs in pmap[v].items():
            q[v] += R[y] * np.sum(C[xs])
    mask = q > 1e-15
    if not np.any(mask):
        return 0.0, q
    qn = q[mask]
    H = -np.sum(qn * np.log2(qn))
    return H, q

def _proj_entropy_grad_from_RC(R, C, pmap, n_bins):
    """Compute projection entropy and gradient wrt R and C."""
    q = np.zeros(n_bins)
    for v in range(n_bins):
        if v not in pmap:
            continue
        for y, xs in pmap[v].items():
            q[v] += R[y] * np.sum(C[xs])
    mask = q > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(R), np.zeros_like(C)
    qn = q[mask]
    H = -np.sum(qn * np.log2(qn))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(qn) - 1.0 / np.log(2.0)
    dR = np.zeros(10)
    dC = np.zeros(10)
    for v in range(n_bins):
        if v not in pmap:
            continue
        gq = grad_q[v]
        if gq == 0:
            continue
        for y, xs in pmap[v].items():
            dR[y] += gq * np.sum(C[xs])
            for x in xs:
                dC[x] += gq * R[y]
    return H, dR, dC

def _product_score(R, C):
    """Compute D/M ratio for product form."""
    R = np.clip(R, 0, None)
    C = np.clip(C, 0, None)
    Rs = R.sum()
    Cs = C.sum()
    if Rs < 1e-15 or Cs < 1e-15:
        return 0.0
    R /= Rs
    C /= Cs
    D_val, _ = _proj_entropy_from_RC(R, C, D_map, D_bins)
    H0_val, _ = _proj_entropy_from_RC(R, C, H0_map, H0_bins)
    H1_val, _ = _proj_entropy_from_RC(R, C, H1_map, H1_bins)
    H2_val, _ = _proj_entropy_from_RC(R, C, H2_map, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def _product_obj_grad(z, temperature=0.05):
    """Objective and gradient for product-form optimization. z = [log_R(10), log_C(10)]."""
    log_R = z[:10]
    log_C = z[10:]
    R = np.exp(np.clip(log_R, -20, 20))
    C = np.exp(np.clip(log_C, -20, 20))
    Rs = R.sum()
    Cs = C.sum()
    if Rs < 1e-15 or Cs < 1e-15:
        return -1e9, np.zeros_like(z)
    R = R / Rs
    C = C / Cs
    
    D_val, dR_D, dC_D = _proj_entropy_grad_from_RC(R, C, D_map, D_bins)
    H0_val, dR_H0, dC_H0 = _proj_entropy_grad_from_RC(R, C, H0_map, H0_bins)
    H1_val, dR_H1, dC_H1 = _proj_entropy_grad_from_RC(R, C, H1_map, H1_bins)
    H2_val, dR_H2, dC_H2 = _proj_entropy_grad_from_RC(R, C, H2_map, H2_bins)
    
    Hs = np.array([H0_val, H1_val, H2_val])
    dRs = [dR_H0, dR_H1, dR_H2]
    dCs = [dC_H0, dC_H1, dC_H2]
    
    H_max = np.max(Hs)
    if H_max < 1e-9:
        return -1e9, np.zeros_like(z)
    exps = np.exp((Hs - H_max) / temperature)
    exps /= exps.sum()
    M_smooth = H_max + temperature * np.log(exps.sum())
    
    dR_M = sum(exps[i] * dRs[i] for i in range(3))
    dC_M = sum(exps[i] * dCs[i] for i in range(3))
    
    ratio = D_val / M_smooth
    dR_ratio = (M_smooth * dR_D - D_val * dR_M) / (M_smooth * M_smooth)
    dC_ratio = (M_smooth * dC_D - D_val * dC_M) / (M_smooth * M_smooth)
    
    dlogR = np.zeros(10)
    dlogC = np.zeros(10)
    for y in range(10):
        for i in range(10):
            dlogR[i] += dR_ratio[y] * R[y] * (1.0 if i == y else 0.0 - R[i])
    for x in range(10):
        for i in range(10):
            dlogC[i] += dC_ratio[x] * C[x] * (1.0 if i == x else 0.0 - C[i])
    
    obj = -ratio
    grad = np.concatenate([dlogR, dlogC])
    return obj, grad

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 630
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Product-form optimization (20 params) ---
    prod_inits = []
    
    # Uniform
    prod_inits.append(np.concatenate([np.zeros(10), np.zeros(10)]))
    
    # Anisotropic Gaussian initializations with negative correlation
    # Key insight: sigma_x ~ 2*sigma_y minimizes H(x+2y) while maximizing H(x-y)/H(x)
    # We encode this by scaling log_R (for x) and log_C (for y) differently
    # and introducing anti-correlation patterns
    
    # Pattern: x-spread wider than y-spread
    for sigma_x in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for sigma_y in [0.3, 0.5, 0.7, 1.0]:
            for rho in [-0.8, -0.5, -0.3, 0.0]:
                log_R = np.zeros(10)
                log_C = np.zeros(10)
                # Create correlated Gaussian-like distributions
                for i in range(10):
                    # Center at 4.5
                    dx = (i - 4.5) / sigma_x
                    dy = (i - 4.5) / sigma_y
                    # Add correlation via offset
                    if rho < 0:
                        # Negative correlation: when x is high, y is low
                        offset_x = dx
                        offset_y = -rho * dx  # y shifts opposite to x
                    else:
                        offset_x = dx
                        offset_y = dx
                    log_R[i] = -0.5 * offset_x**2
                    log_C[i] = -0.5 * offset_y**2
                prod_inits.append(np.concatenate([log_R, log_C]))
    
    # Random initializations
    for _ in range(60):
        log_R = rng.normal(0, 1, 10)
        log_C = rng.normal(0, 1, 10)
        prod_inits.append(np.concatenate([log_R, log_C]))
    
    # Structured patterns emphasizing anisotropy
    for pattern_r in [
        [0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,1],
        [0,1,0,0,0,0,0,1,0,0],
        [1,1,0,0,0,0,0,0,1,1],
        [0,0,1,0,0,0,1,0,0,0],
        [1,0,0,0,1,0,0,0,0,1],
    ]:
        for pattern_c in [
            [0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,1],
            [0,1,0,0,0,0,0,1,0,0],
            [1,1,0,0,0,0,0,0,1,1],
        ]:
            log_R = np.array(pattern_r, dtype=float)
            log_C = np.array(pattern_c, dtype=float)
            prod_inits.append(np.concatenate([log_R, log_C]))
    
    bounds_prod = [(-10, 10)] * 20
    
    def try_optimize_product(z_init, maxiter=500, temperature=0.05):
        nonlocal best_P, best_score
        try:
            result = opt.minimize(
                _product_obj_grad,
                z_init,
                args=(temperature,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds_prod,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            log_R = result.x[:10]
            log_C = result.x[10:]
            R = np.exp(np.clip(log_R, -20, 20))
            C = np.exp(np.clip(log_C, -20, 20))
            Rs = R.sum()
            Cs = C.sum()
            if Rs < 1e-15 or Cs < 1e-15:
                return None
            R /= Rs
            C /= Cs
            P = np.outer(R, C)
            s = P.sum()
            if s < 1e-9:
                return None
            P /= s
            sc = score_P(P)
            if sc > best_score:
                best_score = sc
                best_P = P.copy()
            return P
        except Exception:
            pass
        return None
    
    # Optimize from all product inits
    for z_init in prod_inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        try_optimize_product(z_init, 400, 0.05)
    
    # Refine best with smaller temperature
    if time.time() - start_time < timeout - safety_margin:
        U, s, Vt = np.linalg.svd(best_P)
        if s[0] > 1e-10:
            R = U[:, 0] * np.sqrt(s[0])
            C = Vt[0] * np.sqrt(s[0])
            R = np.abs(R)
            C = np.abs(C)
            if R.sum() > 1e-15 and C.sum() > 1e-15:
                R /= R.sum()
                C /= C.sum()
                log_R = np.log(R + 1e-15)
                log_C = np.log(C + 1e-15)
                z_init = np.concatenate([log_R, log_C])
                try_optimize_product(z_init, 500, 0.02)
                try_optimize_product(z_init, 500, 0.01)
    
    # --- Phase 2: Full 100-dim polish from product-form result ---
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize_lbfgs(P_flat, maxiter=500, temperature=0.05):
        nonlocal best_P, best_score
        try:
            result = opt.minimize(
                objective_and_grad_smooth,
                P_flat,
                args=(temperature,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
                return P_opt
        except Exception:
            pass
        return None
    
    def try_optimize_slsqp(P_flat, maxiter=300, ftol=1e-12):
        nonlocal best_P, best_score
        try:
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
                return P_opt
        except Exception:
            pass
        return None
    
    # Full 100-dim optimization from product-form best
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_lbfgs(P_flat, 400, 0.05)
        try_optimize_lbfgs(P_flat, 400, 0.02)
    
    # SLSQP refinement
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_slsqp(P_flat, 500, ftol=1e-14)
    
    # Local perturbations in 100-dim
    if time.time() - start_time < timeout - safety_margin:
        for _ in range(20):
            if time.time() - start_time > timeout - safety_margin:
                break
            perturbation = rng.random((n, n)) * 0.05
            P_pert = best_P + perturbation
            P_pert = np.clip(P_pert, 0, None)
            s = P_pert.sum()
            if s > 0:
                P_pert /= s
                P_flat = P_pert.ravel()
                try_optimize_lbfgs(P_flat, 300, 0.05)
    
    # Final high-precision polish
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_slsqp(P_flat, 1000, ftol=1e-16)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
