import numpy as np
import time
import scipy.optimize as opt

# Pre-compute the index maps for projections
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
    if not np.any(mask):
        return 0.0
    q_nz = q[mask]
    return -np.sum(q_nz * np.log2(q_nz))

def get_projection_entropy_grad(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    if not np.any(mask):
        return np.zeros_like(P)
    q_nz = q[mask]
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nz) - 1.0 / np.log(2.0)
    return grad_q[inv_flat]

def score_P(P):
    P_flat = P.ravel()
    D = get_projection_entropy(P_flat, D_inv_flat, D_bins)
    H0 = get_projection_entropy(P_flat, H0_inv_flat, H0_bins)
    H1 = get_projection_entropy(P_flat, H1_inv_flat, H1_bins)
    H2 = get_projection_entropy(P_flat, H2_inv_flat, H2_bins)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M

def objective_and_grad(z, smooth_beta=50.0):
    P = np.maximum(z, 1e-15)
    P = P / P.sum()
    
    D_val = get_projection_entropy(P, D_inv_flat, D_bins)
    H0_val = get_projection_entropy(P, H0_inv_flat, H0_bins)
    H1_val = get_projection_entropy(P, H1_inv_flat, H1_bins)
    H2_val = get_projection_entropy(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    
    if np.max(H012) < 1e-9:
        return -1e9, np.zeros_like(z)
    
    # Smooth max approximation
    scaled = smooth_beta * H012
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    weights /= weights.sum()
    
    M_smooth = np.sum(weights * H012)
    
    D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
    grads_012 = np.array([H0_grad, H1_grad, H2_grad])
    M_grad = np.sum(weights[:, None] * grads_012, axis=0)
    
    ratio = D_val / M_smooth
    grad_ratio = (M_smooth * D_grad - D_val * M_grad) / (M_smooth * M_smooth)
    
    # Account for normalization: P = z / sum(z)
    # dP_i/dz_j = (delta_ij * sum(z) - z_i) / sum(z)^2
    S = z.sum()
    S2 = S * S
    # grad with respect to z
    grad_z = np.zeros_like(z)
    for i in range(100):
        dP_dz = np.zeros(100)
        dP_dz[i] = 1.0 / S
        dP_dz -= z / S2
        grad_z[i] = np.dot(grad_ratio, dP_dz)
    
    # Simpler: use chain rule
    # d/dz_j [f(z/S)] = sum_i df_i/dP_i * dP_i/dz_j
    # dP_i/dz_j = delta_ij/S - z_i/S^2
    # So grad_z_j = (1/S) * grad_ratio_j - (1/S^2) * sum_i grad_ratio_i * z_i
    dot_gr_z = np.dot(grad_ratio, z)
    grad_z = grad_ratio / S - dot_gr_z * z / S2
    
    obj = -ratio
    return obj, -grad_z

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 630
    safety_margin = 60
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    # Generate diverse initializations
    inits = []
    inits.append(np.ones((n, n)) / (n * n))
    
    # Random
    np.random.seed(42)
    for _ in range(20):
        p = np.random.rand(n, n) ** np.random.uniform(0.5, 3.0)
        p /= p.sum()
        inits.append(p)
    
    # Heuristic: concentrate on a few anti-diagonals
    for diag in range(-5, 6):
        p = np.zeros((n, n))
        for y in range(n):
            x = y + diag
            if 0 <= x < n:
                p[y, x] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Heuristic: mix of rows and columns
    for _ in range(10):
        p = np.zeros((n, n))
        # Pick a few rows and columns
        rows = np.random.choice(n, size=np.random.randint(2, 6), replace=False)
        cols = np.random.choice(n, size=np.random.randint(2, 6), replace=False)
        for r in rows:
            p[r, :] = 1.0
        for c in cols:
            p[:, c] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Heuristic: power-law along diagonals
    for _ in range(10):
        p = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                d = x - y
                p[y, x] = np.exp(-0.3 * abs(d) ** 1.5)
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Heuristic: concentrate x in narrow range, spread y
    for x_center in range(2, 8):
        for x_width in [1, 2, 3]:
            p = np.zeros((n, n))
            for x in range(max(0, x_center - x_width), min(n, x_center + x_width + 1)):
                p[:, x] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        
        P_flat = P_init.ravel()
        
        try:
            # Try SLSQP with smooth max
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 800, 'ftol': 1e-10, 'disp': False}
            )
            
            P_opt = np.clip(result.x, 0, None)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                P_opt = P_opt.reshape(n, n)
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for local refinement
        if time.time() - start_time > timeout - safety_margin:
            break
        
        try:
            result_nm = opt.minimize(
                lambda x: -score_P(np.clip(x, 0, None).reshape(n, n) / max(np.clip(x, 0, None).sum(), 1e-15)),
                P_flat,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            P_opt = np.clip(result_nm.x, 0, None)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                P_opt = P_opt.reshape(n, n)
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final local refinement on best
    if time.time() - start_time < timeout - safety_margin:
        try:
            P_flat = best_P.ravel()
            result_final = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            P_opt = np.clip(result_final.x, 0, None)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                P_opt = P_opt.reshape(n, n)
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
