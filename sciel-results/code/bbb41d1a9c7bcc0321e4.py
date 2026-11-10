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

def softmax(z):
    z_max = np.max(z)
    e_z = np.exp(z - z_max)
    return e_z / e_z.sum()

def objective_and_grad_z(z, smooth_beta=100.0):
    P = softmax(z)
    
    D_val = get_projection_entropy(P, D_inv_flat, D_bins)
    H0_val = get_projection_entropy(P, H0_inv_flat, H0_bins)
    H1_val = get_projection_entropy(P, H1_inv_flat, H1_bins)
    H2_val = get_projection_entropy(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    
    if np.max(H012) < 1e-9:
        return -1e9, np.zeros_like(z)
    
    # Smooth max approximation for gradient
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
    
    # Gradient of softmax: dP_i/dz_j = P_i (delta_ij - P_j)
    # grad_z_j = sum_i grad_ratio_i * dP_i/dz_j
    # = sum_i grad_ratio_i P_i (delta_ij - P_j)
    # = grad_ratio_j P_j - P_j sum_i grad_ratio_i P_i
    
    dot_grad_P = np.dot(grad_ratio, P)
    grad_z = P * (grad_ratio - dot_grad_P)
    
    obj = -ratio
    return obj, -grad_z

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 630
    safety_margin = 60
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    best_z = np.zeros(n * n)
    
    # Generate diverse initializations in z-space
    inits_z = []
    inits_z.append(np.zeros(n * n))
    
    np.random.seed(42)
    for _ in range(20):
        z = np.random.randn(n * n) * np.random.uniform(0.5, 2.0)
        inits_z.append(z)
        
    # Heuristic: concentrate on anti-diagonals (high D)
    for diag in range(-5, 6):
        P = np.zeros((n, n))
        for y in range(n):
            x = y + diag
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            P = P + 1e-10
            P /= P.sum()
            inits_z.append(np.log(P).ravel())

    # Heuristic: concentrate on lines x+2y=k
    for k in range(0, 20):
        P = np.zeros((n, n))
        for y in range(n):
            x = k - 2 * y
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            P = P + 1e-10
            P /= P.sum()
            inits_z.append(np.log(P).ravel())

    # Heuristic: mix of rows and columns
    for _ in range(10):
        P = np.zeros((n, n))
        rows = np.random.choice(n, size=np.random.randint(2, 6), replace=False)
        cols = np.random.choice(n, size=np.random.randint(2, 6), replace=False)
        for r in rows:
            P[r, :] = 1.0
        for c in cols:
            P[:, c] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            P = P + 1e-10
            P /= P.sum()
            inits_z.append(np.log(P).ravel())

    # Phase 1: Initial Optimization
    for z_init in inits_z:
        if time.time() - start_time > timeout - safety_margin:
            break
        
        try:
            result = opt.minimize(
                objective_and_grad_z,
                z_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            P_opt = softmax(result.x).reshape(n, n)
            sc = score_P(P_opt)
            if sc > best_score:
                best_score = sc
                best_P = P_opt.copy()
                best_z = result.x.copy()
                
        except Exception:
            pass

    # Phase 2: Support Fixing and Refinement
    z_work = best_z.copy()
    P_work = softmax(z_work).reshape(n, n)
    score_work = best_score
    
    for iteration in range(10):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Identify support
        threshold = 1e-5
        support_mask = P_work > threshold
        
        # Fix support
        z_fixed = z_work.copy()
        z_fixed[~support_mask.ravel()] = -100.0
        
        try:
            result = opt.minimize(
                objective_and_grad_z,
                z_fixed,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            P_new = softmax(result.x).reshape(n, n)
            sc_new = score_P(P_new)
            
            if sc_new > score_work:
                score_work = sc_new
                P_work = P_new
                z_work = result.x
                if score_work > best_score:
                    best_score = score_work
                    best_P = P_new.copy()
                    best_z = z_work.copy()
            else:
                # Perturb and re-optimize
                perturbation = np.random.randn(n * n) * 0.5
                z_perturbed = z_work + perturbation
                
                result = opt.minimize(
                    objective_and_grad_z,
                    z_perturbed,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                P_new = softmax(result.x).reshape(n, n)
                sc_new = score_P(P_new)
                
                if sc_new > score_work:
                    score_work = sc_new
                    P_work = P_new
                    z_work = result.x
                    if score_work > best_score:
                        best_score = score_work
                        best_P = P_new.copy()
                        best_z = z_work.copy()
                else:
                    threshold *= 0.1
                    if threshold < 1e-8:
                        break
                        
        except Exception:
            pass

    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
