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

def get_projection_entropy_and_grad(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    q_nonzero = q[mask]
    if len(q_nonzero) == 0:
        return 0.0, np.zeros_like(P)
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    grad_P = grad_q[inv_flat]
    return H, grad_P

def objective_and_grad(z):
    P = z
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
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

def score_P(P):
    P_flat = P.ravel()
    D_val, _ = get_projection_entropy_and_grad(P_flat, D_inv_flat, D_bins)
    H0_val, _ = get_projection_entropy_and_grad(P_flat, H0_inv_flat, H0_bins)
    H1_val, _ = get_projection_entropy_and_grad(P_flat, H1_inv_flat, H1_bins)
    H2_val, _ = get_projection_entropy_and_grad(P_flat, H2_inv_flat, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    inits = []
    rng = np.random.default_rng(42)
    
    # Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # Random inits
    for _ in range(20):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
    
    # Diagonal inits (x-y = c)
    for c in range(-9, 10):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Anti-diagonal inits (x+y = c)
    for c in range(0, 19):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i + j == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Concentrate x: put mass on columns
    for cx in range(n):
        p = np.zeros((n, n))
        p[:, cx] = 1.0
        p /= p.sum()
        inits.append(p)
    
    # Concentrate y: put mass on rows
    for cy in range(n):
        p = np.zeros((n, n))
        p[cy, :] = 1.0
        p /= p.sum()
        inits.append(p)
    
    # Corner concentrations
    for cx in [0, 1, 8, 9]:
        for cy in [0, 1, 8, 9]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    p[i, j] = np.exp(-((i - cy)**2 + (j - cx)**2) / 2.0)
            p /= p.sum()
            inits.append(p)
    
    # Spread x-y: put mass where x-y is extreme
    # High x-y: x=9, y=0; Low x-y: x=0, y=9
    p = np.zeros((n, n))
    p[0, 9] = 1.0  # x-y = -9
    p[9, 0] = 1.0  # x-y = 9
    p /= p.sum()
    inits.append(p)
    
    # More spread: put mass on corners
    p = np.zeros((n, n))
    p[0, 0] = 1.0
    p[0, 9] = 1.0
    p[9, 0] = 1.0
    p[9, 9] = 1.0
    p /= p.sum()
    inits.append(p)
    
    # Try to maximize x-y entropy by spreading across diagonals
    # Put equal mass on each diagonal
    p = np.zeros((n, n))
    for c in range(-9, 10):
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
    p /= p.sum()
    inits.append(p)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize(P_flat, method, maxiter, ftol=None, gtol=None):
        nonlocal best_P, best_score
        try:
            if method == 'SLSQP':
                result = opt.minimize(
                    objective_and_grad,
                    P_flat,
                    jac=True,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': maxiter, 'ftol': ftol or 1e-12, 'disp': False}
                )
            elif method == 'trust-constr':
                from scipy.optimize import LinearConstraint
                lc = LinearConstraint(np.ones((1, 100)), 1.0, 1.0)
                result = opt.minimize(
                    objective_and_grad,
                    P_flat,
                    jac=True,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=[lc],
                    options={'maxiter': maxiter, 'gtol': gtol or 1e-10, 'disp': False}
                )
            elif method == 'L-BFGS-B':
                # L-BFGS-B doesn't support equality constraints well, so we use a penalty
                result = opt.minimize(
                    lambda x: objective_and_grad(x)[0] + 1000 * (np.sum(x) - 1.0)**2,
                    P_flat,
                    jac=lambda x: objective_and_grad(x)[1] + 2000 * (np.sum(x) - 1.0) * np.ones_like(x),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol or 1e-12, 'disp': False}
                )
            else:
                return None
            
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
    
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        
        P_flat = P_init.ravel()
        
        # Try multiple methods from each init
        try_optimize(P_flat, 'SLSQP', 300)
        if time.time() - start_time > timeout - safety_margin:
            break
        try_optimize(P_flat, 'trust-constr', 200)
        if time.time() - start_time > timeout - safety_margin:
            break
        try_optimize(P_flat, 'L-BFGS-B', 200)
    
    # Local refinement from best
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize(P_flat, 'SLSQP', 500, ftol=1e-14)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
