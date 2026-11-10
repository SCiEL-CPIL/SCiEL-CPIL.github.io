import numpy as np
import time
import itertools
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

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    rng = np.random.default_rng(42)
    
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
    
    # Phase 1: 5D Gaussian parametric search for global exploration
    # Parametrize P as a 2D Gaussian: P[y,x] = exp(-((x-mu_x)^2/(2*sigma_x^2) + (y-mu_y)^2/(2*sigma_y^2) + rho*(x-mu_x)*(y-mu_y)/(sigma_x*sigma_y)))
    # Parameters: mu_x, mu_y, sigma_x, sigma_y, rho (correlation)
    
    def gaussian_P(params):
        mu_x, mu_y, log_sigma_x, log_sigma_y, rho = params
        sigma_x = np.exp(log_sigma_x)
        sigma_y = np.exp(log_sigma_y)
        rho = np.tanh(rho)  # Map to (-1, 1)
        
        P = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                dx = x - mu_x
                dy = y - mu_y
                if sigma_x > 1e-6 and sigma_y > 1e-6:
                    val = -0.5 * ((dx/sigma_x)**2 + (dy/sigma_y)**2 + 2*rho*dx*dy/(sigma_x*sigma_y))
                    P[y, x] = np.exp(val)
        s = P.sum()
        if s > 1e-9:
            P /= s
        else:
            P = np.ones((n, n)) / (n*n)
        return P
    
    def gaussian_score(params):
        P = gaussian_P(params)
        return score_P(P)
    
    # Search over Gaussian parameters
    best_gauss_params = None
    best_gauss_score = 0
    
    for restart in range(30):
        if time.time() - start_time > timeout - safety_margin:
            break
        
        # Random initial parameters
        mu_x = rng.uniform(0, n-1)
        mu_y = rng.uniform(0, n-1)
        log_sigma_x = rng.uniform(np.log(0.5), np.log(5))
        log_sigma_y = rng.uniform(np.log(0.5), np.log(5))
        rho = rng.uniform(-1, 1)
        
        init_params = np.array([mu_x, mu_y, log_sigma_x, log_sigma_y, rho])
        
        try:
            result = opt.minimize(
                lambda p: -gaussian_score(p),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            P_opt = gaussian_P(result.x)
            sc = score_P(P_opt)
            
            if sc > best_gauss_score:
                best_gauss_score = sc
                best_gauss_params = result.x.copy()
            
            if sc > best_score:
                best_score = sc
                best_P = P_opt.copy()
                
        except Exception:
            pass
    
    # Phase 2: Refine best Gaussian result with full 100D SLSQP
    if best_gauss_params is not None and time.time() - start_time < timeout - safety_margin:
        P_from_gauss = gaussian_P(best_gauss_params)
        P_flat = P_from_gauss.ravel()
        try_optimize(P_flat, 'SLSQP', 500, ftol=1e-14)
    
    # Phase 3: Additional structured initializations
    inits = []
    
    # Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # Random inits
    for _ in range(30):
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
    for cx in [0, 1, 2, 7, 8, 9]:
        for cy in [0, 1, 2, 7, 8, 9]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    p[i, j] = np.exp(-((i - cy)**2 + (j - cx)**2) / 1.5)
            p /= p.sum()
            inits.append(p)
    
    # Grid inits
    for size in [2, 3, 4, 5]:
        for offset in range(n - size + 1):
            p = np.zeros((n, n))
            p[offset:offset+size, offset:offset+size] = 1.0
            p /= p.sum()
            inits.append(p)
    
    # Sparse grids
    for size in [3, 4, 5]:
        for offset in range(n - size + 1):
            p = np.zeros((n, n))
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        p[offset+i, offset+j] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
    
    # Try to maximize x-y entropy by spreading across diagonals
    p = np.zeros((n, n))
    for c in range(-9, 10):
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
    p /= p.sum()
    inits.append(p)
    
    # Put mass on 4 corners with varying weights
    for w1, w2, w3, w4 in [(1,1,1,1), (1,2,2,1), (2,1,1,2), (1,3,3,1), (3,1,1,3)]:
        p = np.zeros((n, n))
        p[0, 0] = w1
        p[0, 9] = w2
        p[9, 0] = w3
        p[9, 9] = w4
        p /= p.sum()
        inits.append(p)
    
    # Quick SLSQP from all inits
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        P_flat = P_init.ravel()
        try_optimize(P_flat, 'SLSQP', 150)
    
    # Phase 4: Perturbation-based local search from best
    if time.time() - start_time < timeout - safety_margin:
        for _ in range(30):
            if time.time() - start_time > timeout - safety_margin:
                break
            # Random perturbation
            perturbation = rng.normal(0, 0.05, (n, n))
            P_pert = best_P + perturbation
            P_pert = np.clip(P_pert, 0, None)
            s = P_pert.sum()
            if s > 0:
                P_pert /= s
                P_flat = P_pert.ravel()
                try_optimize(P_flat, 'SLSQP', 200)
    
    # Final refinement
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize(P_flat, 'SLSQP', 500, ftol=1e-14)
        try_optimize(P_flat, 'trust-constr', 300)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
