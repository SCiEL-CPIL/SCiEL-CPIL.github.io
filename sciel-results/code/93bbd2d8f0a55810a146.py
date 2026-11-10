import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp

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

def objective_and_grad(z, beta=100.0):
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    scaled_H = beta * H012
    max_scaled = np.max(scaled_H)
    exp_scaled = np.exp(scaled_H - max_scaled)
    sum_exp = np.sum(exp_scaled)
    
    if sum_exp < 1e-300:
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
    
    weights = exp_scaled / sum_exp
    M = (max_scaled + np.log(sum_exp)) / beta
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    M_grad = np.zeros_like(P)
    for i in range(3):
        M_grad += weights[i] * grads_012[i]
    
    ratio = D_val / M
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def coordinate_descent(P, max_iter=1000, tol=1e-8):
    n = P.shape[0]
    P = P.copy()
    current_score = score_P(P)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for j in range(n):
                # Try moving mass from (i,j) to other cells? 
                # Simpler: try adjusting P[i,j] while keeping sum 1.
                # Since we are on a simplex, we can try swapping mass between two cells.
                # Or just do a small random perturbation and check.
                pass
        
        # Alternative: Use SLSQP for a few iterations as a "local" step within CD loop?
        # Or implement a proper 1D line search along a random direction.
        # Let's stick to the SA approach primarily, and use SLSQP at the end.
        # The "coordinate descent" in the knowledge might refer to optimizing one variable at a time 
        # under the constraint. 
        # Let's implement a simple random direction line search.
        
        # Generate a random direction vector d
        d = np.random.randn(n, n)
        # Project d to be orthogonal to the constraint sum(x)=1? 
        # Actually, we can just move P + alpha * d, then clip and renormalize.
        # But renormalization changes the direction.
        # Better: Move along a direction that preserves sum?
        # Let's just use SLSQP for the final polish. The SA is the main improvement.
        
        if it % 100 == 0 and time.time() > 600:
            break
            
    return P

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    rng = np.random.default_rng(42)
    
    inits = []
    inits.append(np.ones((n, n)) / (n * n))
    
    for _ in range(30):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
        
    # Add some structured inits
    for c in range(-9, 10):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
            
    for c in range(0, 19):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i + j == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
            
    # Corners
    for cx in [0, 9]:
        for cy in [0, 9]:
            p = np.zeros((n, n))
            p[cy, cx] = 1.0
            inits.append(p)
            
    # Quick local opt from inits
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize(P_flat, method, maxiter, ftol=None, beta=100.0):
        nonlocal best_P, best_score
        try:
            if method == 'SLSQP':
                result = opt.minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    P_flat,
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
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
        try_optimize(P_init.ravel(), 'SLSQP', 200, beta=50.0)

    # Simulated Annealing
    # Start from best_P found by local opt
    current_P = best_P.copy()
    current_score = best_score
    
    # Parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.9995 # Cooling rate
    max_iters = 50000
    
    # We will run SA for a while, then do more local opt
    # Time budget for SA: ~400s
    sa_deadline = start_time + 400
    
    for i in range(max_iters):
        if time.time() > sa_deadline:
            break
            
        # Generate a new candidate
        # Perturb current_P
        # Add noise, clip, normalize
        noise_scale = 0.1 * T_start / (T_start + i * (T_end - T_start) / max_iters) # Adaptive noise?
        # Simpler: fixed noise scale, but T controls acceptance
        noise = rng.normal(0, 0.1, (n, n))
        candidate = current_P + noise
        candidate = np.clip(candidate, 0, None)
        s = candidate.sum()
        if s < 1e-9:
            candidate = current_P
        else:
            candidate /= s
            
        new_score = score_P(candidate)
        
        # If new score is 0 (invalid), we might want to reject it strongly
        # But if current is also 0, we are stuck. 
        # score_P returns 0 if M < 1e-9.
        
        if new_score > current_score:
            current_P = candidate
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_P = candidate.copy()
        else:
            # Accept with probability exp((new - old) / T)
            T = T_start * (T_end / T_start) ** (i / max_iters)
            if T > 1e-9:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_P = candidate
                        current_score = new_score
                else:
                    current_P = candidate
                    current_score = new_score
    
    # Final local optimization from the SA result
    if time.time() - start_time < timeout - safety_margin:
        try_optimize(current_P.ravel(), 'SLSQP', 500, ftol=1e-14, beta=100.0)
        try_optimize(best_P.ravel(), 'SLSQP', 500, ftol=1e-14, beta=200.0)
        
    # One more SA round if time permits?
    if time.time() - start_time < timeout - safety_margin - 50:
        current_P = best_P.copy()
        current_score = best_score
        T_start = 0.5
        T_end = 1e-8
        max_iters = 20000
        sa_deadline = timeout - safety_margin
        
        for i in range(max_iters):
            if time.time() > sa_deadline:
                break
            noise = rng.normal(0, 0.05, (n, n))
            candidate = current_P + noise
            candidate = np.clip(candidate, 0, None)
            s = candidate.sum()
            if s < 1e-9:
                candidate = current_P
            else:
                candidate /= s
            new_score = score_P(candidate)
            
            if new_score > current_score:
                current_P = candidate
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_P = candidate.copy()
            else:
                T = T_start * (T_end / T_start) ** (i / max_iters)
                if T > 1e-9:
                    delta = new_score - current_score
                    if delta < 0:
                        prob = np.exp(delta / T)
                        if rng.random() < prob:
                            current_P = candidate
                            current_score = new_score
    
    # Final polish
    if time.time() - start_time < timeout - safety_margin:
        try_optimize(best_P.ravel(), 'SLSQP', 1000, ftol=1e-15, beta=500.0)
        
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
