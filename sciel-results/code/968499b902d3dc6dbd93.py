"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius(c):
    """Compute the maximum radius for given centers."""
    c = np.asarray(c)
    # Distance to walls
    wall = 0.5 - np.abs(c)
    wall_min = wall.min()
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min()
    return min(wall_min, pair_min / 2.0)

def _neg_radius_smooth(c, beta):
    """Smooth approximation of negative radius using log-sum-exp."""
    c = np.asarray(c).reshape(-1, 2)
    n = len(c)
    # Wall constraints: 0.5 - |c_ik|
    wall_terms = 0.5 - np.abs(c)
    # Pairwise: (d_ij)/2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    np.fill_diagonal(dist2, np.inf)
    pair_terms = 0.5 * np.sqrt(dist2)
    
    # Soft min via log-sum-exp
    all_terms = np.concatenate([wall_terms.ravel(), pair_terms.ravel()])
    # Use log-sum-exp for numerical stability
    m = all_terms.min()
    if not np.isfinite(m):
        return 0.0
    logsum = m + np.log(np.sum(np.exp(all_terms - m)))
    soft_min = logsum / beta
    return -soft_min

def _neg_radius_grad(c, beta):
    """Gradient of smooth negative radius."""
    c = np.asarray(c).reshape(-1, 2)
    n = len(c)
    grad = np.zeros_like(c)
    
    # Wall terms
    wall_terms = 0.5 - np.abs(c)
    # d(-|c|)/dc = -sign(c)
    wall_grad = -np.sign(c)
    
    # Pairwise terms
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)
    
    # Soft min weights
    all_terms = np.concatenate([(0.5 - np.abs(c)).ravel(), (0.5 * dist).ravel()])
    m = all_terms.min()
    if not np.isfinite(m):
        return grad
    exp_terms = np.exp(all_terms - m)
    total = np.sum(exp_terms)
    
    # Wall weights
    wall_w = exp_terms[:n*2] / total / beta
    # Pair weights
    pair_w = exp_terms[n*2:] / total / beta
    pair_w = pair_w.reshape(n, n)
    np.fill_diagonal(pair_w, 0.0)
    
    # Wall gradient contribution
    grad += wall_w.reshape(n, 2) * wall_grad
    
    # Pair gradient contribution
    # d(0.5 * |c_i - c_j|)/dc_i = 0.5 * (c_i - c_j) / |c_i - c_j|
    unit = diff / dist[:, :, np.newaxis]
    pair_grad = 0.5 * unit
    for i in range(n):
        for j in range(n):
            if i != j:
                grad[i] += pair_w[i, j] * pair_grad[i, j]
                grad[j] -= pair_w[i, j] * pair_grad[i, j]
    
    return grad

def _simulated_annealing(init, max_iter=5000, T0=0.05, T1=0.001, alpha=0.9995):
    """Simulated annealing to escape local minima."""
    c = init.copy()
    best_c = c.copy()
    best_r = _radius(c)
    T = T0
    
    for it in range(max_iter):
        # Random perturbation
        pert = np.random.normal(0, T, size=c.shape)
        c_new = c + pert
        
        # Keep within bounds
        c_new = np.clip(c_new, -0.48, 0.48)
        
        r_new = _radius(c_new)
        r_cur = _radius(c)
        
        # Accept if better or with probability exp((r_new - r_cur)/T)
        if r_new > r_cur or np.random.random() < np.exp((r_new - r_cur) / max(T, 1e-10)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        T = max(T * alpha, T1)
        
        if it % 500 == 0 and it > 0:
            # Refine with L-BFGS periodically
            try:
                res = minimize(_neg_radius_smooth, c, jac=_neg_radius_grad, 
                             args=(100,), method='L-BFGS-B', 
                             bounds=[(-0.49, 0.49)]*2*n,
                             options={'maxiter': 20, 'ftol': 1e-12})
                if res.success:
                    c = res.x
                    r_check = _radius(c)
                    if r_check > best_r:
                        best_r = r_check
                        best_c = c.copy()
            except:
                pass
    
    return best_c, best_r

def _lbfgs_polish(c, beta=100, maxiter=200):
    """Polish with L-BFGS."""
    try:
        res = minimize(_neg_radius_smooth, c, jac=_neg_radius_grad,
                     args=(beta,), method='L-BFGS-B',
                     bounds=[(-0.499, 0.499)]*2*N,
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success:
            return res.x
    except:
        pass
    return c

def construct():
    """Construct centers for N equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial hexagonal packing
    def _hex_init():
        # Create hexagonal grid
        pts = []
        r = 0.075  # Starting radius estimate
        dx = 2 * r
        dy = r * np.sqrt(3)
        x = -0.45
        while x < 0.45:
            y = -0.45
            row = 0
            while y < 0.45:
                x_offset = x + (row % 2) * r
                if -0.49 <= x_offset <= 0.49 and -0.49 <= y <= 0.49:
                    pts.append([x_offset, y])
                y += dy
                row += 1
            x += dx
        pts = np.array(pts)
        # Shuffle and take N
        idx = np.random.choice(len(pts), size=min(N, len(pts)), replace=False)
        return pts[idx]
    
    best_c = None
    best_r = -1
    
    # Multiple random starts
    n_starts = 8
    for start in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Different initialization strategies
        if start == 0:
            # Hexagonal
            c = _hex_init()
            if len(c) < N:
                # Pad with random
                extra = np.random.uniform(-0.4, 0.4, size=(N - len(c), 2))
                c = np.vstack([c, extra])
        elif start == 1:
            # Grid with jitter
            m = 8
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            c = grid[:N].copy()
            c += np.random.normal(0, 0.01, size=c.shape)
        else:
            # Random
            c = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Ensure valid
        c = np.clip(c, -0.48, 0.48)
        
        # SA phase
        if time.time() > deadline - 40:
            break
        
        c_sa, r_sa = _simulated_annealing(c, max_iter=3000)
        
        if r_sa > best_r:
            best_r = r_sa
            best_c = c_sa.copy()
        
        # L-BFGS polish
        if time.time() > deadline - 15:
            break
        
        c_polished = _lbfgs_polish(c_sa, beta=150, maxiter=300)
        r_polished = _radius(c_polished)
        
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
        
        # Final tight polish
        if time.time() > deadline - 5:
            break
        
        c_final = _lbfgs_polish(c_polished, beta=300, maxiter=500)
        r_final = _radius(c_final)
        
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # If we haven't found anything, return grid
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    # Final validation
    best_c = np.clip(best_c, -0.499, 0.499)
    
    return best_c
# EVOLVE-BLOCK-END
