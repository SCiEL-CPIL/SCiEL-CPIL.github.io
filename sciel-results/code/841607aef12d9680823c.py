import numpy as np
import time
import scipy.optimize

N = 50
_T0 = time.time()
_DEADLINE = 118.0  # safety margin below the 150s wall-clock limit

def _score(pts):
    """Radius for a centre array: min(pair_dist/2, wall_dist)."""
    d = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)

def _softmin_objective(pts, beta):
    """
    Returns the negative of the soft-min radius.
    Minimizing this maximizes the radius.
    Uses log-sum-exp for stability.
    """
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    # Take upper triangle to avoid double counting and self-distance
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    wall_dists_flat = wall_dists.flatten()
    
    # Combine all constraints
    vals = np.concatenate([pair_dists, wall_dists_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * vals)))
    # To avoid underflow, subtract max
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = max_val - log_sum / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _softmin_gradient(pts, beta):
    """
    Gradient of the soft-min objective.
    """
    d = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    wall_dists = 0.5 - np.abs(pts)
    wall_dists_flat = wall_dists.flatten()
    
    vals = np.concatenate([pair_dists, wall_dists_flat])
    
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_exp = np.sum(exp_vals)
    
    # Probabilities
    probs = exp_vals / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Gradient of pair distances
    # d/dx_k (dist(i,j)/2) = (x_i - x_j) / (2 * dist(i,j))
    # Note: pair_dists[k] corresponds to (i,k), (j,k) in iu
    # We need to accumulate gradients for each point
    
    # This is a bit slow in pure python/numpy loops for 50 points (1225 pairs), 
    # but acceptable if we do it carefully or use vectorized ops where possible.
    # Actually, let's just compute the gradient of the softmin w.r.t vals, then chain rule.
    
    # d(softmin)/d(vals_m) = -probs[m]
    # d(pair_dist)/d(pts) and d(wall_dist)/d(pts)
    
    # For pairs:
    # val_k = dist(i,j)/2
    # d(val_k)/d(x_i) = (x_i - x_j) / (2 * dist(i,j))
    # d(val_k)/d(x_j) = (x_j - x_i) / (2 * dist(i,j))
    
    # Let's vectorize the pair gradient part.
    # We have pairs (i, j). 
    # diff = pts[i] - pts[j]
    # dist = norm(diff)
    # grad_xi = diff / (2 * dist)
    # grad_xj = -diff / (2 * dist)
    
    # Since N=50, 1225 pairs is small. We can loop over pairs or use fancy indexing.
    i_idx, j_idx = iu
    diffs = pts[i_idx] - pts[j_idx]
    dists_pair = np.linalg.norm(diffs, axis=1)
    # Avoid division by zero
    dists_pair_safe = np.where(dists_pair < 1e-8, 1e-8, dists_pair)
    coeffs = probs[:len(i_idx)] / (2.0 * dists_pair_safe)
    
    # grad_xi = sum of coeffs * diffs for each i
    # grad_xj = sum of -coeffs * diffs for each j
    grad_pairs_i = np.zeros_like(pts)
    grad_pairs_j = np.zeros_like(pts)
    np.add.at(grad_pairs_i, i_idx, coeffs[:, None] * diffs)
    np.add.at(grad_pairs_j, j_idx, -coeffs[:, None] * diffs)
    
    grad += grad_pairs_i + grad_pairs_j
    
    # For walls:
    # val = 0.5 - |x_k|
    # d(val)/d(x_k) = -sign(x_k)
    wall_probs = probs[len(i_idx):]
    wall_grad = -np.sign(pts) * wall_probs.reshape(N, 3)
    grad += wall_grad
    
    # The objective is -softmin, so gradient is -d(softmin)/dx
    # d(softmin)/dx = sum_m (d softmin/d val_m) * (d val_m/dx)
    # d softmin/d val_m = -probs[m]
    # So d(softmin)/dx = - sum_m probs[m] * (d val_m/dx)
    # Objective = -softmin
    # d Obj/dx = sum_m probs[m] * (d val_m/dx)
    
    # Wait, let's re-verify signs.
    # f(x) = - S(x) where S(x) = softmin(vals)
    # S(x) = max_val - (1/beta) log(sum exp(-beta(vals - max_val)))
    # d S / d vals_m = - exp(-beta(vals_m - max_val)) / sum exp(...) = -prob_m
    # d f / d vals_m = - d S / d vals_m = prob_m
    # d f / d x = sum_m prob_m * (d vals_m / d x)
    
    # So the gradient we computed (sum probs * dval/dx) is correct for the objective f.
    return grad

def _polish(pts, beta_init=50.0, beta_final=1000.0, steps=100):
    """
    Polishes the configuration using L-BFGS-B on the soft-min objective.
    """
    cur = pts.copy()
    cur_obj = _softmin_objective(cur, beta_init)
    
    # Simple gradient ascent with decreasing step size? 
    # Or use scipy.optimize.minimize?
    # L-BFGS-B is better.
    
    # We need to be careful with the beta. 
    # Let's do a few iterations with increasing beta.
    
    best_pts = cur.copy()
    best_radius = _score(cur)
    
    for i in range(steps):
        beta = beta_init * (beta_final / beta_init) ** (i / steps)
        
        # Use L-BFGS-B
        res = scipy.optimize.minimize(
            _softmin_objective, 
            cur, 
            args=(beta,), 
            method='L-BFGS-B', 
            jac=_softmin_gradient,
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 5, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success:
            cur = res.x
            # Check if it's valid (inside cube, no overlaps in hard sense? No, softmin handles it)
            # But we must ensure it's inside the cube. L-BFGS-B with bounds should keep it.
            # However, sometimes it might drift slightly. Clip it.
            cur = np.clip(cur, -0.5 + 1e-9, 0.5 - 1e-9)
            
            r = _score(cur)
            if r > best_radius:
                best_radius = r
                best_pts = cur.copy()
                
        if time.time() > _DEADLINE:
            break
            
    return best_pts, best_radius

def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing with random restarts on the minimum radius."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        # Random kick: move a random subset by a small random amount.
        kick = 0.01 * (T / T0 + 0.1)
        idx = rng.choice(N, size=max(1, N // 4), replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand = np.clip(cand, -0.5 + 1e-9, 0.5 - 1e-9)
        
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 100 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r

def construct():
    global _T0
    rng = np.random.default_rng(12345)
    
    # Start from a 4x4x4 grid (the seed's arrangement) plus jittered starts.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base = grid[:N].copy()
    base_r = _score(base)
    best = base.copy()
    best_r = base_r
    
    starts = [base]
    # A few random valid starts (jittered grid) to escape the local optimum.
    for _ in range(6):
        s = base + rng.normal(0.0, 0.05, size=base.shape)
        # Keep inside the cube.
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)
        
    # Also try a random cluster start
    s_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    starts.append(s_rand)
    
    for s in starts:
        if time.time() > _DEADLINE:
            break
            
        # Phase 1: Annealing to escape local minima
        a, ar = _anneal(s, T0=0.03, T1=0.001, steps=2000, rng=rng)
        if ar > best_r:
            best_r, best = ar, a
            
        if time.time() > _DEADLINE:
            break
            
        # Phase 2: Polish with soft-min gradient ascent
        p, pr = _polish(a, beta_init=20.0, beta_final=500.0, steps=50)
        if pr > best_r:
            best_r, best = pr, p
            
        if time.time() > _DEADLINE:
            break
            
    return best
