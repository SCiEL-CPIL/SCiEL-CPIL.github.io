"""Optimized packing of 50 equal spheres in a unit cube."""
import numpy as np
import time

N = 50
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _score(c):
    """Compute the max common radius for centres c (shape N,3)."""
    # pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # distance to walls
    wall = np.min(0.5 - np.abs(c), axis=1)
    return min(min_pair, np.min(wall))

def _gradient_step(c, step):
    """Custom gradient ascent on the max-min objective using soft-min."""
    # pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # wall distances
    wall = 0.5 - np.abs(c)
    wall_min = np.min(wall, axis=1)
    
    # Use soft-min to get smooth gradients
    beta = 50.0
    # For pair distances: gradient pushes apart the closest pair
    # soft-min over pairs
    # We want to maximize min over all constraints
    # Use log-sum-exp trick for soft-min
    
    # Pair soft-min
    pair_vals = dist.reshape(-1)
    pair_idx = np.argwhere(np.triu(np.ones((N, N)), k=1).reshape(-1) > 0)
    pair_vals = dist[np.triu_indices(N, k=1)]
    
    m_pair = np.min(pair_vals)
    # soft min approximation
    exp_pair = np.exp(-beta * (pair_vals - m_pair))
    w_pair = exp_pair / np.sum(exp_pair)
    
    # Wall soft-min
    wall_vals = np.min(wall, axis=1)
    m_wall = np.min(wall_vals)
    exp_wall = np.exp(-beta * (wall_vals - m_wall))
    w_wall = exp_wall / np.sum(exp_wall)
    
    # Combine: we want to increase the minimum
    # Gradient of soft-min w.r.t. c_i
    grad = np.zeros_like(c)
    
    # Pair contribution
    for idx in range(len(w_pair)):
        i, j = np.unravel_index(idx, (N, N))
        if i >= j:
            continue
        w = w_pair[idx]
        diff = c[i] - c[j]
        dist_ij = dist[i, j]
        if dist_ij < 1e-12:
            continue
        # d/d(c_i) of dist_ij = (c_i - c_j) / dist_ij
        # We want to increase the min, so move i away from j and j away from i
        grad[i] += w * diff / dist_ij
        grad[j] -= w * diff / dist_ij
    
    # Wall contribution
    for i in range(N):
        w = w_wall[i]
        # wall = 0.5 - |c_ik|, gradient w.r.t c_ik = -sign(c_ik)
        # We want to increase wall, so move toward center
        grad[i] -= w * np.sign(c[i])
    
    return grad

def construct():
    start_time = time.time()
    
    # 5x5x2 grid starting point
    m1, m2, m3 = 5, 5, 2
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    c = grid[:N].copy()
    
    best_c = c.copy()
    best_score = _score(c)
    
    # Local search with decreasing step size
    step = 0.02
    patience = 0
    max_patience = 200
    
    while time.time() - start_time < TIME_LIMIT:
        # Compute gradient
        grad = _gradient_step(c, step)
        
        # Check if we should reduce step
        if patience >= max_patience:
            step *= 0.8
            patience = 0
            if step < 1e-6:
                break
        
        # Take step
        c_new = c + step * grad
        
        # Clamp to valid region
        c_new = np.clip(c_new, -0.5, 0.5)
        
        score_new = _score(c_new)
        
        if score_new > best_score:
            best_c = c_new.copy()
            best_score = score_new
            patience = 0
            c = c_new
        else:
            patience += 1
            # Try smaller step
            c_trial = c + 0.5 * step * grad
            c_trial = np.clip(c_trial, -0.5, 0.5)
            score_trial = _score(c_trial)
            if score_trial > best_score:
                best_c = c_trial.copy()
                best_score = score_trial
                c = c_trial
                patience = 0
            else:
                step *= 0.95
    
    # Final polish: coordinate descent
    if time.time() - start_time < TIME_LIMIT - 5:
        c = best_c.copy()
        for _ in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            improved = False
            for i in range(N):
                for k in range(3):
                    # Try moving c[i,k] in small increments
                    for delta in [0.001, -0.001, 0.0005, -0.0005]:
                        c_old = c[i, k]
                        c[i, k] = np.clip(c_old + delta, -0.5, 0.5)
                        s = _score(c)
                        if s > best_score:
                            best_score = s
                            best_c = c.copy()
                            improved = True
                        else:
                            c[i, k] = c_old
            if not improved:
                break
    
    return best_c
