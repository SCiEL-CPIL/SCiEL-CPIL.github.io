import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Computes the score for a given set of centers.
    Score = min(min_pair_dist / 2, min_wall_dist)
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin(v, beta):
    """
    Computes the soft-min of vector v with parameter beta.
    """
    min_v = np.min(v)
    # Subtract min for numerical stability
    exp_terms = np.exp(-beta * (v - min_v))
    log_sum = np.log(np.sum(exp_terms))
    return min_v - (1.0 / beta) * log_sum

def _objective(x, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes -softmin(gaps, beta).
    """
    centers = x.reshape(N, 3)
    
    # Pairwise gaps
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_gaps = dists / 2.0
    
    # Wall gaps
    wall_gaps = 0.5 - np.abs(centers)
    
    # Combine all gaps into one vector
    v = np.concatenate([pair_gaps.flatten(), wall_gaps.flatten()])
    
    # Maximize softmin -> Minimize -softmin
    return -_softmin(v, beta)

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_centers = None
    best_score = -1.0
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Define restarts
    # We will use a mix of random and perturbed grid initializations
    
    for i in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial guess
        if i % 3 == 0:
            # Random uniform
            x0 = np.random.uniform(-0.5, 0.5, (N, 3)).flatten()
        elif i % 3 == 1:
            # Perturbed Grid (4x4x4)
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(len(grid))
            grid = grid[idx[:N]]
            # Add small noise
            grid += np.random.normal(0, 0.02, grid.shape)
            grid = np.clip(grid, -0.49, 0.49)
            x0 = grid.flatten()
        else:
            # Rotated Grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(len(grid))
            grid = grid[idx[:N]]
            
            # Apply random rotation
            # Generate random rotation matrix
            # Using scipy.stats? Or simple random orthogonal matrix
            # Simple: random normal, QR decomposition
            A = np.random.normal(0, 1, (3, 3))
            Q, R = np.linalg.qr(A)
            # Ensure proper rotation (det = 1)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            grid = grid @ Q.T
            
            # Center the rotated grid? It's already centered.
            # But rotation might push points out of bounds.
            # Scale down to fit?
            # The grid points are in [-0.5, 0.5]. Rotation keeps them in sphere of radius sqrt(3)/2 ~ 0.866.
            # They might be outside the cube [-0.5, 0.5].
            # We should clip or scale.
            # Scaling is better to keep relative structure.
            max_coord = np.max(np.abs(grid))
            if max_coord > 0.5:
                grid = grid * (0.5 / max_coord)
            
            x0 = grid.flatten()
            
        # Optimization Stages
        # Stage 1: Low Beta (Smooth)
        try:
            res1 = minimize(_objective, x0, args=(10.0,), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
        except Exception:
            continue
            
        if time.time() - start_time > time_limit:
            break
            
        # Stage 2: Medium Beta
        try:
            res2 = minimize(_objective, res1.x, args=(100.0,), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9})
        except Exception:
            continue
            
        if time.time() - start_time > time_limit:
            break
            
        # Stage 3: High Beta (Polish)
        try:
            res3 = minimize(_objective, res2.x, args=(1000.0,), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        except Exception:
            continue
            
        # Evaluate Result
        centers = res3.x.reshape(N, 3)
        
        # Check validity (finite values)
        if not np.all(np.isfinite(centers)):
            continue
            
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
        # Optional: Early termination if we reach a very high score
        # The best known is ~0.13595
        if best_score > 0.1359:
            # We can try to polish further or just break?
            # Let's keep searching to see if we can beat it, but maybe reduce restarts?
            # For now, just keep going until time limit.
            pass

    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
