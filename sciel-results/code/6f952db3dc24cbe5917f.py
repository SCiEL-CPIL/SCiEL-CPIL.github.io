"""Optimized packing of 50 equal spheres in a unit cube via smooth gradient ascent."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s kill


def _score(c):
    """Compute the actual packing radius (min distance/2, min wall distance)."""
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    # Wall distances
    wall = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)


def _soft_min_and_grad(c, beta):
    """Soft-min of all constraints with gradient. Returns (value, grad)."""
    # Pairwise: for each i, soft-min over j!=i of |c_i - c_j|
    # Wall: for each i, soft-min over k of (0.5 - |c_ik|)
    n = len(c)
    # Build pairwise distance matrix
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min of distances for each i (excluding self)
    # log-sum-exp for stability
    # For each row i, soft_min_j dist[i,j]
    # Use: soft_min = -log(sum(exp(-dist/beta))) * beta
    # But we want the gradient of the overall min over all constraints
    
    # Overall soft-min: combine all pair distances and wall distances
    # Let's build a vector of all constraints
    # Pair constraints: dist[i,j] for i<j
    # Wall constraints: 0.5 - |c[i,k]| for all i,k
    
    # For efficiency, compute soft-min over all constraints
    # Pair distances (i<j): n*(n-1)/2 values
    # Wall distances: n*3 values
    # Total: 50*49/2 + 150 = 1225 + 150 = 1375 constraints
    
    # Get upper triangle distances
    iu = np.triu_indices(n, 1)
    pair_dists = dist[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)  # shape (n, 3)
    wall_flat = wall_dists.flatten()
    
    # All constraints
    all_c = np.concatenate([pair_dists, wall_flat])
    
    # Soft-min via log-sum-exp
    # soft_min = -beta * log(sum(exp(-c/beta)))
    # For numerical stability
    max_c = np.max(all_c)
    exp_terms = np.exp(-(all_c - max_c) / beta)
    log_sum = np.log(np.sum(exp_terms)) + (max_c - max_c) / beta  # = log(sum(exp(-c/beta)))
    # Actually: sum(exp(-c/beta)) = exp(-max_c/beta) * sum(exp(-(c-max_c)/beta))
    # log(sum(exp(-c/beta))) = -max_c/beta + log(sum(exp(-(c-max_c)/beta)))
    # soft_min = -beta * log(sum(exp(-c/beta))) = max_c - beta*log(sum(exp(-(c-max_c)/beta)))
    
    soft_min_val = max_c - beta * np.log(np.sum(exp_terms))
    
    # Gradient: d(soft_min)/d(all_c[k]) = -exp(-all_c[k]/beta) / sum(exp(-all_c/beta))
    # = -exp_terms[k] / sum(exp_terms)
    weights = exp_terms / np.sum(exp_terms)
    
    # Now propagate gradient back to c
    grad = np.zeros_like(c)
    
    # Pair distance gradient: d|c_i - c_j|/d c_i = (c_i - c_j)/|c_i - c_j|
    # For pair (i,j), contribution to c[i] is weight * (c[i]-c[j])/dist[i,j]
    # contribution to c[j] is weight * (c[j]-c[i])/dist[i,j]
    for idx in range(len(pair_dists)):
        i, j = iu[0][idx], iu[1][idx]
        d = dist[i, j]
        if d > 1e-12:
            w = weights[idx]
            diff_ij = c[i] - c[j]
            grad[i] += w * diff_ij / d
            grad[j] -= w * diff_ij / d
    
    # Wall distance gradient: d(0.5 - |c_ik|)/d c_ik = -sign(c_ik)
    # For wall constraint (i,k), contribution to c[i,k] is weight * (-sign(c[i,k]))
    wall_idx = 0
    for i in range(n):
        for k in range(3):
            w = weights[wall_idx]
            grad[i, k] += w * (-np.sign(c[i, k]))
            wall_idx += 1
    
    return soft_min_val, grad


def _init_bcc():
    """Initialize with BCC lattice points in the cube."""
    # BCC: corners + body center of each subcube
    # For 50 spheres, use a 4x4x4 BCC-like arrangement
    # 4^3 = 64 BCC points (corners + centers), take first 50
    m = 4
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner
                x = (ix + 0.5) / m - 0.5
                y = (iy + 0.5) / m - 0.5
                z = (iz + 0.5) / m - 0.5
                points.append([x, y, z])
                # Body center (offset by half cell)
                x2 = (ix + 1.0) / m - 0.5
                y2 = (iy + 1.0) / m - 0.5
                z2 = (iz + 1.0) / m - 0.5
                points.append([x2, y2, z2])
    points = np.array(points[:N])
    return points


def _init_random(seed):
    """Random initialization with minimum separation."""
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.4, 0.4, size=(N, 3))
    return c


def _gradient_ascent(c, beta_start, beta_end, n_iters, deadline):
    """Run smooth gradient ascent on soft-min radius."""
    n = len(c)
    lr = 0.01
    best_c = c.copy()
    best_val = _score(c)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        val, grad = _soft_min_and_grad(c, beta_start)
        
        # Adaptive learning rate
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            step = lr / gnorm * 0.1
            c = c + step * grad
        
        # Project back into cube
        c = np.clip(c, -0.45, 0.45)
        
        # Check improvement
        if it % 5 == 0:
            cur_val = _score(c)
            if cur_val > best_val:
                best_val = cur_val
                best_c = c.copy()
    
    return best_c, best_val


def _local_polish(c, deadline):
    """Final high-precision polish using direct gradient on exact min."""
    n = len(c)
    best_c = c.copy()
    best_val = _score(c)
    lr = 0.005
    n_iters = 200
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Find the constraining pair or wall
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair_d = np.min(d)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        
        wall = 0.5 - np.abs(c)
        min_wall = np.min(wall)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        
        if min_pair_d / 2.0 < min_wall:
            # Pair constraint is binding
            i, j = min_pair_idx
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad_i = diff / dist
                grad_j = -diff / dist
                c[i] += lr * grad_i
                c[j] += lr * grad_j
        else:
            # Wall constraint is binding
            i, k = min_wall_idx
            c[i, k] += lr * np.sign(c[i, k])  # push away from wall
        
        # Project into cube
        c = np.clip(c, -0.49, 0.49)
        
        cur_val = _score(c)
        if cur_val > best_val:
            best_val = cur_val
            best_c = c.copy()
    
    return best_c, best_val


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multiple strategies
    best_c = None
    best_val = -1.0
    
    # Strategy 1: BCC initialization + gradient ascent
    c_bcc = _init_bcc()
    val = _score(c_bcc)
    if val > best_val:
        best_val = val
        best_c = c_bcc.copy()
    
    if time.time() < deadline - 5:
        c_polished, val2 = _gradient_ascent(c_bcc, 0.05, 0.01, 500, deadline - 10)
        if val2 > best_val:
            best_val = val2
            best_c = c_polished.copy()
    
    # Strategy 2: Random restarts with gradient ascent
    n_restarts = 8
    for seed in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        c_rand = _init_random(seed + 42)
        val = _score(c_rand)
        if val > best_val:
            best_val = val
            best_c = c_rand.copy()
        
        if time.time() < deadline - 10:
            c_polished, val2 = _gradient_ascent(c_rand, 0.05, 0.01, 300, deadline - 10)
            if val2 > best_val:
                best_val = val2
                best_c = c_polished.copy()
    
    # Final polish on best solution
    if best_c is not None and time.time() < deadline - 2:
        c_final, val_final = _local_polish(best_c, deadline - 1)
        if val_final > best_val:
            best_val = val_final
            best_c = c_final.copy()
    
    # Ensure validity: clip strictly inside
    if best_c is not None:
        best_c = np.clip(best_c, -0.499, 0.499)
        # Verify no coincident points
        d = cdist(best_c, best_c)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-10:
            # Add tiny perturbation
            best_c += np.random.RandomState(0).normal(0, 1e-6, best_c.shape)
            best_c = np.clip(best_c, -0.499, 0.499)
    
    return best_c
