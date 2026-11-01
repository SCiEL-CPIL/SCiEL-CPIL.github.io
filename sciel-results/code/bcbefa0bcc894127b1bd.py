"""Improved packing: vectorized repulsion annealing + Metropolis MC + polish."""
import numpy as np
import time

N = 50
DEADLINE = 110.0  # safety margin before 120s limit
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _deadline_exceeded(margin=5.0):
    return _elapsed() > DEADLINE - margin


def _min_dist(c):
    """Compute the minimum pairwise distance (not halved)."""
    n = c.shape[0]
    if n < 2:
        return np.inf
    # Compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)


def _boundary_dist(c):
    """Compute the minimum distance from any center to the cube boundary."""
    # Cube is [-0.5, 0.5]^3
    return np.min(0.5 - np.abs(c))


def _score(c):
    """Compute the actual score (radius)."""
    if not np.all(np.isfinite(c)):
        return 0.0
    if c.shape != (N, 3):
        return 0.0
    if np.any(np.abs(c) > 0.5):
        return 0.0
    md = _min_dist(c)
    bd = _boundary_dist(c)
    return min(md, bd) / 2.0


def _soft_min_obj(c, beta):
    """
    Smooth surrogate: minimize a soft-max of the negative of the minimum distance.
    We want to maximize r = min(pairwise_dists/2, boundary_dists).
    Equivalent to minimizing -r, but we use a soft version.
    
    Actually, let's think differently. We want to maximize the minimum distance.
    Let d_ij be pairwise distances, d_i be boundary distances.
    r = min(all d's) / 2.
    
    Soft-min of the distances: -log(sum(exp(-d/beta))) * beta approximates min(d) as beta->0.
    We want to maximize this, so minimize the negative.
    """
    n = c.shape[0]
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    bd = 0.5 - np.abs(c)  # shape (N, 3)
    bd_min = np.min(bd, axis=1)  # shape (N,)
    
    # Collect all distances
    all_d = np.concatenate([dist[dist < np.inf].flatten(), bd_min])
    
    # Soft min: -beta * log(sum(exp(-d/beta)))
    # For numerical stability
    d_min = np.min(all_d)
    exp_terms = np.exp(-(all_d - d_min) / beta)
    soft_min_val = d_min - beta * np.log(np.sum(exp_terms))
    
    return -soft_min_val  # We minimize this (so maximize soft_min)


def _soft_min_grad(c, beta):
    """Gradient of the soft-min objective."""
    n = c.shape[0]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    bd = 0.5 - np.abs(c)
    bd_min = np.min(bd, axis=1)
    
    all_d = np.concatenate([dist[dist < np.inf].flatten(), bd_min])
    
    d_min = np.min(all_d)
    exp_terms = np.exp(-(all_d - d_min) / beta)
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(c)
    
    # Pairwise contribution
    # dist[i,j] = |c_i - c_j|, d/d c_i dist[i,j] = (c_i - c_j) / dist[i,j]
    # We want d/d c_i of -soft_min = d/d c_i of [beta * log(sum(exp(-d/beta)))] - d_min (d_min is constant w.r.t. c)
    # Actually, let me recompute.
    
    # soft_min = d_min - beta * log(sum(exp(-(d - d_min)/beta)))
    # d(soft_min)/d c_i = sum_j weights_j * d(-d_ij)/d c_i  (for pairwise)
    # = -sum_j weights_j * (c_i - c_j)/d_ij
    
    # Let's just do it numerically for safety, or compute properly.
    
    # For pairwise: contribution to grad[i] from pair (i,j) is -weights_pair[i,j] * (c_i - c_j)/d_ij
    # For boundary: contribution to grad[i] from boundary k is -weights_bd[i,k] * d(bd_ik)/d c_ik
    
    # This is getting complex. Let's use finite differences for the polish, or a simpler approach.
    # Actually, for the MC phase we don't need gradients. For polish, let's use a direct approach.
    
    return None  # We'll use finite differences or a different approach for polish


def _repulsion_anneal(n_iter=5000, seed=42):
    """
    Vectorized repulsion annealing.
    Spheres repel each other and are attracted to the center.
    """
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, (N, 3))
    
    best_c = c.copy()
    best_score = _score(c)
    
    # Annealing: start with strong repulsion, decrease
    for it in range(n_iter):
        if _deadline_exceeded():
            break
        
        # Temperature annealing
        t = 1.0 - it / n_iter
        temp = 0.01 * (1.0 + 10.0 * t)
        
        # Compute forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-10)
        np.fill_diagonal(dist, 1.0)  # Avoid division by zero
        
        # Repulsive force: F_ij = diff_ij / dist_ij^2 (normalized)
        # Only consider close pairs for efficiency
        mask = dist < 0.5
        force = np.zeros_like(c)
        
        # Vectorized repulsion
        inv_dist2 = np.where(mask, 1.0 / (dist * dist + 1e-10), 0.0)
        force = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
        
        # Boundary force: push inward if too close to wall
        for k in range(3):
            close_to_wall = np.abs(c[:, k]) > 0.4
            force[close_to_wall, k] -= np.sign(c[close_to_wall, k]) * 0.5
        
        # Center attraction (weak)
        force -= 0.01 * c
        
        # Update
        step = temp * 0.1
        c_new = c + step * force
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Accept with Metropolis criterion
        score_new = _score(c_new)
        if score_new > best_score:
            best_c = c_new.copy()
            best_score = score_new
        
        if score_new >= best_score or rng.random() < np.exp((score_new - best_score) / (temp * 0.01)):
            c = c_new
        
        # Periodically reheat
        if it % 500 == 0 and it > 0:
            c += rng.normal(0, 0.05, (N, 3))
            c = np.clip(c, -0.5, 0.5)
    
    return best_c, best_score


def _metropolis_mc(c, n_iter=20000, seed=123):
    """
    Metropolis Monte Carlo with temperature annealing.
    """
    rng = np.random.RandomState(seed)
    best_c = c.copy()
    best_score = _score(c)
    
    for it in range(n_iter):
        if _deadline_exceeded():
            break
        
        # Temperature annealing
        t = 1.0 - it / n_iter
        temp = 0.005 * (1.0 + 5.0 * t)
        step_size = 0.02 * (1.0 + 2.0 * t)
        
        # Propose a move for a random sphere
        i = rng.randint(N)
        c_new = c.copy()
        c_new[i] += rng.normal(0, step_size, 3)
        c_new[i] = np.clip(c_new[i], -0.5, 0.5)
        
        score_new = _score(c_new)
        
        if score_new > best_score:
            best_c = c_new.copy()
            best_score = score_new
        
        if score_new >= best_score or rng.random() < np.exp((score_new - best_score) / temp):
            c = c_new
    
    return best_c, best_score


def _polish(c, n_iter=500, tol=1e-10):
    """
    Final high-precision polish using coordinate descent.
    """
    best_c = c.copy()
    best_score = _score(c)
    
    for it in range(n_iter):
        if _deadline_exceeded(margin=10.0):
            break
        
        improved = False
        for i in range(N):
            if _deadline_exceeded(margin=10.0):
                break
            
            # Try to move sphere i in each direction
            for k in range(3):
                for direction in [-1, 1]:
                    # Binary search for optimal position along axis k
                    lo = -0.5
                    hi = 0.5
                    best_pos = c[i, k]
                    best_local_score = best_score
                    
                    for _ in range(30):
                        mid1 = lo + (hi - lo) / 3
                        mid2 = hi - (hi - lo) / 3
                        
                        c_test1 = c.copy()
                        c_test1[i, k] = mid1
                        s1 = _score(c_test1)
                        
                        c_test2 = c.copy()
                        c_test2[i, k] = mid2
                        s2 = _score(c_test2)
                        
                        if s1 > s2:
                            hi = mid2
                            if s1 > best_local_score:
                                best_local_score = s1
                                best_pos = mid1
                        else:
                            lo = mid1
                            if s2 > best_local_score:
                                best_local_score = s2
                                best_pos = mid2
                        
                        if hi - lo < tol:
                            break
                    
                    if best_local_score > best_score:
                        c[i, k] = best_pos
                        best_score = best_local_score
                        best_c = c.copy()
                        improved = True
        
        if not improved:
            break
    
    return best_c, best_score


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Phase 1: Repulsion annealing
    c, score = _repulsion_anneal(n_iter=3000, seed=42)
    
    # Phase 2: Metropolis MC
    if not _deadline_exceeded():
        c, score = _metropolis_mc(c, n_iter=15000, seed=123)
    
    # Phase 3: Polish
    if not _deadline_exceeded(margin=15.0):
        c, score = _polish(c, n_iter=200)
    
    # Ensure valid output
    c = np.clip(c, -0.5, 0.5)
    
    return c
