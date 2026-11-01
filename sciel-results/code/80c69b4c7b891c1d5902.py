"""Rewritten program: force-directed packing with simulated annealing + L-BFGS-B polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
HALF = 0.5

def _score(pts):
    """Compute the packing radius (score) for a set of points."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check boundaries
    if np.any(pts < -HALF + 1e-15) or np.any(pts > HALF - 1e-15):
        # Clamp to boundary for safety, but score uses actual positions
        pass
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    # Distance to boundary
    min_boundary = np.min(np.minimum(
        np.minimum(pts[:, 0] + HALF, pts[:, 1] + HALF),
        np.minimum(HALF - pts[:, 0], HALF - pts[:, 1])
    ))
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _energy(pts, beta=50.0):
    """
    Soft-min energy: penalize pairs closer than 1 and points too close to boundary.
    Uses log-sum-exp style for smoothness.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Pairwise: penalize when dist < 1 (i.e., circles overlap with radius 0.5)
    # Use soft min: -log(sum(exp(-beta * dist))) approximates -min(dist)
    # We want to maximize min dist, so minimize -min dist
    # Soft min of dist: m = -1/beta * log(sum(exp(-beta*dist)))
    # We want to push dist up, so energy = sum of exp(-beta*(dist-1)) for pairs where dist<1
    # Actually simpler: energy = sum over pairs of max(0, 1-dist)^2 + boundary terms
    # But for smoothness use: sum(exp(-beta*(dist-1))) for all pairs (large when dist<1)
    
    # For efficiency with N=200, pairwise is 200*199/2 = 19900 pairs
    # dist matrix is 200x200
    
    # Soft min surrogate for pairwise
    # E_pair = sum_{i<j} exp(-beta * max(0, 1 - dist_ij))
    # This is smooth and sharply penalizes overlaps
    
    # Boundary energy
    # E_bound = sum_i sum_k exp(-beta * max(0, 0.5 - |c_ik|))
    
    # Compute pairwise energy
    # For pairs where dist < 1, we get significant energy
    pair_energy = np.sum(np.exp(-beta * np.maximum(0, 1.0 - dist)))
    
    # Boundary energy
    bound_dists = np.minimum(
        np.minimum(pts[:, 0] + HALF, pts[:, 1] + HALF),
        np.minimum(HALF - pts[:, 0], HALF - pts[:, 1])
    )
    bound_energy = np.sum(np.exp(-beta * np.maximum(0, HALF - bound_dists)))
    
    return pair_energy + bound_energy


def _gradient(pts, beta=50.0):
    """Gradient of the energy function."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Gradient of E_pair = sum_{i<j} exp(-beta * max(0, 1 - dist_ij))
    # dE/d(pts_i) = sum_{j != i} exp(-beta * max(0, 1 - dist_ij)) * (-beta) * d(max(0,1-dist_ij))/d(pts_i)
    # d(max(0, 1-dist))/d(pts_i) = -sign(pts_i - pts_j) / dist * (1 if dist < 1 else 0)
    
    grad = np.zeros_like(pts)
    
    # For pairs where dist < 1
    mask = dist < 1.0
    np.fill_diagonal(mask, False)
    
    # Weight matrix: w_ij = exp(-beta * (1 - dist_ij)) for dist < 1, else 0
    w = np.where(mask, np.exp(-beta * (1.0 - dist)), 0.0)
    
    # grad_i = sum_j w_ij * beta * (pts_i - pts_j) / dist_ij
    # = beta * sum_j w_ij * (pts_i - pts_j) / dist_ij
    
    # (pts_i - pts_j) / dist_ij = diff[i,j] / dist[i,j]
    unit_vec = diff / dist[:, :, np.newaxis]
    np.fill_diagonal(unit_vec[:, :, 0], 0.0)
    np.fill_diagonal(unit_vec[:, :, 1], 0.0)
    
    grad = beta * np.sum(w[:, :, np.newaxis] * unit_vec, axis=1)
    
    # Boundary gradient
    bound_dists = np.minimum(
        np.minimum(pts[:, 0] + HALF, pts[:, 1] + HALF),
        np.minimum(HALF - pts[:, 0], HALF - pts[:, 1])
    )
    
    # For boundary: E_bound = sum_i exp(-beta * max(0, 0.5 - bound_dist_i))
    # dE/d(pts_ik) = exp(-beta * max(0, 0.5 - bd_i)) * (-beta) * d(max(0, 0.5-bd_i))/d(pts_ik)
    # d(max(0, 0.5-bd))/d(pts_ik) = -sign(pts_ik - boundary) if bd < 0.5 else 0
    
    bound_mask = bound_dists < HALF
    bound_w = np.where(bound_mask, np.exp(-beta * (HALF - bound_dists)), 0.0)
    
    # For each point, determine which boundary is closest
    for i in range(N):
        if not bound_mask[i]:
            continue
        x, y = pts[i]
        # Distances to each wall
        dl = x + HALF
        dr = HALF - x
        db = y + HALF
        dt = HALF - y
        min_d = min(dl, dr, db, dt)
        if min_d < HALF:
            w_i = bound_w[i]
            if dl == min_d:
                grad[i, 0] -= beta * w_i
            if dr == min_d:
                grad[i, 0] += beta * w_i
            if db == min_d:
                grad[i, 1] -= beta * w_i
            if dt == min_d:
                grad[i, 1] += beta_i if False else beta * w_i
    
    return grad


def _lbfgs_polish(pts, max_iter=200, beta=50.0):
    """Use L-BFGS-B to polish the packing."""
    def objective(x):
        pts = x.reshape(N, 2)
        return _energy(pts, beta)
    
    def gradient(x):
        pts = x.reshape(N, 2)
        g = _gradient(pts, beta)
        return g.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-HALF, HALF)] * (2 * N)
    
    result = minimize(
        objective,
        pts.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return result.x.reshape(N, 2)


def _simulated_annealing(pts, n_steps=5000, beta_start=10.0, beta_end=50.0, 
                          temp_start=0.05, temp_end=0.001, deadline=None):
    """Multi-scale simulated annealing on the force-directed layout."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_energy = _energy(pts, beta_start)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Annealing schedule
        t = step / max(n_steps - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        temp = temp_start + (temp_end - temp_start) * t
        
        # Generate perturbation: move a few random points
        n_move = np.random.randint(1, 6)
        idx = np.random.choice(N, n_move, replace=False)
        noise = np.random.normal(0, temp, size=(n_move, 2))
        new_pts = pts.copy()
        new_pts[idx] += noise
        
        # Clamp to boundary
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_energy = _energy(new_pts, beta)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(temp, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            
            # Track best
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Phase 1: Simulated annealing
    sa_pts, sa_score = _simulated_annealing(
        pts, 
        n_steps=8000,
        beta_start=5.0, beta_end=40.0,
        temp_start=0.08, temp_end=0.005,
        deadline=deadline
    )
    if sa_score > best_score:
        best_pts = sa_pts
        best_score = sa_score
    
    # Phase 2: L-BFGS-B polish
    if time.time() < deadline - 5.0:
        polished = _lbfgs_polish(best_pts, max_iter=300, beta=50.0)
        pol_score = _score(polished)
        if pol_score > best_score:
            best_pts = polished
            best_score = pol_score
    
    # Phase 3: Another round of SA with smaller temperature + polish
    if time.time() < deadline - 10.0:
        sa_pts2, sa_score2 = _simulated_annealing(
            best_pts,
            n_steps=5000,
            beta_start=20.0, beta_end=60.0,
            temp_start=0.02, temp_end=0.001,
            deadline=deadline
        )
        if sa_score2 > best_score:
            best_pts = sa_pts2
            best_score = sa_score2
    
    # Final polish
    if time.time() < deadline - 3.0:
        polished2 = _lbfgs_polish(best_pts, max_iter=200, beta=60.0)
        pol_score2 = _score(polished2)
        if pol_score2 > best_score:
            best_pts = polished2
            best_score = pol_score2
    
    return best_pts
