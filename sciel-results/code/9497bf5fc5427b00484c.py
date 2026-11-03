"""
Graph-Based Greedy Independent Set + Simulated Annealing Packer.

Strategy:
1. Discretize the cube into a dense lattice (e.g., 10x10x10 or 12x12x12).
2. For a target radius r, determine which lattice points are "compatible" 
   (distance >= 2r) and "valid" (margin >= r).
3. Use a Greedy Maximal Independent Set algorithm to select N=100 points 
   from this lattice. This provides a good initial configuration that 
   respects the non-overlapping constraint roughly.
4. Refine the positions using Simulated Annealing (SA) with a smooth 
   objective function (Log-Sum-Exp approximation of the minimum radius).
   SA allows for larger jumps and escape from local minima compared to 
   L-BFGS, which can get trapped in shallow local optima of the non-smooth 
   min-distance landscape.
5. Final polish with a few iterations of L-BFGS on the smooth objective 
   to converge to a local optimum.

This approach leverages the discrete structure to find a diverse set of 
starting points and uses stochastic search to explore the continuous space 
more broadly than pure gradient descent.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = CUBE_HALF - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _smooth_min_radius(points, beta=100.0):
    """
    Calculate the negative of the smooth approximation of the minimum radius.
    We want to maximize r = min(min_dist/2, min_margin).
    This is equivalent to minimizing -r.
    Using Log-Sum-Exp: 
    min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    So -min(x) ≈ 1/beta * log(sum(exp(-beta * x)))
    
    To handle large beta and avoid overflow/underflow, we subtract the min first.
    """
    pts = np.asarray(points, dtype=np.float64)
    margin = CUBE_HALF - np.abs(pts).max(axis=1)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    dists_half = dists / 2.0
    
    # Combine all constraints
    cons = np.concatenate([margin, dists_half.flatten()])
    
    # Soft min approximation
    x_min = cons.min()
    shifted = cons - x_min
    # exp(-beta * shifted) is <= 1, so safe from overflow. 
    # If shifted is large, exp is 0.
    exp_terms = np.exp(-beta * shifted)
    s = np.sum(exp_terms)
    soft_min_val = x_min - (1.0 / beta) * np.log(s)
    
    return -soft_min_val

def _lattice_init(radius_target, lattice_size=12):
    """
    Generate a set of N points using a greedy independent set on a lattice.
    """
    # Create lattice points
    # We want points inside the cube with margin >= radius_target
    # So they must be in [-0.5 + r, 0.5 - r]
    r = radius_target
    if r >= 0.5:
        r = 0.49
    
    # Lattice spacing
    # We define lattice indices 0..L-1
    # Coordinate: -0.5 + r + i * (1 - 2r) / (L - 1)
    # But for independent set, it's better to just generate a dense grid of candidates
    # and filter.
    
    # Let's use a fixed dense grid of candidates, say 15x15x15
    L = 15
    coords = np.linspace(-0.5 + r, 0.5 - r, L)
    lattice_points = np.array([(x, y, z) for z in coords for y in coords for x in coords])
    
    # Filter valid points (margin >= r) - already done by linspace range, but check just in case
    # Actually, linspace ensures they are within [-0.5+r, 0.5-r], so margin is exactly r at boundaries.
    # We need distance between any two selected points >= 2r.
    
    selected = []
    # Shuffle lattice points to break symmetry and allow random restarts
    idx = np.arange(len(lattice_points))
    np.random.shuffle(idx)
    
    for i in idx:
        pt = lattice_points[i]
        # Check if this point is far enough from all selected points
        if len(selected) == 0:
            selected.append(pt)
        else:
            sel_arr = np.array(selected)
            dists = np.linalg.norm(sel_arr - pt, axis=1)
            if np.min(dists) >= 2 * r - 1e-9:
                selected.append(pt)
        
        if len(selected) >= N:
            break
            
    if len(selected) < N:
        # If we couldn't find enough, return what we have padded with random valid points
        # This shouldn't happen for reasonable r, but handle gracefully
        while len(selected) < N:
            # Random point in valid region
            pt = np.random.uniform(-0.5 + r, 0.5 - r, 3)
            selected.append(pt)
            
    return np.array(selected[:N])

def _simulated_annealing(points, start_temp=0.05, end_temp=0.001, steps=5000, beta=50.0):
    """
    Simulated Annealing to maximize the smooth minimum radius.
    """
    pts = points.copy()
    current_score = _score(pts)
    current_obj = _smooth_min_radius(pts, beta=beta)
    
    best_pts = pts.copy()
    best_obj = current_obj
    best_score = current_score
    
    # Pre-compute step size schedule
    # Linear cooling
    temps = np.linspace(start_temp, end_temp, steps)
    
    for t in range(steps):
        temp = temps[t]
        
        # Generate a random move
        # Perturb a random subset of points
        num_points_to_move = np.random.randint(1, 10)
        indices = np.random.choice(N, num_points_to_move, replace=False)
        
        new_pts = pts.copy()
        # Gaussian perturbation
        noise = np.random.normal(0, temp, (num_points_to_move, 3))
        new_pts[indices] += noise
        
        # Keep inside cube (hard constraint for validity, though soft obj handles it)
        # If a point goes outside, it's invalid. 
        # We can either reject or clip. Clipping distorts the search. 
        # Let's check validity.
        if np.any(np.abs(new_pts) > 0.5):
            continue
            
        new_obj = _smooth_min_radius(new_pts, beta=beta)
        delta_obj = new_obj - current_obj
        
        # Metropolis criterion
        if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / temp):
            pts = new_pts
            current_obj = new_obj
            current_score = _score(pts)
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
                best_score = current_score
                
    return best_pts, best_score

def _polish_lbfags(points, maxiter=100):
    """Final polish with L-BFGS-B on the smooth objective."""
    def obj(x):
        return _smooth_min_radius(x.reshape(N, 3), beta=50.0)
        
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < obj(x0):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Lattice-based initialization with varying target radii
    # Try a range of target radii to find a good skeleton
    target_radii = [0.11, 0.111, 0.1113, 0.11138, 0.112]
    
    for r_target in target_radii:
        if time.time() > deadline - 30:
            break
            
        # Generate lattice-based start
        pts = _lattice_init(r_target, lattice_size=15)
        
        # Refine with Simulated Annealing
        sa_pts, sa_score = _simulated_annealing(pts, start_temp=0.03, end_temp=0.001, steps=3000)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
        # Polish the SA result
        if time.time() < deadline - 10:
            polished_pts = _polish_lbfags(sa_pts, maxiter=50)
            pol_score = _score(polished_pts)
            
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished_pts.copy()

    # Phase 2: If we have time, do a few more random restarts from the best solution
    # with small perturbations and SA
    while time.time() < deadline - 20:
        if best_points is None:
            break
            
        # Perturb best solution
        noise = np.random.normal(0, 0.01, (N, 3))
        perturbed = best_points + noise
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Short SA
        sa_pts, sa_score = _simulated_annealing(perturbed, start_temp=0.01, end_temp=0.0005, steps=2000)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
        # Polish
        if time.time() < deadline - 5:
            polished_pts = _polish_lbfags(sa_pts, maxiter=30)
            pol_score = _score(polished_pts)
            
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished_pts.copy()

    # Final Polish on the absolute best
    if best_points is not None and time.time() < deadline - 2:
        final_pts = _polish_lbfags(best_points, maxiter=100)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score

    if best_points is None:
        # Fallback to grid
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.linspace(-0.375, 0.375, 4)
        best_points = np.array([(x, y, z) for z in zs for y in ys for x in xs])

    return best_points

# EVOLVE-BLOCK-END
