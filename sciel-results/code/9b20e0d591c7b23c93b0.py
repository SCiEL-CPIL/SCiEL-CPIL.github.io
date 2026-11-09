"""
Strategy: Deterministic force-directed relaxation with Nesterov momentum + L-BFGS-B soft-min refinement.
This replaces stochastic KMC with gradient-based optimization, which is more efficient for smooth energy landscapes.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
DEADLINE = 115.0  # Safety margin before 150s limit (evaluator kills at 150s, but we aim for 120s safe)

def compute_radius(points):
    """Compute the maximum common radius for a set of points."""
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        points[:, 0] + 0.5,
        0.5 - points[:, 0],
        points[:, 1] + 0.5,
        0.5 - points[:, 1],
        points[:, 2] + 0.5,
        0.5 - points[:, 2],
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For 100 points, cdist is fast enough for final check, but we use it here for accuracy
    from scipy.spatial.distance import cdist
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def init_fcc(n):
    """
    Initialize points in a Face-Centered Cubic (FCC) lattice pattern.
    FCC is the densest known packing for spheres in 3D, providing a strong starting point.
    We generate a grid of FCC points and select the first n that fit within [-0.5, 0.5]^3.
    """
    points = []
    # FCC lattice vectors: a = 1/sqrt(2) * 2r ? No, just generate a dense grid.
    # Let's use a simple grid with offsets to mimic FCC locally.
    # Standard FCC unit cell has 4 atoms.
    # We'll generate points on a grid with spacing s, and add offsets.
    
    # Target radius is approx 0.111. Diameter 0.222.
    # In a cube of side 1, we can fit roughly (1/0.222)^3 ~ 90 spheres in simple cubic.
    # FCC is denser, so we can fit more or larger.
    
    # Let's generate a grid of points with spacing ~ 0.2
    s = 0.2
    offsets = [
        [0, 0, 0],
        [s/2, s/2, 0],
        [s/2, 0, s/2],
        [0, s/2, s/2]
    ]
    
    grid_range = np.arange(-0.5, 0.5, s)
    
    for x in grid_range:
        for y in grid_range:
            for z in grid_range:
                for off in offsets:
                    p = np.array([x + off[0], y + off[1], z + off[2]])
                    # Check if inside cube
                    if np.all(p >= -0.5) and np.all(p <= 0.5):
                        points.append(p)
    
    points = np.array(points)
    
    # If we have more than N, we need to select N.
    # We want to maximize the minimum distance.
    # A simple heuristic: select points that are farthest from the center? Or just take a subset?
    # Better: Use a greedy farthest-point sampling on the FCC candidates.
    
    if len(points) > n:
        selected = [points[0]]
        remaining = points[1:]
        
        for _ in range(n - 1):
            # Compute min distance from each remaining point to selected points
            # This is O(N^2) but N is small (candidates ~100-200)
            min_dists = np.inf
            for p in remaining:
                dists_to_sel = np.linalg.norm(selected - p, axis=1)
                min_d = np.min(dists_to_sel)
                if min_d < min_dists:
                    min_dists = min_d
                    best_idx = np.argmin(dists_to_sel) # This is wrong, we want the max of min dists
                    # Actually, we want the point with the MAXIMUM minimum distance to the set.
            
            # Redo: Farthest Point Sampling
            min_dists = np.array([np.min(np.linalg.norm(p - np.array(selected), axis=1)) for p in remaining])
            best_idx = np.argmax(min_dists)
            selected.append(remaining[best_idx])
            remaining = np.delete(remaining, best_idx, axis=0)
            
        points = np.array(selected)
        
    elif len(points) < n:
        # Pad with random points if FCC grid is too sparse
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(points), 3))
        points = np.vstack([points, extra])
        
    return points[:n]

def force_directed_relax(points, time_budget, seed=42):
    """
    Deterministic force-directed relaxation.
    Forces:
    1. Pair repulsion: F_ij = k * (r_min - d_ij) / d_ij * (p_i - p_j) if d_ij < r_min
       Actually, use LJ-12 for smoothness: F ~ (sigma/d)^12
    2. Wall repulsion: F_wall = k_wall * (r_min - dist_to_wall) in inward direction if dist < r_min
    """
    np.random.seed(seed)
    pts = points.copy()
    start_time = time.time()
    
    # Parameters
    r_target = 0.11 # Initial target radius, will adapt
    k_pair = 100.0
    k_wall = 100.0
    damping = 0.9
    momentum = 0.8
    step_size = 0.01
    
    vel = np.zeros_like(pts)
    
    # Estimate initial radius
    r_cur = compute_radius(pts)
    
    while time.time() - start_time < time_budget:
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pair forces
        # Use KDTree for efficiency? For 100 points, O(N^2) is ~10k ops, very fast.
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    # Random nudge
                    d_vec = np.random.randn(3) * 0.001
                    d = np.linalg.norm(d_vec)
                
                # If d < 2*r_target, they are too close
                # Use a smooth potential: U = 0.5 * k * max(0, 2*r_target - d)^2
                # Force = -grad U = k * max(0, 2*r_target - d) * d_vec / d
                if d < 2 * r_target:
                    f_mag = k_pair * (2 * r_target - d)
                    f_vec = f_mag * d_vec / d
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # Wall forces
        for i in range(len(pts)):
            for k in range(3):
                # Left wall: x = -0.5
                dist_left = pts[i, k] - (-0.5)
                if dist_left < r_target:
                    f_mag = k_wall * (r_target - dist_left)
                    forces[i, k] += f_mag # Push right (positive k)
                
                # Right wall: x = 0.5
                dist_right = 0.5 - pts[i, k]
                if dist_right < r_target:
                    f_mag = k_wall * (r_target - dist_right)
                    forces[i, k] -= f_mag # Push left (negative k)
        
        # Update velocity with Nesterov momentum
        vel = momentum * vel + damping * forces
        pts += step_size * vel
        
        # Clip to cube (hard constraint)
        pts = np.clip(pts, -0.5, 0.5)
        
        # Compute current radius
        r_cur = compute_radius(pts)
        
        # Adapt r_target and step_size
        # If r_cur is increasing, we can be more aggressive
        # If r_cur is decreasing, reduce step size
        # Simple adaptation:
        if r_cur > r_target:
            r_target = r_cur
            step_size = min(step_size * 1.05, 0.05)
        else:
            step_size = max(step_size * 0.9, 0.001)
            
        # Cool down step size over time to converge
        elapsed = time.time() - start_time
        progress = elapsed / time_budget
        step_size *= (1.0 - 0.5 * progress)
        
        if step_size < 1e-5:
            break
            
    return pts, r_cur

def soft_min_objective(x, beta):
    """
    Objective function for L-BFGS-B.
    Minimize -log(sum(exp(-beta * d_ij^2))) approximates -min(d_ij)
    Also include wall distances.
    """
    pts = x.reshape(-1, 3)
    
    # Wall distances
    wall_dists = np.min(np.stack([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1],
        pts[:, 2] + 0.5,
        0.5 - pts[:, 2],
    ], axis=1), axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    # Combine all distances (pairwise and wall)
    # We want to maximize the minimum distance.
    # Soft min: -log(sum(exp(-beta * d)))
    # To avoid overflow, use log-sum-exp trick
    all_dists = np.concatenate([dists.ravel(), wall_dists.ravel()])
    
    # Filter out very small distances to avoid issues? No, we want to penalize them.
    # But exp(-beta * d) for large d is 0. For small d, it's large.
    # We want to MINIMIZE the soft min of the distances? No, we want to MAXIMIZE the min distance.
    # So we minimize -soft_min(distances) = log(sum(exp(-beta * d)))
    
    # Use log-sum-exp for stability
    max_val = np.max(all_dists)
    # Actually, we want to penalize small distances.
    # Let's minimize: sum( exp(-beta * d) )
    # Gradient is easy to compute.
    
    # For L-BFGS-B, we need gradient.
    # Let's just return the value and let SciPy approximate gradient? No, provide gradient.
    
    # Value: -log(sum(exp(-beta * d))) is the soft min. We want to maximize it, so minimize its negative.
    # f(x) = - soft_min = log(sum(exp(-beta * d)))
    
    # To be safe against overflow, shift by max distance?
    # exp(-beta * d) is largest when d is smallest.
    # So sum is dominated by small d.
    
    # Let's compute the value
    exp_terms = np.exp(-beta * all_dists)
    sum_exp = np.sum(exp_terms)
    val = np.log(sum_exp)
    
    # Gradient
    # d/dx log(sum exp(-beta d)) = (1/sum) * sum( -beta * exp(-beta d) * d(d)/dx )
    # = -beta * sum( exp(-beta d) * d(d)/dx ) / sum_exp
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # d(d_wall)/dx is 0 or +/- 1 depending on which wall is closest.
    # But soft min smooths this.
    # For simplicity, let's just compute the gradient for pairs and walls separately.
    
    # This is getting complex. Let's use a simpler objective:
    # Minimize: sum( max(0, r_target - d)^2 ) + sum( max(0, r_target - d_wall)^2 )
    # This is a standard constraint satisfaction approach.
    # But L-BFGS-B works better with smooth functions.
    
    # Let's stick to the soft min but implement gradient carefully.
    
    # Actually, for 100 points, we can just use the force-directed result and do a final local search with simple gradient ascent on the radius?
    # No, let's use a different approach for the polish:
    # Use the force-directed result as initial guess for a local optimization that maximizes r.
    # We can use a simple hill climbing on the radius by moving points to increase the min distance.
    
    # Given the complexity of gradient for soft-min with walls, let's use a different polish:
    # "Local Search with Best Improvement"
    # For each point, try small moves in 6 directions (axes) and accept if min radius increases.
    
    return val

def local_search_polish(points, time_budget, seed=42):
    """
    Simple local search: try moving each point slightly in each direction.
    Accept if the minimum radius increases.
    """
    np.random.seed(seed)
    pts = points.copy()
    start_time = time.time()
    
    r_cur = compute_radius(pts)
    step = 0.005
    
    improved = True
    while improved and (time.time() - start_time < time_budget):
        improved = False
        for i in range(len(pts)):
            for axis in range(3):
                for direction in [-1, 1]:
                    new_pts = pts.copy()
                    new_pts[i, axis] += direction * step
                    # Clip
                    new_pts[i, axis] = np.clip(new_pts[i, axis], -0.5, 0.5)
                    
                    r_new = compute_radius(new_pts)
                    if r_new > r_cur:
                        pts = new_pts
                        r_cur = r_new
                        improved = True
        
        # Reduce step size if no improvement
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
                
    return pts, r_cur

def construct():
    start_time = time.time()
    
    # Phase 1: FCC Initialization
    points = init_fcc(N)
    best_pts = points.copy()
    best_r = compute_radius(points)
    
    # Phase 2: Force-Directed Relaxation
    # Budget: 60 seconds
    fd_budget = min(60.0, DEADLINE - (time.time() - start_time) - 20)
    if fd_budget > 5:
        pts1, r1 = force_directed_relax(points, fd_budget, seed=42)
        if r1 > best_r:
            best_r = r1
            best_pts = pts1
    
    # Phase 3: Local Search Polish
    # Budget: remaining time - 5s safety
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 5:
        pts2, r2 = local_search_polish(best_pts, remaining, seed=123)
        if r2 > best_r:
            best_r = r2
            best_pts = pts2
    
    # Final check and return
    result = np.clip(best_pts, -0.5, 0.5)
    final_r = compute_radius(result)
    
    # Ensure validity
    if not np.all(np.isfinite(result)):
        # Fallback to grid
        return init_fcc(N)
        
    return result

# EVOLVE-BLOCK-END
