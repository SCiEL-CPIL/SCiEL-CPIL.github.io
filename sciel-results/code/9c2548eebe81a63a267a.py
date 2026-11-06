import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _fcc_lattice():
    """
    Generate a Face-Centered Cubic (FCC) lattice packing.
    FCC has a packing density of ~74%, higher than the simple cubic grid.
    We generate a dense cluster and trim/pad to N=100.
    """
    # FCC basis vectors
    a = 0.3 # approximate spacing for 100 spheres in unit cube
    # Standard FCC points in a unit cell:
    # (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # plus translations by integer lattice vectors.
    
    # Generate a grid of FCC points
    # We need roughly 4^(1/3) * 5 ~ 8x8x8 lattice points?
    # Volume of cube is 1. Volume of sphere is (4/3)pi r^3.
    # If r ~ 0.11, V_s ~ 0.0055. 100 * 0.0055 = 0.55.
    # FCC density is 0.74. So 0.55/0.74 ~ 0.74 volume occupied by "cells"?
    # Let's just generate a large set and pick the 100 closest to center.
    
    points = []
    # Range from -2 to 2 in lattice units (scaled by 'a')
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                base = np.array([i, j, k]) * a
                # 4 points per cell
                points.append(base + np.array([0, 0, 0]) * a)
                points.append(base + np.array([0.5, 0.5, 0]) * a)
                points.append(base + np.array([0.5, 0, 0.5]) * a)
                points.append(base + np.array([0, 0.5, 0.5]) * a)
                
    points = np.array(points)
    
    # Keep only points inside the cube [-0.5, 0.5]^3
    # Actually, we want them inside, so filter strictly
    mask = np.all(np.abs(points) < 0.45, axis=1) # slight margin
    points = points[mask]
    
    # If we have more than N, pick the N closest to the origin to maintain symmetry
    if len(points) > N:
        dists_to_origin = np.sum(points**2, axis=1)
        idx = np.argsort(dists_to_origin)
        points = points[idx[:N]]
    elif len(points) < N:
        # Pad with random points if necessary (unlikely with this grid)
        random_pts = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        points = np.vstack([points, random_pts])
        
    # Center the points? The FCC lattice is centered at 0.
    # Ensure they are well distributed.
    return points

def _nelder_mead_step(pts, max_iter=200):
    """
    Perform a Nelder-Mead optimization step.
    Objective: Minimize -score.
    Since score is non-differentiable (min function), NM is robust.
    """
    # Flatten points
    x0 = pts.flatten()
    
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        # Clip to valid range to avoid invalid scores (though NM might wander)
        # We rely on the score function returning 0 for invalid, which acts as a barrier
        # but NM might get stuck at 0. 
        # Better: penalize heavily if out of bounds.
        if np.any(x < -0.5) or np.any(x > 0.5):
            return 10.0 # High penalty
        return -_calc_score(x)
        
    # NM requires a good initial simplex.
    # We use the current points as the initial vertex.
    # Perturb slightly to create the simplex?
    # scipy's NM generates the simplex automatically based on x0.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        new_pts = res.x.reshape(N, 3)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        return new_pts
    except Exception:
        return pts

def _simulated_annealing_min(pts, start_temp=0.05, end_temp=0.005, n_iter=2000, time_limit=20.0):
    """
    Simulated Annealing specifically targeting the MIN distance.
    This is different from the soft-min gradient approach.
    It accepts moves that increase the min-distance (decrease energy) or
    occasionally accepts worse moves to escape local minima.
    """
    x = pts.copy()
    best_x = x.copy()
    best_score = _calc_score(x)
    current_score = best_score
    
    start_time = time.time()
    np.random.seed(42)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        t = i / n_iter
        # Geometric cooling
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Perturbation: move a random subset of points
        # Moving all points is too expensive and noisy.
        # Move 10-20% of points.
        n_move = max(1, int(N * 0.1))
        indices = np.random.choice(N, n_move, replace=False)
        
        dx = np.random.normal(0, temp, (n_move, 3))
        
        x_new = x.copy()
        x_new[indices] += dx
        x_new = np.clip(x_new, -0.5, 0.5)
        
        new_score = _calc_score(x_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            x = x_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Boltzmann acceptance
            delta = current_score - new_score # positive delta means new is worse
            # We want to accept if exp(-delta/temp) > random
            # Since score is radius, delta is small (e.g., 0.001)
            # temp is around 0.01-0.05
            if np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                x = x_new
                current_score = new_score
                
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initialize with FCC lattice
    # The FCC lattice is a much better starting point than a simple grid
    # because it already respects the "non-overlapping" constraint more naturally
    # and has higher density.
    pts = _fcc_lattice()
    
    # Ensure valid
    pts = np.clip(pts, -0.5, 0.5)
    best_x = pts.copy()
    best_score = _calc_score(best_x)
    
    # 2. Nelder-Mead refinement
    # NM is good at exploiting the local structure of the min-distance landscape.
    if time.time() < deadline - 30:
        refined_pts = _nelder_mead_step(best_x, max_iter=300)
        refined_score = _calc_score(refined_pts)
        if refined_score > best_score:
            best_score = refined_score
            best_x = refined_pts
    
    # 3. Simulated Annealing to escape local minima
    # We run multiple short SA runs with different seeds/temperatures
    # to explore the configuration space.
    sa_configs = [
        {"start_temp": 0.08, "end_temp": 0.01, "n_iter": 1500, "time_limit": 20.0},
        {"start_temp": 0.05, "end_temp": 0.005, "n_iter": 2000, "time_limit": 20.0},
        {"start_temp": 0.10, "end_temp": 0.02, "n_iter": 1000, "time_limit": 15.0},
    ]
    
    for config in sa_configs:
        if time.time() > deadline - 20:
            break
            
        sa_x, sa_score = _simulated_annealing_min(best_x, **config)
        if sa_score > best_score:
            best_score = sa_score
            best_x = sa_x
            
            # After a good SA jump, do a quick NM polish
            if time.time() < deadline - 10:
                polished_pts = _nelder_mead_step(best_x, max_iter=100)
                polished_score = _calc_score(polished_pts)
                if polished_score > best_score:
                    best_score = polished_score
                    best_x = polished_pts
    
    # 4. Final NM polish
    if time.time() < deadline - 5:
        final_pts = _nelder_mead_step(best_x, max_iter=200)
        final_score = _calc_score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_x = final_pts
    
    # Ensure validity
    best_x = np.clip(best_x, -0.5, 0.5)
    if not np.all(np.isfinite(best_x)):
        best_x = _fcc_lattice().copy()
        best_x = np.clip(best_x, -0.5, 0.5)
        
    return best_x
