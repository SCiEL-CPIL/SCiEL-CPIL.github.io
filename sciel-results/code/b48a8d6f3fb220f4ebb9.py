"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random

N = 50

def construct():
    # Seed random for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initial positions: hexagonal lattice distorted slightly
    points = []
    r_hex = 0.14  # approximate spacing
    sqrt3 = np.sqrt(3.0)
    
    # Generate hex grid
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * r_hex + (j % 2) * r_hex / 2.0
            y = j * r_hex * sqrt3 / 2.0
            if x*x + y*y <= 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, select N closest to center
    if len(points) > N:
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points if needed
        while len(points) < N:
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 0.8)
            points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
    
    pos = points.copy()
    
    # Energy function: Repulsive potential to maximize minimum distance
    # Minimizing sum of 1/d^2 pushes particles apart
    # Minimizing sum of 1/(1-r)^2 pushes particles to boundary
    def energy(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        # Container violations: push to boundary
        dists_to_center = np.sqrt(np.sum(pos**2, axis=1))
        # Avoid division by zero if pos is at origin (unlikely) or outside
        # We want to maximize (1 - dists_to_center), so minimize 1/(1-dists_to_center)
        # If dists_to_center >= 1, this is invalid/infinite
        # Use a safe distance
        safe_dist_to_center = np.clip(dists_to_center, 0, 0.999)
        boundary_term = np.sum(1.0 / (1.0 - safe_dist_to_center)**2)
        
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, 1.0) # Avoid div by zero on diagonal
        
        # Minimize 1/d^2 to maximize d
        # Clip distances to avoid huge values if they overlap significantly
        safe_dist_mat = np.clip(dist_mat, 0.01, 2.0)
        pair_term = np.sum(1.0 / safe_dist_mat**2)
        
        return boundary_term + pair_term

    # Simulated Annealing
    T = 100.0
    T_min = 1e-5
    cooling = 0.999
    max_iter = 50000
    step_size = 0.1
    
    best_pos = pos.copy()
    best_energy = energy(pos.reshape(-1))
    current_energy = best_energy
    
    for it in range(max_iter):
        # Move: random displacement of one particle
        new_pos = pos.copy()
        idx = np.random.randint(0, N)
        new_pos[idx] += np.random.normal(0, step_size, 2)
        
        new_energy = energy(new_pos.reshape(-1))
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Anneal step size
        if it % 5000 == 0:
            step_size *= 0.8

    pos = best_pos

    # Refine with local optimization
    # L-BFGS-B requires gradients or finite differences. Finite differences are slow for 100 vars.
    # We can use BFGS or Nelder-Mead, but L-BFGS-B is good if we provide a smooth function.
    # The 1/d^2 function is smooth enough.
    
    def objective_for_minimize(pos_flat):
        return energy(pos_flat)

    # Use L-BFGS-B
    try:
        result = minimize(objective_for_minimize, pos.reshape(-1), method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        if result.fun < best_energy:
            pos = result.x.reshape(-1, 2)
    except Exception:
        pass

    # Compute actual radius
    pos = pos.reshape(-1, 2)
    
    # Calculate minimum pairwise distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    
    # Calculate minimum distance to boundary
    dists = np.sqrt(np.sum(pos**2, axis=1))
    min_boundary_dist = np.min(1.0 - dists)
    
    # Current radius is min(min_pair_dist/2, min_boundary_dist)
    r_pair = min_pair_dist / 2.0
    r_bound = min_boundary_dist
    
    # Optimization of scale
    # We want to find scale s such that min(s * r_pair, 1 - s * max_dist) is maximized
    # Note: max_dist is the max distance of any center from origin.
    max_dist = np.max(dists)
    
    # Function to evaluate radius at scale s
    def radius_at_scale(s):
        # New pair dist: s * min_pair_dist
        # New boundary dist: 1 - s * max_dist
        # Constraints: s * max_dist <= 1 (centers inside)
        if s * max_dist > 1.0:
            return 0.0
        r_p = s * r_pair
        r_b = 1.0 - s * max_dist
        return min(r_p, r_b)
    
    # Find optimal s
    # The function is concave (min of two linear functions, one increasing, one decreasing)
    # Intersection: s * r_pair = 1 - s * max_dist  => s(r_pair + max_dist) = 1 => s = 1/(r_pair + max_dist)
    s_opt = 1.0 / (r_pair + max_dist)
    
    # Check if s_opt is valid (max_dist * s_opt <= 1 is always true by definition)
    # Also check if we are currently at s=1
    # If s_opt > 1, it means we can scale up.
    # If s_opt < 1, it means we should scale down? 
    # Wait. 
    # r(s) = min(s*r_pair, 1 - s*max_dist)
    # If s=1, r = min(r_pair, 1-max_dist).
    # If we scale UP (s>1), s*r_pair increases, 1-s*max_dist decreases.
    # So we can scale up until they are equal, provided 1-s*max_dist > 0.
    # If s_opt > 1, we can scale up.
    # If s_opt < 1, it implies that at s=1, 1-max_dist > r_pair? 
    # Let's check: s_opt = 1/(r_pair + max_dist).
    # If r_pair + max_dist > 1, then s_opt < 1.
    # This means the "crossing point" is at s < 1.
    # In this case, at s=1, we are on the "decreasing" part of the boundary term?
    # No, if s_opt < 1, then at s=1, s > s_opt, so we are on the side where boundary term is smaller?
    # Let's trace:
    # f1(s) = s * r_pair (increasing)
    # f2(s) = 1 - s * max_dist (decreasing)
    # Intersection at s*.
    # If s* < 1, then for s=1, f1(1) > f2(1). So min is f2(1).
    # To maximize min, we should move s towards s*. Since s* < 1, we should DECREASE s?
    # If we decrease s, f2(s) increases and f1(s) decreases.
    # The max of min(f1, f2) is at s*.
    # So yes, optimal scale is s_opt.
    
    # However, we must ensure s_opt is not too small or large.
    # Also, we should check if s_opt is close to 1.
    
    s_best = s_opt
    # Clamp s to reasonable range
    s_best = np.clip(s_best, 0.5, 1.5)
    
    # Evaluate radius at s_best
    r_final = radius_at_scale(s_best)
    
    # Compare with original s=1
    r_orig = radius_at_scale(1.0)
    
    if r_final > r_orig:
        pos = pos * s_best
        r = r_final
    else:
        r = r_orig
    
    # Final validation and projection if necessary
    dists = np.sqrt(np.sum(pos**2, axis=1))
    if np.any(dists > 1.0):
        mask = dists > 1.0
        pos[mask] = pos[mask] / dists[mask, np.newaxis] * 0.9999
        # Recalculate r
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        dists = np.sqrt(np.sum(pos**2, axis=1))
        min_boundary_dist = np.min(1.0 - dists)
        r = min(min_pair_dist / 2.0, min_boundary_dist)

    return pos
