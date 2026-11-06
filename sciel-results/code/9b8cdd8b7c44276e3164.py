"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Strategy:
    # 1. Construct a "Ring Packing" initial guess.
    #    - Divide the unit disk into concentric rings.
    #    - For each ring, determine the maximum number of circles that fit 
    #      given a target radius r, and place them evenly.
    #    - This provides a much better structured start than random or spiral.
    # 2. Refine using L-BFGS-B to maximize the minimum clearance.
    #    - Objective: Maximize min( min_pair_dist/2, min_boundary_dist )
    #    - We can approximate this by maximizing a smooth function related to the 
    #      smallest distances, or simply using a penalty method on the minimum.
    #    - Given the time constraint, we will do a few iterations of "move the 
    #      worst offender" or a small-scale global optimization.
    
    # --- Stage 1: Ring-based Initialization ---
    
    # We want to find a good initial radius r.
    # Area argument: N * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(N) ~ 0.0577
    # Best known is ~0.053. Let's aim for r ~ 0.053 in the construction.
    
    # We will build the packing ring by ring from outside in or inside out?
    # Inside out is easier to manage counts.
    
    # Let's try to fit as many as possible in rings.
    # Ring k has radius R_k. Number of circles n_k in ring k.
    # Constraint: distance between adjacent circles in ring k >= 2r.
    # 2 * R_k * sin(pi / n_k) >= 2r  =>  R_k >= r / sin(pi / n_k)
    # Also R_k <= 1 - r (boundary constraint)
    # Distance between rings: (R_{k+1} - R_k) >= 2r (approx, actually depends on alignment)
    # A simpler heuristic: 
    # Assume hexagonal-like packing.
    # Let's just pick a target r and see if we can place N circles.
    
    target_r = 0.0531 # Slightly below best known to allow for optimization
    # Actually, let's construct based on a "greedy" ring filling with a slightly 
    # smaller radius to ensure validity, then expand.
    
    # Let's use a specific heuristic for ring counts.
    # For a ring of radius R, max circles m ~ floor(pi / arcsin(r/R))
    
    points = []
    # Start from center?
    # Center point is (0,0).
    # If we place one at center, it uses up space.
    # Usually optimal packings for N=300 don't have a single center point 
    # but a small cluster or empty center. 
    # However, for N=300, the center is likely filled.
    
    # Let's try filling rings.
    # We need to assign radii R_k.
    # Let's assume the rings are packed tightly.
    # R_k approx k * 2r * (something).
    
    # A robust initialization method:
    # 1. Define a grid of potential radii.
    # 2. For each radius, calculate max capacity.
    # 3. Fill until N points are placed.
    
    # To make it simple and fast:
    # Use a spiral but "snapped" to hexagonal lattice positions?
    # No, the prompt says "MATERIALLY DIFFERENT".
    # The current program uses Fermat spiral + Gradient Descent.
    # My strategy: Ring Packing + Local Refinement.
    
    # Let's implement a simple ring packer.
    # We will determine the number of circles per ring.
    # Let r_init be the radius we use for placement.
    # We want to place N circles.
    
    # Heuristic for number of circles in ring i (0-indexed, 0 is center):
    # Ring 0: 1 circle at (0,0)
    # Ring i: n_i circles.
    # Distance from center R_i.
    # R_i approx (2 * i + 1) * r * sqrt(3)/2 ? No.
    # In hex packing, radial distance grows by ~2r*sqrt(3)/2 per ring?
    # Let's just iterate.
    
    # We will place points on rings.
    # We need to choose R_i and n_i such that:
    # 1. 2 * R_i * sin(pi/n_i) >= 2 * r_init  (circles in ring don't overlap)
    # 2. R_i - R_{i-1} >= 2 * r_init (rings don't overlap, approx)
    # 3. R_i + r_init <= 1
    
    # Let's fix r_init = 0.05.
    r_init = 0.05
    
    # Determine rings
    R_prev = 0
    n_prev = 1 # Center
    points_list = [[0.0, 0.0]]
    
    ring_idx = 0
    while len(points_list) < N and time.time() < deadline - 50:
        ring_idx += 1
        # Estimate R for this ring
        # R should be large enough to hold circles, and separated from prev ring.
        # Let's try increasing R until we can fit enough circles to reach N.
        # Or just use a standard spacing.
        # In hex packing, the distance between ring centers is approx 2*r*sqrt(3)/2 = r*sqrt(3).
        # Let's use R_i = R_{i-1} + r_init * 1.732 (approx)
        # But we must also satisfy the angular constraint.
        
        # Let's pick a candidate R
        R_candidate = R_prev + r_init * 1.732 * 0.9 # Slightly tighter to fit more
        
        # Max circles in this ring at R_candidate with radius r_init
        # 2 * R * sin(pi/n) >= 2r  =>  n <= pi / asin(r/R)
        if R_candidate > r_init:
            max_n = int(np.pi / np.arcsin(r_init / R_candidate))
        else:
            max_n = 1
            
        # We want to fill as many as possible, but not exceed max_n
        # Also, we shouldn't put too few if we can put more, to keep density high?
        # Actually, for optimal packing, we usually fill rings completely 
        # except the last one.
        
        # Let's just take max_n
        n_to_add = max_n
        
        # Check if this ring fits in the container
        if R_candidate + r_init > 1.0:
            # Adjust R_candidate to fit
            R_candidate = 1.0 - r_init
            # Recalculate max_n with new R
            if R_candidate > r_init:
                max_n = int(np.pi / np.arcsin(r_init / R_candidate))
            else:
                max_n = 1
            n_to_add = max_n
            
        # Place n_to_add circles
        if n_to_add > 0:
            angles = np.linspace(0, 2 * np.pi, n_to_add, endpoint=False)
            # Add a random offset to break symmetry? 
            # Symmetry might help or hurt. Let's keep it symmetric for now.
            # Offset by ring_idx * 0.1 to stagger?
            offset = ring_idx * 0.5
            angles += offset
            
            for theta in angles:
                x = R_candidate * np.cos(theta)
                y = R_candidate * np.sin(theta)
                points_list.append([x, y])
                
        R_prev = R_candidate
        
        # If we added 0 or 1, we might be stuck. 
        # If max_n is small, we might need to increase R more?
        # The logic above increases R linearly. 
        # If we run out of space (R > 1-r), we stop.
        
    # If we have more than N points, truncate.
    # If we have less, we failed to pack enough. 
    # For N=300 and r=0.05, this should work.
    
    if len(points_list) < N:
        # Fallback: Use a denser spiral if ring packing failed (unlikely)
        print("Ring packing failed, using spiral fallback")
        i = np.arange(N)
        phi = np.pi * (3. - np.sqrt(5.))
        r_spiral = np.sqrt(i / N) * (1.0 - r_init)
        theta = i * phi
        x = r_spiral * np.cos(theta)
        y = r_spiral * np.sin(theta)
        points_list = list(zip(x, y))
    else:
        points_list = points_list[:N]
        
    points = np.array(points_list)
    
    # --- Stage 2: Local Refinement ---
    # We want to maximize r = min( min_pair_dist/2, min_boundary_dist )
    # Current r is approx r_init.
    # We can try to "push" points apart slightly.
    
    # Use a simple iterative method:
    # Find the pair with minimum distance.
    # Find the point with minimum boundary distance.
    # Move the involved points to increase these distances.
    
    # To be efficient, we will use a "Repulsion" step similar to the current program
    # but initialized with this much better structure.
    # The current program's gradient descent was likely slow because of bad init.
    # With good init, even simple methods converge fast.
    
    # Let's do a few iterations of:
    # 1. Calculate all pair distances.
    # 2. Identify "critical" pairs (dist < 2*r_target) and "critical" boundary points.
    # 3. Apply forces to separate them.
    
    # Actually, let's just use the gradient ascent from the current program 
    # but start here. It's safer than inventing a new optimizer.
    # The key improvement is the INITIALIZATION.
    
    def get_min_dist_info(pts):
        # Vectorized distance calculation for N=300 is 300x300 matrix, fast.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dist[i_idx, j_idx]
        
        norms = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - norms
        
        min_pair_dist = np.min(pair_dists)
        min_boundary_dist = np.min(boundary_dists)
        
        r_pair = min_pair_dist / 2.0
        r_bound = min_boundary_dist
        
        if r_pair < r_bound:
            # Active constraint is pair
            min_val = r_pair
            idx = np.argmin(pair_dists)
            i, j = i_idx[idx], j_idx[idx]
            d_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(d_ij)
            if dist_ij < 1e-9:
                grad = np.zeros_like(pts)
            else:
                grad = np.zeros_like(pts)
                grad[i] += 0.5 * d_ij / dist_ij
                grad[j] -= 0.5 * d_ij / dist_ij
        else:
            # Active constraint is boundary
            min_val = r_bound
            idx = np.argmin(boundary_dists)
            i = idx
            p = pts[i]
            norm_p = np.linalg.norm(p)
            if norm_p < 1e-9:
                grad = np.zeros_like(pts)
            else:
                grad = np.zeros_like(pts)
                grad[i] -= p / norm_p
                
        return min_val, grad

    # Steepest Ascent with Line Search (simple version)
    step_size = 0.002
    best_r, _ = get_min_dist_info(points)
    
    # We will iterate for a fixed number of steps or until time limit
    # N=300, matrix ops are ~ 300^2 = 90k ops. Very fast.
    # We can do many iterations.
    
    for step in range(2000):
        if time.time() > deadline - 5:
            break
            
        current_r, grad = get_min_dist_info(points)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-9:
            break
        grad = grad / grad_norm
        
        # Propose move
        new_points = points + step_size * grad
        
        # Project to disk
        norms = np.sqrt(np.sum(new_points**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] *= (1.0 / norms[mask])[:, np.newaxis]
            
        new_r, _ = get_min_dist_info(new_points)
        
        if new_r > current_r:
            points = new_points
            if new_r > best_r:
                best_r = new_r
            # Increase step slightly
            step_size *= 1.02
        else:
            # Decrease step
            step_size *= 0.9
            if step_size < 1e-6:
                break
        
        step_size = min(step_size, 0.01)

    return points

# EVOLVE-BLOCK-END
