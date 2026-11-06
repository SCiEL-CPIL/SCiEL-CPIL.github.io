"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Stage 1: Robust Initialization ---
    # Use a spiral packing which is denser than random and often better than 
    # naive rings for non-trivial N.
    # We start with a radius slightly smaller than the target to ensure validity.
    # Area bound: r <= 1/sqrt(300) ~ 0.0577. Best known ~0.0531.
    # Let's init with r_init = 0.045 to have plenty of slack.
    
    i = np.arange(N)
    # Golden angle spiral
    phi = np.pi * (3. - np.sqrt(5.))
    # Scale radius to fit in unit disk with some margin
    r_max = 1.0 - 0.05 
    r_spiral = np.sqrt(i / N) * r_max
    theta = i * phi
    
    points = np.column_stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)])
    
    # --- Stage 2: Simulated Annealing Optimization ---
    # Objective: Maximize r = min(min_pair_dist/2, min_boundary_dist)
    # Equivalent to: Minimize Penalty = sum( max(0, 2*r_target - dist_ij)^2 ) + sum( max(0, |p_i| + r_target - 1)^2 )
    # We will use a "soft" target radius that increases over time.
    
    # Initial target radius
    r_target = 0.050
    # Final target radius (we hope to reach > 0.053)
    r_final = 0.0535
    
    # Temperature for SA
    T = 0.01
    T_min = 1e-6
    alpha = 0.9995 # Cooling rate
    
    best_r = -1.0
    best_points = points.copy()
    
    # Pre-allocate for speed? No, KNN rebuild is fast enough for 300 points.
    
    def compute_score(pts, r):
        """
        Computes the actual radius r achieved by the configuration.
        Returns (r, tree) where tree is the KDTree for the points.
        """
        tree = cKDTree(pts)
        # Find nearest neighbors
        dists, _ = tree.query(pts, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        r_val = min(min_pair_dist / 2.0, min_boundary_dist)
        return r_val, tree

    def compute_forces(pts, r, tree):
        """
        Computes repulsive forces from violating constraints at radius r.
        We use a soft potential: F = k * (2r - d) for d < 2r.
        """
        forces = np.zeros_like(pts)
        
        # 1. Pairwise forces
        # Only check neighbors within 2*r + epsilon
        # Using query_pairs is efficient
        # But we need the vectors. 
        # Let's use a simpler approach: query k=10 neighbors to ensure we catch all potential overlaps
        # For N=300, k=10 is cheap.
        dists, idxs = tree.query(pts, k=10)
        
        for i in range(N):
            for j_idx in range(1, 10): # k=10 means 10 neighbors, index 0 is self
                j = idxs[i, j_idx]
                d = dists[i, j_idx]
                if d < 1e-9:
                    continue
                if d < 2 * r:
                    # Repulsive force
                    # Magnitude proportional to overlap
                    overlap = 2 * r - d
                    # Force direction: from j to i
                    vec = pts[i] - pts[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        forces[i] += (overlap / norm) * vec
        
        # 2. Boundary forces
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] + r > 1.0:
                overlap = norms[i] + r - 1.0
                # Force direction: towards center
                if norms[i] > 1e-9:
                    forces[i] -= (overlap / norms[i]) * pts[i]
            
        return forces

    # Main SA Loop
    num_steps = 0
    while time.time() < deadline - 5.0:
        # Annealing schedule
        # Linearly increase r_target towards r_final
        # Exponentially decrease T
        
        # Update r_target
        # We want to push r_target up slowly.
        # Let's tie it to step count or time.
        # Simple: r_target = r_start + (r_final - r_start) * (1 - exp(-t/tau))
        # But we don't know t exactly. Let's just increment it slightly each step.
        # Actually, better to fix r_target for a while, let it converge, then increase.
        
        # For simplicity and robustness in code generation:
        # We will run multiple "phases".
        # Phase 1: r=0.050, T=0.01
        # Phase 2: r=0.051, T=0.005
        # etc.
        
        # Let's implement a dynamic r_target increase based on success.
        
        # Current Score
        current_r, tree = compute_score(points, r_target)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # If current_r is already above r_target, we can increase r_target
        if current_r > r_target:
            r_target = min(r_final, r_target + 0.0001)
            T = max(T_min, T * 0.9) # Cool down a bit when progress is made?
            # Or keep T constant to explore?
            # Standard SA: T decreases. r_target increases.
        
        # Generate Move
        # Pick a random subset of points to move (cluster move) or single point?
        # Single point moves are local. Cluster moves can escape local minima better.
        # Let's try moving 1-5 random points.
        n_move = np.random.randint(1, 6)
        move_indices = np.random.choice(N, n_move, replace=False)
        
        # Random displacement
        # Scale displacement by T
        disp = np.random.randn(n_move, 2) * T * 2.0 # Factor 2 for scale
        # Limit displacement to prevent huge jumps
        max_disp = 0.05
        disp_norms = np.linalg.norm(disp, axis=1)
        mask = disp_norms > max_disp
        if np.any(mask):
            disp[mask] *= (max_disp / disp_norms[mask])[:, np.newaxis]
            
        new_points = points.copy()
        new_points[move_indices] += disp
        
        # Project to disk? 
        # If a point goes outside, the boundary force will push it back.
        # But for validity during scoring, we must ensure it's inside.
        # Let's project to disk [0, 1-r_target] roughly? 
        # No, just let the forces handle it, but compute score on projected points?
        # The evaluator requires points inside.
        # If new_points has points outside, the score will be invalid (or 0).
        # To be safe, project new_points to the disk of radius 1.
        # But we want to allow "virtual" expansion during search?
        # No, the state space is the disk.
        norms = np.linalg.norm(new_points, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            new_points[mask_out] *= (1.0 / norms[mask_out])[:, np.newaxis]
            
        # Compute Score of New Points
        new_r, new_tree = compute_score(new_points, r_target)
        
        # Acceptance Criterion
        delta_r = new_r - current_r
        
        # If new_r is better, accept.
        # If worse, accept with probability exp(delta_r / T)
        # Note: delta_r can be negative.
        
        if delta_r >= 0:
            points = new_points
            # T cooling
            T = max(T_min, T * alpha)
        else:
            prob = np.exp(delta_r / T)
            if np.random.rand() < prob:
                points = new_points
                T = max(T_min, T * alpha)
            else:
                # Reject move, but still cool down?
                # Standard SA cools regardless.
                T = max(T_min, T * alpha)
                
        num_steps += 1
        
        # Check time
        if time.time() > deadline - 10.0:
            break
            
    # Final Polish
    # Run a few steps of steepest ascent on the best found points to squeeze out max r
    # using the forces from the last known good r.
    
    # Re-compute best_r and best_points one last time
    final_r, _ = compute_score(best_points, 0)
    
    # If the best_points found during SA is valid, return it.
    # Ensure validity
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
         best_points[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]
         
    return best_points

# EVOLVE-BLOCK-END
