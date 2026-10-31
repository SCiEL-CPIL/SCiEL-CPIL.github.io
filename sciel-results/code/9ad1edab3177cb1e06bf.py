import numpy as np
import time
import sys

N = 38

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate FCC lattice points
    # Points (i,j,k) such that i+j+k is even.
    # Range -3 to 3 is enough to get >38 points.
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    
    pts = np.array(pts, dtype=float)
    
    # 2. Sort by distance from origin
    dist_sq = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dist_sq)
    pts_sorted = pts[sorted_indices]
    
    # 3. Take first N points
    # Note: There might be ties in distance.
    # For N=38, we take 1 (d=0) + 12 (d=sqrt2) + 6 (d=2) + 19 (d=sqrt6).
    # The 19 points from the 24 available in the d=sqrt6 shell.
    # This selection is arbitrary based on array order.
    # To improve symmetry, we could shuffle the d=sqrt6 points before taking 19,
    # but the specific subset doesn't matter as much as the geometry,
    # and L-BFGS will relax it.
    # However, to be more robust, we can try a few random subsets of the 24 points.
    
    # Identify the shells
    # d_sq = 0, 2, 4, 6
    # We want 38 points.
    # Inner 19 are fixed (d_sq <= 4).
    # Outer 19 are chosen from d_sq == 6.
    
    inner_mask = dist_sq < 4.5 # d_sq 0, 2, 4
    inner_pts = pts[inner_mask]
    
    outer_mask = (dist_sq >= 4.5) & (dist_sq <= 6.5) # d_sq 6
    outer_pts = pts[outer_mask]
    
    if len(inner_pts) + len(outer_pts) < N:
        # Fallback to simple closest N
        init_pts = pts_sorted[:N]
    else:
        # We have enough.
        # Randomly select 19 from outer_pts to break symmetry randomly?
        # Or just take the first 19?
        # Taking the first 19 might be biased.
        # Let's try a few random selections.
        
        best_energy = np.inf
        best_pos = None
        
        # Deterministic first try: first 19
        # Then a few random tries if time permits
        
        num_outer_needed = N - len(inner_pts)
        
        # Try 1: First 19
        selected_outer = outer_pts[:num_outer_needed]
        pos = np.vstack([inner_pts, selected_outer])
        
        # Scale to optimal LJ distance
        # Nearest neighbor in this lattice is sqrt(2).
        # Optimal LJ r is 2**(1/6).
        scale = 2**(1/6) / np.sqrt(2)
        pos = pos * scale
        
        # Relax
        try:
            relaxed_pos, energy = science_chemistry.relax(pos)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except Exception:
            pass
        
        # Try 2-5: Random subsets if time allows
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline:
                break
            
            # Randomly select 19 from outer_pts
            indices = rng.choice(len(outer_pts), num_outer_needed, replace=False)
            selected_outer = outer_pts[indices]
            pos = np.vstack([inner_pts, selected_outer])
            pos = pos * scale
            
            # Add tiny noise to break symmetry further
            pos += rng.normal(0, 0.01, pos.shape)
            
            try:
                relaxed_pos, energy = science_chemistry.relax(pos)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
            except Exception:
                pass
        
        if best_pos is not None:
            return best_pos
        else:
            return pos # Return unrelaxed if relax failed, though unlikely

    # Fallback if logic above didn't return
    return pts_sorted[:N] * (2**(1/6) / np.sqrt(2))

# Note: science_chemistry is assumed to be available in the environment as per task description.
