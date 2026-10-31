"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_dists(pts):
        # Returns array of squared distances for i < j
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy:
    # 1. Generate high-quality initial configurations based on known packing structures.
    #    For 16 points, a subset of the triangular (hexagonal) lattice is a strong candidate.
    #    The triangular lattice minimizes the maximum distance for a given minimum distance
    #    in infinite space. We select 16 points from a large triangular grid and refine.
    
    # Generate a triangular lattice
    def generate_triangular_lattice_subset(n_points):
        # Triangular lattice basis
        # v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
        # We generate a large grid and pick the n_points closest to the center
        # to ensure a compact, symmetric cluster.
        
        # Generate indices for a large enough grid
        # Side length approx sqrt(n_points) * factor
        side = int(np.ceil(np.sqrt(n_points) * 2))
        
        pts = []
        for i in range(-side, side + 1):
            for j in range(-side, side + 1):
                # Map (i,j) to 2D coordinates
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                pts.append([x, y])
        
        pts = np.array(pts)
        # Center the lattice
        pts -= np.mean(pts, axis=0)
        
        # Sort by distance from center and take the first n_points
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)[:n_points]
        selected_pts = pts[indices]
        
        return selected_pts

    # Also consider a square grid subset, though triangular is usually better for ratio
    def generate_square_grid_subset(n_points):
        side = int(np.ceil(np.sqrt(n_points)))
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append([i - side/2 + 0.5, j - side/2 + 0.5])
                if len(pts) == n_points:
                    break
            if len(pts) == n_points:
                break
        return np.array(pts)

    # Local Search Function
    # Uses a simple hill-climbing approach with random perturbations.
    # To handle the non-smooth nature, we try to reduce the max distance or increase the min distance.
    # Since the score is (max/min)^2, reducing max is often harder if min is fixed, 
    # but increasing min is constrained by max. 
    # We use a standard "simulated annealing" like step size reduction.
    
    def local_search(pts, max_time=5.0):
        current_pts = pts.copy()
        current_score = score_raw(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Initial step size relative to typical distance
        d_sq = get_dists(current_pts)
        typical_d = np.sqrt(np.mean(d_sq))
        step_size = 0.1 * typical_d
        
        rng = np.random.default_rng(42)
        last_improvement = time.time()
        
        while time.time() < start_time + max_time:
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = rng.integers(0, N)
            
            # Generate a random direction
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Try a few different step magnitudes (exploration)
            found_improvement = False
            for factor in [1.0, 0.5, 0.1, 2.0]:
                new_pts = current_pts.copy()
                new_pts[idx] += direction * step_size * factor
                
                # Ensure no points overlap (min dist > 0)
                d_sq_new = get_dists(new_pts)
                min_d_sq = np.min(d_sq_new)
                if min_d_sq < 1e-12:
                    continue
                
                new_score = score_raw(new_pts)
                
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
                    found_improvement = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = current_pts.copy()
                        last_improvement = time.time()
                    break # Accept the first improvement found for this direction? 
                         # Or try all? Trying all is safer but slower. 
                         # Let's accept the best of the factors.
            
            # If no improvement found, reduce step size (cooling)
            if not found_improvement:
                step_size *= 0.95
                # Occasionally, if stuck for a long time, do a larger random jump to escape local min
                if time.time() - last_improvement > 2.0:
                    # Random restart of one point within a small neighborhood
                    jump_idx = rng.integers(0, N)
                    jump_angle = rng.uniform(0, 2 * np.pi)
                    jump_dist = typical_d * rng.uniform(0.5, 1.5)
                    new_pts = current_pts.copy()
                    new_pts[jump_idx] += np.array([np.cos(jump_angle), np.sin(jump_angle)]) * jump_dist
                    new_score = score_raw(new_pts)
                    if new_score < current_score * 1.1: # Allow slight worsening to escape
                        current_pts = new_pts
                        current_score = new_score
            
            # Check time
            if time.time() > start_time + max_time:
                break
        
        return best_pts, best_score

    # Main Loop: Try multiple initial configurations
    
    best_overall_pts = None
    best_overall_score = 1e10
    
    # 1. Triangular Lattice Subset
    init_tri = generate_triangular_lattice_subset(N)
    init_tri = normalize(init_tri)
    pts_tri, score_tri = local_search(init_tri, max_time=30.0)
    if score_tri < best_overall_score:
        best_overall_score = score_tri
        best_overall_pts = pts_tri
    
    if time.time() > deadline:
        return best_overall_pts if best_overall_pts is not None else init_tri

    # 2. Square Grid Subset
    init_sq = generate_square_grid_subset(N)
    init_sq = normalize(init_sq)
    pts_sq, score_sq = local_search(init_sq, max_time=20.0)
    if score_sq < best_overall_score:
        best_overall_score = score_sq
        best_overall_pts = pts_sq
    
    if time.time() > deadline:
        return best_overall_pts

    # 3. Random Points in Circle (Multiple restarts)
    rng_global = np.random.default_rng(123)
    for i in range(5):
        if time.time() > deadline - 20:
            break
        angles = rng_global.uniform(0, 2 * np.pi, N)
        radii = rng_global.uniform(0.2, 1.0, N) # Avoid center clustering
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = normalize(pts)
        
        # Shorter search time for random restarts
        pts_rand, score_rand = local_search(pts, max_time=10.0)
        if score_rand < best_overall_score:
            best_overall_score = score_rand
            best_overall_pts = pts_rand

    # 4. If we still have time, refine the best found configuration with a longer search
    if best_overall_pts is not None and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            # Use the best found as the start for a final, longer optimization
            # We can also try to "anneal" it further
            final_pts, final_score = local_search(best_overall_pts, max_time=remaining_time)
            if final_score < best_overall_score:
                best_overall_score = final_score
                best_overall_pts = final_pts

    # Fallback if nothing worked
    if best_overall_pts is None:
        best_overall_pts = normalize(generate_triangular_lattice_subset(N))

    return best_overall_pts
# EVOLVE-BLOCK-END
