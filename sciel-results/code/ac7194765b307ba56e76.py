import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    best_radius_overall = 0.0
    best_pts_overall = None
    
    def compute_radius(pts):
        """Compute the actual packing radius."""
        # Pairwise distances
        d = pdist(pts)
        min_pair = np.min(d) / 2.0
        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - r_pts)
        return min(min_pair, min_boundary)
    
    def hex_lattice_init(n, seed=0):
        """Generate a hexagonal lattice initialization."""
        rng = np.random.RandomState(seed)
        best_points = None
        best_min_dist = 0
        
        for trial in range(5):
            # Adjust spacing based on trial
            base_spacing = 2.0 / (np.sqrt(n) + 1.0)
            spacing = base_spacing * (1.0 - trial * 0.02)
            if spacing <= 0:
                continue
            
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        points.append([x, y])
            
            if len(points) < n:
                continue
            
            points = np.array(points)
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n]
            pts_sel = points[idx]
            
            # Add small random perturbation to break symmetry
            pts_sel = pts_sel + rng.normal(0, spacing * 0.01, pts_sel.shape)
            
            tree = KDTree(pts_sel)
            pairs = tree.query_pairs(spacing * 1.5)
            if len(pairs) > 0:
                pair_dists = [np.linalg.norm(pts_sel[a] - pts_sel[b]) for a, b in pairs]
                min_pair = min(pair_dists)
            else:
                min_pair = spacing
            min_boundary = 1.0 - np.max(dists[idx])
            min_r = min(min_pair / 2.0, min_boundary)
            
            if best_points is None or min_r > best_min_dist:
                best_min_dist = min_r
                best_points = pts_sel.copy()
        
        return best_points
    
    def simulated_annealing(pts_init, max_time=50.0):
        """Run Simulated Annealing to optimize the packing."""
        start_sa = time.time()
        pts = pts_init.copy()
        current_radius = compute_radius(pts)
        best_radius = current_radius
        best_pts = pts.copy()
        
        # Cooling schedule parameters
        T_start = 0.05
        T_end = 0.001
        alpha = 0.995  # Cooling rate
        T = T_start
        
        # Precompute KDTree
        tree = KDTree(pts)
        
        it = 0
        while T > T_end and (time.time() - start_sa) < max_time:
            # Pick a random point to move
            i = np.random.randint(N)
            old_pos = pts[i].copy()
            
            # Generate a random move
            # Move magnitude is proportional to temperature
            step_size = T * 0.5
            delta = np.random.normal(0, step_size, 2)
            new_pos = old_pos + delta
            
            # Ensure the new position is inside the container
            r_new = np.hypot(new_pos[0], new_pos[1])
            if r_new > 1.0 - current_radius:
                # Project back to the boundary
                new_pos = new_pos / r_new * (1.0 - current_radius)
            
            # Update the point
            pts[i] = new_pos
            
            # Compute the new radius
            # To speed up, we can only check distances to neighbors
            # But for accuracy, we compute the full radius occasionally
            # For now, let's use a fast approximate radius check
            
            # Fast check: only check against a subset of neighbors
            # Find neighbors of the moved point
            neighbors = tree.query_ball_point(new_pos, max(current_radius * 2, 0.1))
            
            min_dist = 1.0  # Initialize to a large value
            # Check distance to neighbors
            for j in neighbors:
                if j == i:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
            
            # Check boundary
            r_i = np.hypot(new_pos[0], new_pos[1])
            boundary_dist = 1.0 - r_i
            if boundary_dist < min_dist:
                min_dist = boundary_dist
            
            new_radius = min_dist / 2.0
            
            # Acceptance criterion
            delta_E = new_radius - current_radius
            if delta_E > 0 or np.random.rand() < np.exp(delta_E / T):
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
                # Update KDTree (rebuild periodically or use a dynamic structure)
                # For simplicity, rebuild every 100 iterations
                if it % 100 == 0:
                    tree = KDTree(pts)
            else:
                # Reject the move
                pts[i] = old_pos
            
            # Cool down
            T *= alpha
            it += 1
            
            # Time check
            if time.time() > deadline - 5:
                break
        
        return best_radius, best_pts
    
    # Multi-start Simulated Annealing
    num_starts = 5
    time_per_start = (deadline - start_time - 5.0) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate a different initialization for each start
        if s == 0:
            pts_init = hex_lattice_init(N, seed=0)
        else:
            # Use a slightly perturbed version of the previous best or a new lattice
            pts_init = hex_lattice_init(N, seed=s * 10 + 1)
        
        if pts_init is None:
            continue
            
        # Run Simulated Annealing
        r, pts = simulated_annealing(pts_init, max_time=time_per_start)
        
        if r > best_radius_overall:
            best_radius_overall = r
            best_pts_overall = pts.copy()
        
        # Check time
        if time.time() > deadline - 20:
            break
    
    # Final polish: if time remains, do a quick local optimization on the best result
    if best_pts_overall is not None and time.time() < deadline - 5:
        # Simple local refinement
        pts = best_pts_overall.copy()
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            tree = KDTree(pts)
            improved = False
            for i in range(N):
                # Try small random moves
                for _ in range(10):
                    old_pos = pts[i].copy()
                    step_size = 0.001
                    delta = np.random.normal(0, step_size, 2)
                    new_pos = old_pos + delta
                    r_new = np.hypot(new_pos[0], new_pos[1])
                    if r_new > 1.0:
                        continue
                    pts[i] = new_pos
                    # Fast radius check
                    neighbors = tree.query_ball_point(new_pos, 0.1)
                    min_dist = 1.0
                    for j in neighbors:
                        if j == i:
                            continue
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_dist:
                            min_dist = d
                    boundary_dist = 1.0 - r_new
                    if boundary_dist < min_dist:
                        min_dist = boundary_dist
                    new_r = min_dist / 2.0
                    if new_r > best_radius_overall:
                        best_radius_overall = new_r
                        best_pts_overall = pts.copy()
                        improved = True
                    else:
                        pts[i] = old_pos
            if not improved:
                break
    
    return best_pts_overall
