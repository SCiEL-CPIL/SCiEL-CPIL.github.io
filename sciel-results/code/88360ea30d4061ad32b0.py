"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        n = len(centers)
        # Use cKDTree for efficient nearest-neighbor distance
        tree = cKDTree(centers)
        # Find minimum pairwise distance
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)

    def generate_hex_candidates(n_rows, n_cols):
        """Generate a hexagonal lattice of candidate positions."""
        pts = []
        if n_rows <= 0:
            return np.array([])
        d_max_x = 1.0 / n_cols
        d_max_h = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1)
        d = min(d_max_x, d_max_h)
        x_offset = -0.5 + d / 2
        y_offset = -0.5 + d / 2
        for row in range(n_rows):
            y = y_offset + row * d * math.sqrt(3) / 2
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = x_offset
            else:
                n_in_row = n_cols - 1
                x_start = x_offset + d / 2
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        return pts

    def greedy_select(candidates, n_select=N):
        """Greedy selection: pick points that maximize minimum distance to already-selected."""
        if len(candidates) < n_select:
            return candidates
        selected = []
        # Start with a random point
        idx = np.random.randint(len(candidates))
        selected.append(candidates[idx])
        remaining = list(range(len(candidates)))
        remaining.remove(idx)
        
        while len(selected) < n_select and remaining:
            sel_arr = np.array(selected)
            # Find candidate with max min-distance to selected
            best_idx = -1
            best_min_dist = -1
            # Vectorized: for each remaining candidate, compute min dist to selected
            for ri in remaining:
                cand = candidates[ri]
                dists = np.sqrt(np.sum((sel_arr - cand)**2, axis=1))
                md = np.min(dists)
                if md > best_min_dist:
                    best_min_dist = md
                    best_idx = ri
            if best_idx < 0:
                break
            selected.append(candidates[best_idx])
            remaining.remove(best_idx)
        
        return np.array(selected[:n_select])

    def delaunay_local_search(centers, max_iters=500, step_size=0.002):
        """
        Delaunay-based local optimization:
        - Maintain Delaunay triangulation
        - For each point, compute gradient of min-distance-to-neighbors
        - Push points to increase the minimum Delaunay edge length
        - Also handle wall constraints
        """
        centers = centers.copy()
        n = len(centers)
        best_r = compute_radius(centers)
        no_improve_count = 0
        current_step = step_size
        
        for iteration in range(max_iters):
            if time_left() < 5.0:
                break
            
            # Build Delaunay triangulation
            try:
                tri = Delaunay(centers)
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            a, b = simplex[i], simplex[j]
                            if a > b:
                                a, b = b, a
                            edges.add((a, b))
                edges = list(edges)
            except Exception:
                break
            
            if not edges:
                break
            
            # Compute current radius
            r = compute_radius(centers)
            
            # For each point, compute the direction to increase min distance
            # to its Delaunay neighbors
            forces = np.zeros_like(centers)
            min_edge_len = np.inf
            min_edge_pair = None
            
            for (a, b) in edges:
                diff = centers[a] - centers[b]
                dist = np.sqrt(np.dot(diff, diff))
                if dist < min_edge_len:
                    min_edge_len = dist
                    min_edge_pair = (a, b)
                # Normalize and add force
                if dist > 1e-10:
                    unit = diff / dist
                    # Force proportional to (target - dist) where target is current min
                    # Simpler: just push apart
                    forces[a] += unit
                    forces[b] -= unit
            
            # Wall forces
            for i in range(n):
                for dim in range(2):
                    if centers[i, dim] > 0.5 - r:
                        forces[i, dim] -= (centers[i, dim] - (0.5 - r)) * 5.0
                    elif centers[i, dim] < -0.5 + r:
                        forces[i, dim] += ((-0.5 + r) - centers[i, dim]) * 5.0
            
            # Normalize forces
            force_norms = np.sqrt(np.sum(forces**2, axis=1))
            for i in range(n):
                if force_norms[i] > 1e-10:
                    forces[i] /= force_norms[i]
            
            # Apply forces
            new_centers = centers + current_step * forces
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_r = compute_radius(new_centers)
            if new_r > r + 1e-12:
                centers = new_centers
                best_r = max(best_r, new_r)
                no_improve_count = 0
                # Slightly increase step if improving
                current_step = min(current_step * 1.05, 0.01)
            else:
                no_improve_count += 1
                current_step = max(current_step * 0.9, 0.0005)
                if no_improve_count > 30:
                    break
        
        return centers, best_r

    def parallel_local_search(centers, n_restarts=5):
        """Try multiple small perturbations and local search each."""
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        for restart in range(n_restarts):
            if time_left() < 10.0:
                break
            
            # Perturb
            perturbed = centers + np.random.randn(N, 2) * 0.003
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            # Local search
            result, r = delaunay_local_search(perturbed, max_iters=200, step_size=0.001)
            if r > best_r:
                best_r = r
                best_centers = result
        
        return best_centers, best_r

    def final_polish(centers, max_iters=100):
        """Final fine-tuning with very small steps."""
        centers = centers.copy()
        best_r = compute_radius(centers)
        
        for iteration in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Try moving each point in 4 directions with tiny step
            improved = False
            step = 0.0005
            for i in range(N):
                if time_left() < 3.0:
                    break
                orig = centers[i].copy()
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),(step,step),(-step,-step),(step,-step),(-step,step)]:
                    new_pos = orig + np.array([dx, dy])
                    if new_pos[0] < -0.5 or new_pos[0] > 0.5:
                        continue
                    if new_pos[1] < -0.5 or new_pos[1] > 0.5:
                        continue
                    centers[i] = new_pos
                    r = compute_radius(centers)
                    if r > best_r + 1e-12:
                        best_r = r
                        improved = True
                    else:
                        centers[i] = orig.copy()
            
            if not improved:
                break
        
        return centers, best_r

    # Main strategy:
    # 1. Generate dense hexagonal candidate pool
    # 2. Greedy select N points to maximize spread
    # 3. Delaunay local search to refine
    # 4. Parallel restarts
    # 5. Final polish

    best_centers = None
    best_radius = 0.0

    # Generate multiple candidate pools with different densities
    candidate_pools = []
    for n_rows in [14, 15, 16, 17, 18]:
        n_cols = math.ceil(N / n_rows) + 2  # extra columns for greedy selection
        pool = generate_hex_candidates(n_rows, n_cols)
        if len(pool) >= N:
            candidate_pools.append(pool)
    
    # Also add a square grid pool
    m = 16
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pool = np.array([(x, y) for y in ticks for x in ticks])
    candidate_pools.append(grid_pool)

    # Greedy select from each pool and run local search
    for pool in candidate_pools:
        if time_left() < 15.0:
            break
        
        # Greedy select
        selected = greedy_select(pool, N)
        if len(selected) < N:
            continue
        
        # Initial Delaunay local search
        result, r = delaunay_local_search(selected, max_iters=300, step_size=0.002)
        
        if r > best_radius:
            best_radius = r
            best_centers = result.copy()
    
    # Parallel restarts on the best configuration
    if best_centers is not None and time_left() > 20.0:
        polished, r = parallel_local_search(best_centers, n_restarts=4)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
    
    # Final polish
    if best_centers is not None and time_left() > 5.0:
        polished, r = final_polish(best_centers, max_iters=80)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
    
    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
