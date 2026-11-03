"""
Discrete Lattice Search for 100 equal spheres in a unit cube.

Strategy:
1. Discretize the cube [-0.5, 0.5]^3 into a grid.
2. For a target radius r, build a graph where nodes are grid cells and edges connect
   cells closer than 2r.
3. Use a greedy Maximum Independent Set (MIS) heuristic to select 100 non-conflicting cells.
4. Refine the selected centers using local continuous optimization (jitter + force)
   to maximize the minimum distance, starting from the discrete solution.
5. Iterate with increasing target radius or different grid resolutions.
"""
import numpy as np
import time
import sys

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
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _greedy_mis(nodes, adj_matrix, k):
    """
    Greedy Maximum Independent Set.
    nodes: list of node indices
    adj_matrix: boolean NxN matrix
    k: number of nodes to select
    Returns list of selected node indices.
    """
    selected = []
    available = set(nodes)
    
    # Sort nodes by degree (ascending) to prioritize "loose" nodes first? 
    # Or descending to fill tight spots? 
    # Usually, for packing, we want to place spheres in "hard" spots first or 
    # just greedily pick the node with the fewest conflicts among remaining available.
    # Let's try: pick node with min degree in the subgraph of available nodes.
    
    while len(selected) < k and available:
        # Calculate degrees in the available subgraph
        degrees = np.zeros(len(nodes), dtype=int)
        for u in available:
            # Count neighbors in available
            neighbors = np.where(adj_matrix[u])[0]
            count = 0
            for v in neighbors:
                if v in available:
                    count += 1
            degrees[u] = count
        
        # Find node with min degree
        min_deg = np.inf
        best_node = -1
        for u in available:
            if degrees[u] < min_deg:
                min_deg = degrees[u]
                best_node = u
        
        if best_node == -1:
            break
            
        selected.append(best_node)
        available.remove(best_node)
        
        # Remove neighbors of best_node from available
        neighbors = np.where(adj_matrix[best_node])[0]
        for v in neighbors:
            if v in available:
                available.remove(v)
                
    return selected

def _discrete_search(grid_res, target_r, max_nodes_to_consider=5000):
    """
    Perform discrete search on a grid of resolution 'grid_res'.
    Returns a set of 100 points if a valid packing is found, else None.
    """
    # Generate grid points
    # We need points to be at least target_r away from walls.
    # So valid range for coordinates is [-0.5+target_r, 0.5-target_r]
    span = 1.0 - 2 * target_r
    if span <= 0:
        return None
    
    # Create grid coordinates
    # To ensure we have enough candidates, we map grid indices to physical coords
    # We don't need the grid to be uniform in physical space, but it's a good start.
    # Let's use a uniform grid over the valid range.
    
    # Number of points per axis
    # Total nodes ~ grid_res^3. 
    # For grid_res=20, 8000 nodes. Adj matrix 8000x8000 is 64M bools = 64MB. OK.
    # For grid_res=30, 27000 nodes. 729M bools = 729MB. Too big for naive bool matrix in Python/NumPy if not careful.
    # Let's stick to grid_res <= 25 for memory safety in this simple implementation.
    
    coords = np.linspace(-0.5 + target_r, 0.5 - target_r, grid_res)
    grid_points = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    
    num_nodes = len(grid_points)
    if num_nodes < N:
        return None
        
    # Build adjacency matrix: edge if dist < 2 * target_r
    # Use squared distances to avoid sqrt
    min_dist_sq = (2 * target_r) ** 2
    
    # Efficient distance calculation for grid? 
    # Since it's a grid, we can exploit locality, but for N<=15000, O(N^2) might be slow in pure Python loops.
    # NumPy broadcasting is O(N^2) memory. 
    # For 25^3 = 15625 nodes, 15625^2 = 244M floats. ~2GB. Too much.
    
    # Alternative: Only check neighbors within a bounding box.
    # In a grid, if |dx| > 2r, they can't be connected.
    # Let's implement a sparse adjacency list building using grid properties.
    
    # Map grid index to (ix, iy, iz)
    # Index = ix * grid_res^2 + iy * grid_res + iz
    
    adj_list = [[] for _ in range(num_nodes)]
    
    # Precompute which grid steps are within 2r
    # step_size = span / (grid_res - 1)
    step_size = span / (grid_res - 1) if grid_res > 1 else 1.0
    max_step = int(np.ceil(2 * target_r / step_size))
    
    for ix in range(grid_res):
        for iy in range(grid_res):
            for iz in range(grid_res):
                u = ix * grid_res * grid_res + iy * grid_res + iz
                px, py, pz = grid_points[u]
                
                # Check neighbors in the local window
                for dx in range(-max_step, max_step + 1):
                    for dy in range(-max_step, max_step + 1):
                        for dz in range(-max_step, max_step + 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nx_, ny_, nz_ = ix + dx, iy + dy, iz + dz
                            if 0 <= nx_ < grid_res and 0 <= ny_ < grid_res and 0 <= nz_ < grid_res:
                                v = nx_ * grid_res * grid_res + ny_ * grid_res + nz_
                                # Calculate dist
                                dx_phys = (nx_ - ix) * step_size
                                dy_phys = (ny_ - iy) * step_size
                                dz_phys = (nz_ - iz) * step_size
                                dist_sq = dx_phys**2 + dy_phys**2 + dz_phys**2
                                if dist_sq < min_dist_sq:
                                    adj_list[u].append(v)
                                    
    # Now we have adjacency list. 
    # Greedy MIS on adjacency list is easier than matrix.
    
    selected_indices = []
    available = set(range(num_nodes))
    
    # Heuristic: Pick node with min degree in available subgraph
    # To speed up, we can just pick randomly or by fixed order if min-degree is too slow.
    # Let's try min-degree but with a cap on time.
    
    import random
    # Shuffle available to break symmetry? Or just iterate.
    # Min-degree calculation requires checking neighbors in available.
    
    while len(selected_indices) < N and available:
        # Find min degree node
        best_node = -1
        min_deg = float('inf')
        
        # Sample a subset of available nodes to find a good candidate to save time?
        # For now, check all. If num_nodes is 15000, and we do 100 iterations, 
        # each iteration checks 15000 nodes * avg_degree. 
        # Avg degree in 3D packing is roughly 12-20. 
        # 15000 * 20 = 300k ops per iter. 100 iters = 30M ops. Should be fine in NumPy/Python if optimized.
        # But pure Python loops for this are slow.
        
        # Let's use a simpler heuristic: Random selection with conflict check, 
        # or "First Fit Decreasing" style?
        # Actually, let's just pick the node with the fewest neighbors in `available`.
        
        # To make this faster, we can maintain a degree count.
        # But updating degrees is complex.
        
        # Let's try a simple random greedy:
        # Pick a random node from available. If it has conflicts, remove conflicts? 
        # No, that's not MIS.
        
        # Let's stick to min-degree but vectorize the degree calc if possible?
        # It's hard to vectorize "degree in available subgraph" easily without a matrix.
        
        # Given time constraints, let's use a "Random Greedy" with a twist:
        # Pick random node. If it's in available, select it. Remove it and its neighbors from available.
        # This is a valid greedy MIS approximation.
        
        # To improve quality, we can sort available by degree (pre-computed full degree) 
        # and pick the one with lowest full degree that is still available.
        
        # Pre-compute full degrees
        full_degrees = np.array([len(neigh) for neigh in adj_list])
        
        # Get sorted indices by full degree
        sorted_indices = np.argsort(full_degrees)
        
        # Iterate through sorted indices, pick first one that is in available
        found = False
        for idx in sorted_indices:
            if idx in available:
                best_node = idx
                found = True
                break
        
        if not found:
            break
            
        selected_indices.append(best_node)
        available.remove(best_node)
        
        for v in adj_list[best_node]:
            if v in available:
                available.remove(v)
                
    if len(selected_indices) < N:
        return None
        
    return grid_points[selected_indices]

def _local_polish(points, iterations=100, step_size=0.005):
    """
    Local continuous optimization to polish the discrete solution.
    Uses a simple gradient step based on repulsion forces.
    """
    pts = points.copy()
    
    for _ in range(iterations):
        # Calculate forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, 1e-6)
        
        # Force proportional to 1/d^2 for pairs closer than 2*r_current?
        # We don't know r_current exactly, so use a fixed small cutoff or just 1/d^2 for all.
        # To focus on the minimum, use 1/d^4.
        
        # Avoid huge forces for very small distances
        dists_safe = np.maximum(dists, 0.01)
        
        # Weights
        weights = 1.0 / (dists_safe ** 4)
        np.fill_diagonal(weights, 0)
        
        # Force on i: sum_j w_ij * (p_i - p_j)
        sum_w = weights.sum(axis=1, keepdims=True)
        weighted_sum = weights @ pts
        forces = sum_w * pts - weighted_sum
        
        # Wall forces
        # Push away from walls if close
        wall_dists = 0.5 - np.abs(pts).max(axis=1)
        wall_forces = np.zeros_like(pts)
        for i in range(N):
            if wall_dists[i] < 0.1:
                # Direction to center
                wall_forces[i] = -pts[i] / (np.linalg.norm(pts[i]) + 1e-6)
                # Scale by how close to wall
                wall_forces[i] *= (0.1 / max(wall_dists[i], 1e-6)) ** 2
                
        total_forces = forces + wall_forces
        
        # Update
        pts += step_size * total_forces
        pts = np.clip(pts, -0.49, 0.49)
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = 0.0
    
    # Initial random points to have a baseline
    np.random.seed(42)
    init_points = np.random.uniform(-0.4, 0.4, (N, 3))
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Try different grid resolutions and target radii
    # Start with coarse grid and small radius, then refine
    grid_resolutions = [15, 18, 20, 22]
    target_radii = [0.10, 0.105, 0.11, 0.111]
    
    for res in grid_resolutions:
        if time.time() > deadline - 20:
            break
        for r_target in target_radii:
            if time.time() > deadline - 10:
                break
            
            # Discrete search
            discrete_pts = _discrete_search(res, r_target)
            
            if discrete_pts is not None:
                # Polish
                polished = _local_polish(discrete_pts, iterations=50, step_size=0.002)
                score = _score(polished)
                
                if score > best_score:
                    best_score = score
                    best_points = polished
                    # If we found a high score, we might want to try higher radii with same res
                    # But let's just continue to next config for now.
            
            # Also try to polish the current best points directly
            # This helps if the discrete search failed but local search can improve
            if time.time() < deadline - 5:
                polished_best = _local_polish(best_points, iterations=20, step_size=0.001)
                score = _score(polished_best)
                if score > best_score:
                    best_score = score
                    best_points = polished_best

    # Final polish
    if time.time() < deadline - 2:
        final_polish = _local_polish(best_points, iterations=50, step_size=0.0005)
        score = _score(final_polish)
        if score > best_score:
            best_points = final_polish
            best_score = score

    return best_points

# EVOLVE-BLOCK-END
