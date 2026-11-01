import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to minimum pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_icosahedron_grid():
    """
    Generate the 12 vertices of a regular icosahedron.
    """
    phi = (1 + 5**0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)

def _generate_cuboctahedron_grid():
    """
    Generate the 12 vertices of a cuboctahedron.
    """
    verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)

def _generate_fibonacci_sphere(n_points, seed=0):
    """
    Generate points on a sphere using the Fibonacci spiral method.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n_points)
    # Add a small random jitter to break perfect symmetry if needed, 
    # but standard Fibonacci is deterministic.
    # We can rotate the sphere randomly for diversity in restarts.
    
    phi = np.arccos(1 - 2*(i+0.5)/n_points)
    theta = np.pi*(1+5**0.5)*i + rng.uniform(0, 2*np.pi)
    
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    
    return np.column_stack([x, y, z])

def _generate_candidate_pool():
    """
    Generate a diverse pool of candidate points on the unit sphere.
    We combine high-symmetry polyhedra vertices and Fibonacci points.
    """
    pool = []
    
    # 1. Icosahedron vertices (12 points)
    pool.append(_generate_icosahedron_grid())
    
    # 2. Cuboctahedron vertices (12 points)
    pool.append(_generate_cuboctahedron_grid())
    
    # 3. Multiple Fibonacci spheres with different rotations
    for i in range(20):
        pool.append(_generate_fibonacci_sphere(100, seed=i))
        
    # 4. Random points on sphere (for diversity)
    rng = np.random.default_rng(123)
    for i in range(10):
        pts = rng.normal(0, 1, size=(100, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pool.append(pts)
        
    # Concatenate all
    all_pts = np.vstack(pool)
    return all_pts

def _greedy_select_min_dist(pts, k):
    """
    Greedily select k points from pts to maximize the minimum pairwise distance.
    This is an approximation to the Tammes problem.
    """
    selected = []
    remaining = np.arange(len(pts))
    
    # Start with a random point or the one with highest average distance to others?
    # Let's start with a random point to allow different restarts.
    # Or, to be deterministic but diverse, we can try multiple starts.
    # For now, let's pick the point that is farthest from the centroid (which is 0 for sphere)
    # Actually, all points on unit sphere are equidistant from centroid.
    # Let's just pick index 0 for the first greedy pass, but we will do multiple passes.
    
    # Better: Pick a random start point for this greedy run.
    # But this function is called inside a loop where we control the start.
    # Let's modify this to take a start index.
    
    # Actually, a common heuristic is to start with a random point.
    # We will implement a version that takes a start index.
    
    pass # Replaced by _greedy_select_with_start

def _greedy_select_with_start(pts, k, start_idx):
    """
    Greedily select k points from pts starting with start_idx.
    Objective: Maximize the minimum pairwise distance (Tammes).
    """
    selected_indices = [start_idx]
    selected_pts = [pts[start_idx]]
    
    # Current min distances to selected set for each remaining point
    # dist[i] = min distance from pts[i] to any point in selected_pts
    
    # Initialize distances
    d = cdist(pts, pts)
    
    # For efficiency, we can update distances incrementally, but for k=14 and pool ~1000,
    # full cdist is fast enough.
    
    for _ in range(k - 1):
        # Calculate distance from each unselected point to the closest selected point
        min_dists = np.full(len(pts), np.inf)
        for s_idx in selected_indices:
            min_dists = np.minimum(min_dists, d[:, s_idx])
        
        # Mask out already selected points
        for s_idx in selected_indices:
            min_dists[s_idx] = -1
            
        # Select the point with the maximum minimum distance
        next_idx = np.argmax(min_dists)
        
        if min_dists[next_idx] < 1e-9:
            break # All points are too close
            
        selected_indices.append(next_idx)
        selected_pts.append(pts[next_idx])
        
    return np.array(selected_pts)

def _local_search_swap(pts, max_iters=1000, time_limit=5.0):
    """
    Discrete local search: try swapping one selected point with one unselected point
    if it improves the minimum pairwise distance.
    """
    start_time = time.time()
    n_pool = len(pts)
    selected = np.arange(N) # Indices into the pool? 
    # Wait, the input to this function should be the current set of points and the pool.
    # Let's define the interface clearly:
    # current_pts: (N, 3) array of selected points
    # pool: (M, 3) array of candidate points
    
    # Actually, it's easier to work with indices.
    # Let's refactor: The main loop maintains a set of indices into the pool.
    
    pass # Replaced by index-based logic in construct

def _optimize_set_indices(pool, indices, time_limit=5.0):
    """
    Refine the set of indices using swap-based local search.
    Objective: Maximize the minimum pairwise distance among the selected points.
    """
    start_time = time.time()
    current_indices = indices.copy()
    
    # Precompute distance matrix for the pool
    d_pool = cdist(pool, pool)
    
    # Current min distance
    def get_min_dist(idxs):
        d_sub = d_pool[np.ix_(idxs, idxs)]
        np.fill_diagonal(d_sub, np.inf)
        return d_sub.min()
    
    current_min_d = get_min_dist(current_indices)
    
    improved = True
    iters = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        iters += 1
        if iters > 500: break # Safety
        
        # Try swapping each selected point with each unselected point
        selected_set = set(current_indices)
        unselected_list = [i for i in range(len(pool)) if i not in selected_set]
        
        # To speed up, we can sample swaps or iterate all.
        # N=14, M~1000. 14 * 1000 = 14000 swaps to check per iteration.
        # Each check is O(1) if we update min distance efficiently?
        # Actually, recalculating min distance for the new set is O(N^2) = 196 ops.
        # 14000 * 196 = 2.7M ops per iteration. Very fast.
        
        best_swap = None
        best_min_d = current_min_d
        
        for i_sel in range(N):
            for i_unsel in unselected_list:
                # Simulate swap
                new_indices = current_indices.copy()
                new_indices[i_sel] = i_unsel
                
                # Calculate new min distance
                # We can optimize this by noting that only distances involving i_unsel change.
                # But for simplicity and correctness, let's just compute.
                # Optimization: 
                # The new min distance is min(
                #   min distance among (current \ {i_sel})
                #   min distance from i_unsel to (current \ {i_sel})
                # )
                
                # Let's implement the fast check.
                # 1. Min distance among the remaining N-1 selected points
                # This is expensive to compute from scratch every time.
                # Let's precompute the min distance for all subsets of size N-1? No.
                
                # Simpler fast check:
                # d_new = d_pool[i_unsel, current_indices]
                # d_new[i_sel] = np.inf # Remove self
                # d_to_others = d_new.min()
                
                # We also need the min distance among the *other* N-1 points.
                # Let's precompute a matrix of min distances excluding one point?
                # Or just accept O(N^2) for the check if it's fast enough in numpy.
                
                # Let's try the direct numpy calculation for the new set's min dist.
                # It might be slow if we do it in a python loop.
                # Let's vectorize the swap evaluation?
                
                # Alternative: Just iterate swaps and compute min dist.
                # To make it faster, we can precompute the "min distance to all other selected points" for each selected point.
                
                # Let's stick to a simpler but effective heuristic:
                # Only try swaps that seem promising.
                # Or, just do the full O(N^2) check in numpy.
                
                # Vectorized check for one swap:
                # new_pts_idx = np.delete(current_indices, i_sel)
                # new_pts_idx = np.append(new_pts_idx, i_unsel)
                
                # This is getting complex. Let's use a simpler local search:
                # "Best improvement" search.
                
                # For each selected point i, find the unselected point j that maximizes
                # the minimum distance to the *other* selected points, subject to
                # not decreasing the global min distance too much.
                
                # Let's just do a random sampling of swaps to speed up.
                pass # This loop is too complex to implement correctly in short time.
        
        # Fallback: Use a simpler local search:
        # Try to move each selected point to the "best" unselected point that increases the min distance.
        
        # Let's implement a robust but simple swap search:
        # 1. Compute current min distance.
        # 2. For each pair (i, j) where i is selected, j is unselected:
        #    - Temporarily swap.
        #    - Compute new min distance.
        #    - If better, keep track.
        # 3. Apply best swap.
        
        # To avoid O(N*M*N^2) cost, we can limit M or use heuristics.
        # With M=1000, N=14, N*M = 14000.
        # If each check takes ~10us (numpy overhead), 14000 * 10us = 140ms per iteration.
        # 500 iterations = 70s. Too slow.
        
        # We need to be smarter.
        # Let's restrict the pool to the top 100 candidates closest to the "gaps"?
        # Or just use a smaller pool for local search.
        
        break # Stop this complex loop, implement simpler below
    
    return current_indices

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    # 1. Generate Pool
    pool = _generate_candidate_pool()
    M = len(pool)
    
    # Precompute distance matrix for the pool
    d_pool = cdist(pool, pool)
    
    best_pts = None
    best_score = np.inf
    
    # 2. Multi-start Greedy Selection
    # We will try multiple random start points for the greedy algorithm.
    num_starts = 50
    
    for start_idx in range(num_starts):
        if time.time() - _t0 > DEADLINE - 20:
            break
            
        # Greedy select
        selected_indices = [start_idx]
        
        for _ in range(N - 1):
            # Calculate min distance from each unselected point to selected set
            min_dists = np.full(M, np.inf)
            for s_idx in selected_indices:
                min_dists = np.minimum(min_dists, d_pool[:, s_idx])
                
            for s_idx in selected_indices:
                min_dists[s_idx] = -1
                
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)
            
        # Convert indices to points
        pts = pool[selected_indices]
        
        # Evaluate score
        score = _true_score(pts)
        
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
            
        # 3. Local Search Refinement
        # Try to improve the current set by swapping points.
        # We will perform a limited number of swap iterations.
        current_indices = np.array(selected_indices)
        current_min_d = np.inf
        d_sub = d_pool[np.ix_(current_indices, current_indices)]
        np.fill_diagonal(d_sub, np.inf)
        current_min_d = d_sub.min()
        
        # Simple swap search
        # We will try 100 random swaps or best improvement among a subset.
        # To be fast, we will try swapping each selected point with the top K unselected points
        # that are farthest from the rest.
        
        selected_set = set(current_indices)
        
        # Identify "promising" unselected points:
        # Those with high min distance to the current selected set.
        min_dists_to_selected = np.full(M, np.inf)
        for s_idx in current_indices:
            min_dists_to_selected = np.minimum(min_dists_to_selected, d_pool[:, s_idx])
            
        for s_idx in current_indices:
            min_dists_to_selected[s_idx] = -1
            
        # Get top 20 unselected points by min distance to selected set
        top_candidates_idx = np.argsort(min_dists_to_selected)[-20:]
        
        improved = True
        local_iters = 0
        
        while improved and local_iters < 50 and (time.time() - _t0 < DEADLINE - 10):
            improved = False
            local_iters += 1
            
            best_swap_gain = 0
            best_swap_i = -1
            best_swap_j = -1
            
            for i in range(N):
                # Try swapping current_indices[i] with each candidate j
                for j in top_candidates_idx:
                    if j in selected_set:
                        continue
                    
                    # Simulate swap
                    new_indices = current_indices.copy()
                    new_indices[i] = j
                    
                    # Calculate new min distance
                    # Fast check:
                    # 1. Min distance among the other N-1 points (unchanged from current, except if i was the bottleneck)
                    # 2. Min distance from j to the other N-1 points
                    
                    # Let's just compute the full min distance for the new set.
                    # Since N=14, this is fast.
                    d_new = d_pool[np.ix_(new_indices, new_indices)]
                    np.fill_diagonal(d_new, np.inf)
                    new_min_d = d_new.min()
                    
                    gain = new_min_d - current_min_d
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_i = i
                        best_swap_j = j
                        
            if best_swap_i != -1:
                # Apply swap
                current_indices[best_swap_i] = best_swap_j
                selected_set.remove(current_indices[best_swap_i] if current_indices[best_swap_i] != best_swap_j else best_swap_j)
                # Update set properly
                selected_set = set(current_indices)
                
                # Update current min distance
                d_sub = d_pool[np.ix_(current_indices, current_indices)]
                np.fill_diagonal(d_sub, np.inf)
                current_min_d = d_sub.min()
                
                improved = True
                
                # Update best_pts if this set is better
                pts = pool[current_indices]
                score = _true_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
    # Fallback
    if best_pts is None:
        # Just return a Fibonacci sphere
        best_pts = _generate_fibonacci_sphere(N, seed=0)
        
    # Final Check
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        best_pts += rng.normal(0, 1e-4, size=(N, D))
        
    return best_pts
