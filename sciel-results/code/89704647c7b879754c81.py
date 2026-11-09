"""Model-and-Optimize: Build a high-quality reference tour using Cheapest Insertion 
and restricted 2-opt local search on a k-NN graph, then promote its edges in the heuristic matrix."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    # Precompute k-NN graph (k=6 is standard for TSP in unit square)
    k = 6
    # For each node, find k nearest neighbors
    # D is symmetric
    nn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        # Sort neighbors by distance
        # Exclude self
        dists = D[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        nn_mask[i, nn_indices] = True
        nn_mask[nn_indices, i] = True # Symmetric
        
    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt_restricted(t, max_iter=500):
        """
        2-opt restricted to k-NN graph edges.
        We only accept a move if the new edges (a,c) and (b,d) are in the k-NN graph.
        This keeps the tour "local" and avoids long crossings.
        """
        t = list(t)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() > deadline:
                return t
            
            # Vectorized delta calculation
            # We iterate i, and for each i, calculate deltas for all j > i+1
            # But we must check if D[a,c] and D[b,d] are allowed (in k-NN)
            # Since checking k-NN membership is O(1) with a boolean mask, we can do this.
            
            # Optimization: Pre-filter js for a given i? 
            # For a fixed i (a=t[i], b=t[i+1]), we want j such that 
            # nn_mask[a, t[j]] and nn_mask[b, t[j+1]] are True.
            
            for i in range(n - 2):
                if time.time() > deadline:
                    return t
                a = t[i]
                b = t[i + 1]
                
                # j ranges from i+2 to n-1
                js = np.arange(i + 2, n)
                if len(js) == 0: continue
                
                c_indices = np.array([t[j] for j in js])
                d_indices = np.array([t[(j + 1) % n] for j in js])
                
                # Check k-NN validity
                # Edge (a, c) must be in k-NN
                valid_ac = nn_mask[a, c_indices]
                # Edge (b, d) must be in k-NN
                valid_bd = nn_mask[b, d_indices]
                
                if not np.any(valid_ac & valid_bd):
                    continue
                    
                valid_indices = np.where(valid_ac & valid_bd)[0]
                if len(valid_indices) == 0:
                    continue
                
                c_valid = c_indices[valid_indices]
                d_valid = d_indices[valid_indices]
                js_valid = js[valid_indices]
                
                # Delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                deltas = D[a, c_valid] + D[b, d_valid] - D[a, b] - D[c_valid, d_valid]
                
                min_delta = np.min(deltas)
                if min_delta < -1e-10:
                    best_idx = np.argmin(deltas)
                    j = js_valid[best_idx]
                    # Reverse segment t[i+1 : j+1]
                    t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
                    improved = True
                    # After a reversal, restart the i-loop
                    break 
        return t

    def cheapest_insertion(rng):
        """Construct a tour using Cheapest Insertion heuristic."""
        i = int(rng.integers(n))
        d = D[i, :].copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        
        tour = [i, j]
        visited = {i, j}
        
        while len(tour) < n:
            unvisited = [k for k in range(n) if k not in visited]
            if not unvisited:
                break
            
            best_cost = float('inf')
            best_node = -1
            best_pos = -1
            
            for node in unvisited:
                # Try inserting node at every position
                for pos in range(len(tour)):
                    prev_node = tour[pos - 1]
                    next_node = tour[pos]
                    delta = D[prev_node, node] + D[node, next_node] - D[prev_node, next_node]
                    if delta < best_cost:
                        best_cost = delta
                        best_node = node
                        best_pos = pos
                        
            if best_node != -1:
                tour.insert(best_pos, best_node)
                visited.add(best_node)
                
        return tour

    def refine(tour):
        """Apply restricted 2-opt improvements."""
        t = two_opt_restricted(list(tour))
        # A second pass with relaxed k-NN (e.g., k=8) or unrestricted if time permits?
        # Let's stick to restricted for speed.
        return t

    # --- Main Optimization Loop ---
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    num_starts = 0
    # Run as many starts as possible
    while time.time() < deadline - 10.0:
        num_starts += 1
        tour = cheapest_insertion(rng)
        tour = refine(tour)
        L = tour_len(tour)
        
        if L < best_len:
            best_len = L
            best_tour = tour
            
        # If we found a very good tour (close to lower bound?), we might want to stop earlier?
        # Lower bound for n=50 uniform random is roughly ~2.5-2.7.
        # If best_len < 2.6, it's likely optimal or near-optimal.
        if best_len < 2.6:
             # Do a few more to be sure, but break if stable
             if num_starts > 50:
                 break

    # Final polish: Run a more aggressive local search on the best tour found
    # Maybe use a larger k-NN or unrestricted 2-opt for the final candidate
    if best_tour is not None:
        # Unrestricted 2-opt for final polish
        def two_opt_unrestricted(t, max_iter=200):
            t = list(t)
            improved = True
            iters = 0
            while improved and iters < max_iter:
                improved = False
                iters += 1
                if time.time() > deadline:
                    return t
                for i in range(n - 2):
                    if time.time() > deadline:
                        return t
                    a = t[i]
                    b = t[i + 1]
                    js = np.arange(i + 2, n)
                    if len(js) == 0: continue
                    c_indices = np.array([t[j] for j in js])
                    d_indices = np.array([t[(j + 1) % n] for j in js])
                    deltas = D[a, c_indices] + D[b, d_indices] - D[a, b] - D[c_indices, d_indices]
                    min_delta = np.min(deltas)
                    if min_delta < -1e-10:
                        best_idx = np.argmin(deltas)
                        j = js[best_idx]
                        t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
                        improved = True
                        break
            return t

        best_tour = two_opt_unrestricted(best_tour)
        best_len = tour_len(best_tour)

    if best_tour is None:
        best_tour = list(range(n))
        best_tour = refine(best_tour)
        best_len = tour_len(best_tour)

    # --- Build Heuristic Matrix ---
    diag = np.eye(n, dtype=bool)
    safe_D = D.copy()
    safe_D[diag] = 1.0
    
    # Base heuristic: Inverse distance
    base = 1.0 / safe_D
    
    # Promote reference tour edges
    # Use a large factor to dominate, but keep it finite
    # 1e4 is standard in previous iterations. Let's try 1e5 for stronger guidance.
    for k_idx in range(n):
        i = best_tour[k_idx]
        j = best_tour[(k_idx + 1) % n]
        base[i, j] = 1.0e5

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
# EVOLVE-BLOCK-END
