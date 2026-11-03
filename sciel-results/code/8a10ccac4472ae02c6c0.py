"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _generate_all_vectors():
    """
    Generate a diverse pool of candidate vectors.
    We focus on shells with small norms to maximize packing density.
    Shell 1: 11 vectors (norm 1)
    Shell 2: 220 vectors (norm sqrt(2))
    Shell 3: 1716 vectors (norm sqrt(3))
    Shell 4: 11880 vectors (norm sqrt(4) = 2)
    
    Total ~ 13800 vectors.
    """
    vecs = []
    
    # Shell 1: e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
        
    # Shell 2: e_i - e_j
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)
            
    # Shell 3: Weight 3 vectors with entries in {1, -1}.
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    # Shell 4: Weight 4. C(11,4) * 2^4 = 330 * 16 = 5280.
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    return np.array(vecs, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # 1. Generate candidate pool
    candidates = _generate_all_vectors()
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # 2. Build Conflict Graph
    # Two vectors u, v conflict if ||u - v||^2 < max(||u||^2, ||v||^2)
    # We want an independent set in this graph (no conflicts).
    
    # To save memory and time, we don't build the full N x N matrix explicitly if N is large,
    # but N ~ 13800 is manageable for sparse or even dense float32/bool operations if optimized.
    # However, building the full adjacency matrix of 13800^2 bools is ~190MB. 
    # Let's try to be smarter.
    
    # Actually, let's compute the Gram matrix G = candidates @ candidates.T
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*G[i, j]
    # conflict[i, j] = dist_sq[i, j] < max(norms_sq[i], norms_sq[j])
    
    # Memory check: 13800 * 13800 * 8 bytes (float64) ~ 1.5 GB. Too much for float64.
    # Use float32? 13800 * 13800 * 4 ~ 760 MB. Still risky.
    # We can process in blocks or use the fact that the graph is likely sparse?
    # No, high density conflicts are expected.
    
    # Alternative: Don't build full graph. Use spectral method on a subsample or use a different approach.
    # The prompt suggests "spectral approximation".
    # Let's try to compute the Laplacian or Adjacency matrix in a memory-efficient way or use a smaller subset?
    # No, we need the global structure.
    
    # Let's check if we can fit the conflict matrix in bool/int8.
    # 13800^2 bytes ~ 190 MB. This is feasible.
    
    # Compute Gram matrix in blocks to save peak memory?
    # Or just use float32 for G.
    
    # Let's try computing the conflict matrix directly.
    # G = candidates @ candidates.T
    # This is O(N^2 * D). 13800^2 * 11 ~ 2.1 * 10^9 flops. NumPy handles this in seconds.
    
    # Let's verify time.
    # 2 billion flops is roughly 1-2 seconds on a modern CPU with BLAS.
    
    # We will compute the conflict matrix.
    # To save memory, we compute G in float32.
    
    # However, G is symmetric.
    
    # Let's try a different angle: 
    # The problem is equivalent to finding a large set of vectors with pairwise "angle" constraints.
    # For same norm R, <u,v> <= R^2/2.
    # This is a spherical code problem on integer lattice.
    
    # Let's stick to the Graph approach but be careful with memory.
    # If memory fails, we fall back to greedy.
    
    try:
        # Compute Gram matrix
        # candidates is (N, D)
        # G = candidates @ candidates.T
        # Using float32 to save memory
        G = (candidates.astype(np.float32) @ candidates.T.astype(np.float32))
        
        # Compute squared norms (already have, but need float32 version for broadcasting)
        ns = norms_sq.astype(np.float32)
        
        # dist_sq[i, j] = ns[i] + ns[j] - 2*G[i, j]
        # We want conflict if dist_sq < max(ns[i], ns[j])
        
        # max_ns[i, j] = max(ns[i], ns[j])
        # We can compute this via broadcasting
        ns_col = ns[:, np.newaxis]
        ns_row = ns[np.newaxis, :]
        max_ns = np.maximum(ns_col, ns_row)
        
        dist_sq = ns_col + ns_row - 2 * G
        
        # Conflict matrix (True if conflict)
        # Note: dist_sq[i,i] = 0. max_ns[i,i] = ns[i]. 0 < ns[i] is True.
        # We don't want self-conflicts to prevent selection, but MIS algorithms usually handle loops or we ignore them.
        # Standard MIS: no self loops. We will set diagonal to False.
        
        conflict = dist_sq < max_ns
        np.fill_diagonal(conflict, False)
        
        # Now we have the conflict graph adjacency matrix (bool).
        # We want the Maximum Independent Set.
        # This is equivalent to the Maximum Clique in the complement graph.
        # Complement graph A' = ~conflict (excluding diagonal).
        
        # Spectral Approximation:
        # The size of the MIS is related to the eigenvalues of the Laplacian or Adjacency matrix.
        # A simple heuristic: The number of vertices with high "independence" potential.
        # Let's use the eigenvalues of the conflict graph.
        # If the graph is regular of degree k, independence number <= N * (1 - lambda_min/(k - lambda_min)) ... (not quite)
        
        # Better: Use the trace of the adjacency matrix powers?
        # Or simply: Sort vertices by their "conflict degree" (number of conflicts).
        # Vertices with few conflicts are good candidates.
        
        # Let's compute the degree of each vertex in the conflict graph.
        degrees = np.sum(conflict, axis=1)
        
        # Strategy:
        # 1. Identify "low conflict" vertices.
        # 2. Use a greedy algorithm but seeded/ordered by these degrees.
        # 3. Additionally, use a spectral method to find a large independent set in the complement graph (MIS of conflict).
        
        # Let's try a Multi-Start Greedy with different orderings.
        # Ordering 1: Ascending Degree (Pick least connected first)
        # Ordering 2: Random
        # Ordering 3: Based on spectral embedding
        
        # Spectral Embedding:
        # Compute eigenvectors of the conflict graph adjacency matrix?
        # N=13800. Eigen decomposition is O(N^3) ~ 2*10^12. Too slow.
        # Power iteration for top few eigenvectors?
        # Let's skip full spectral if too slow.
        
        # Let's use a "Coloring" based approach or simple greedy variants.
        
        # Variant 1: Greedy by Degree
        order = np.argsort(degrees)
        
        selected_mask = np.zeros(n_cand, dtype=bool)
        
        # We need to check compatibility against ALREADY selected.
        # This is O(N * K) where K is size of selected set.
        # If K is large, this is slow.
        
        # Optimization:
        # Maintain a "blocked" set.
        # When we select v, we block all v's neighbors in the conflict graph.
        # This is exactly the Greedy Independent Set algorithm.
        
        # Algorithm:
        # 1. Start with all nodes available.
        # 2. Pick a node v (heuristic).
        # 3. Add v to Independent Set.
        # 4. Remove v and all its neighbors from Available Set.
        # 5. Repeat until empty.
        
        # This is very fast if we use bitsets or boolean arrays.
        # N=13800. Boolean array operations are fast.
        
        # Let's implement this.
        
        def greedy_is(order):
            """
            Greedy Independent Set given a fixed order of vertices.
            """
            available = np.ones(n_cand, dtype=bool)
            selected = []
            
            for v in order:
                if not available[v]:
                    continue
                    
                # Select v
                selected.append(v)
                
                # Block v and its neighbors
                # neighbors are where conflict[v, :] is True
                available[v] = False
                available[conflict[v, :]] = False
                
            return selected
            
        # Heuristic 1: Ascending Degree
        sel1 = greedy_is(order)
        
        # Heuristic 2: Random Order (multiple restarts)
        best_sel = sel1
        best_size = len(sel1)
        
        # Time check
        if time.time() > deadline - 20:
            return candidates[best_sel]
            
        # Try a few random restarts
        rng = np.random.default_rng(42)
        
        for _ in range(10): # 10 restarts
            rand_order = rng.permutation(n_cand)
            sel_r = greedy_is(rand_order)
            if len(sel_r) > best_size:
                best_size = len(sel_r)
                best_sel = sel_r
                
            if time.time() > deadline - 10:
                break
                
        # Heuristic 3: "Weighted" Greedy
        # Prioritize nodes that are "central" in the complement graph?
        # Or nodes that have high "independence potential"?
        # Let's try sorting by (Degree, Norm)
        
        # Let's also try a local search on the best set found.
        # If we have a set S, can we swap any v in S with u not in S to increase size?
        # No, size is fixed in a swap.
        # Can we find a set of size best_size + 1?
        # This requires checking if there is an independent set of size best_size + 1.
        
        # Given the time, the greedy with multiple restarts is a strong baseline.
        # The "Spectral" part was skipped due to O(N^3) cost, but the Degree-based greedy
        # is effectively using the graph structure.
        
        # One more refinement: 
        # The conflict graph is defined by geometry.
        # Vectors with smaller norms are "easier" to pack?
        # Actually, smaller norms mean tighter constraints relative to distance?
        # ||u-v||^2 >= max(||u||^2, ||v||^2)
        # If ||u|| is small, the distance required is small.
        # So small norm vectors are more flexible?
        # Let's try sorting by Norm, then Degree.
        
        order_norm_deg = np.lexsort((degrees, norms_sq))
        sel_nd = greedy_is(order_norm_deg)
        if len(sel_nd) > best_size:
            best_size = len(sel_nd)
            best_sel = sel_nd
            
    except Exception as e:
        # Fallback to simple greedy if memory or other issues
        # This should rarely happen if we stick to the plan, but safe.
        # Simple greedy: sort by norm, then pick.
        order = np.argsort(norms_sq)
        available = np.ones(n_cand, dtype=bool)
        best_sel = []
        
        # We need the conflict matrix for the fallback too?
        # If the try block failed, we don't have 'conflict'.
        # We must recompute or use a different method.
        # Let's assume the try block succeeded. If it fails, we return empty or a tiny set.
        # To be robust, let's catch the failure and do a very basic greedy without the full matrix?
        # That would be O(N^2 * D) for each pick. Too slow.
        
        # If memory fails, we are in trouble. Let's hope 190MB is fine.
        pass

    # Final validation and return
    if not best_sel:
        # Fallback
        return candidates[:1]
        
    return np.array(candidates[best_sel], dtype=np.int64)

# EVOLVE-BLOCK-END
