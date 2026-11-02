"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # Focus on weight-4 vectors (4 non-zero coordinates, values +/-1).
    # Norm^2 = 4.
    # Condition for compatibility: ||u - v||^2 >= max(||u||^2, ||v||^2)
    # For weight-4 vectors: 4 + 4 - 2 u.v >= 4 => 8 - 2 u.v >= 4 => u.v <= 2.
    # Conflict if u.v > 2.
    
    candidates = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # 2. Build Conflict Graph
    # Compute Gram matrix G = candidates @ candidates.T
    G = candidates @ candidates.T
    
    # Conflict if G[i, j] > 2
    # We want to find a large Independent Set in the Conflict Graph.
    # This is equivalent to finding a large Clique in the Compatibility Graph.
    # Compatibility Graph Edge[i, j] = (G[i, j] <= 2) AND (i != j)
    
    # Spectral Relaxation for Max Independent Set (or Max Clique in Compat Graph)
    # We can use the Laplacian of the Compatibility Graph or Conflict Graph.
    # Let's use the Conflict Graph Laplacian.
    # L = D - A_conflict
    # The smallest non-zero eigenvalue (Fiedler value) gives a partition.
    # However, for a large independent set, we can look at the eigenvector
    # corresponding to the smallest eigenvalue of the Conflict Graph's Laplacian?
    # No, that's for bipartition.
    
    # Alternative: Spectral Relaxation for Max Clique in Compatibility Graph.
    # Let A_comp be the adjacency matrix of the Compatibility Graph.
    # A_comp[i, j] = 1 if compatible, 0 otherwise.
    # We want x^T A_comp x to be large, with x in {0, 1}.
    # Relax to x in R, ||x|| = 1.
    # The solution is the leading eigenvector of A_comp.
    # Then, we can select the vertices with the largest absolute values in the eigenvector.
    
    # Build A_comp
    # A_comp = (G <= 2)
    # Diagonal must be 0.
    A_comp = (G <= 2).astype(np.float64)
    np.fill_diagonal(A_comp, 0)
    
    # Compute leading eigenvector of A_comp
    # Using scipy.sparse.linalg.eigsh for efficiency if possible, but A_comp is dense.
    # n_cand = 5280. 5280x5280 matrix.
    # np.linalg.eigh is O(N^3). 5280^3 ~ 1.5e11 ops. Too slow in Python/NumPy for 150s?
    # Let's check time.
    # Actually, 5000^3 is very large.
    # We can use power iteration or scipy.sparse.linalg.eigsh if we convert to sparse.
    # A_comp is dense? 
    # G[i, j] is sum of products. 
    # Two weight-4 vectors are compatible if they share at most 1 coordinate with same sign?
    # u.v = sum(u_k v_k).
    # If they share k coordinates, u.v is sum of signs.
    # Max u.v = 4 (if identical).
    # u.v > 2 means they share at least 3 coordinates with same sign?
    # Or 2 coords same sign, 0 diff? No.
    # If they share 2 coords with same sign and 2 with diff sign, u.v = 0.
    # If they share 3 coords with same sign, u.v = 3 (conflict).
    # If they share 4 coords with same sign, u.v = 4 (conflict).
    # If they share 3 coords with same sign and 1 diff? Not possible with weight 4?
    # If they share 3 coords, the 4th is different.
    # u.v = 3 (if signs match on 3) or -1 (if 2 match, 1 diff? No, 3 shared indices).
    # Let's just use the matrix.
    
    # To speed up eigendecomposition, we can use sparse matrix if A_comp is sparse.
    # Is A_comp sparse?
    # Number of conflicts per node:
    # How many vectors conflict with a given weight-4 vector?
    # Conflict if u.v > 2.
    # This likely means A_comp is dense (most vectors are compatible).
    # So we need dense eigendecomposition.
    # 5280^3 is too slow for 150s in pure Python/NumPy?
    # Let's try a smaller pool or a faster method.
    
    # Alternative: Use a subset of candidates?
    # Or use a greedy approach on the eigenvector values?
    
    # Let's try to compute the eigenvector.
    # If it's too slow, we fall back to greedy.
    
    try:
        # Use scipy.sparse.linalg.eigsh if available and matrix is sparse enough?
        # But it's dense.
        # Let's use np.linalg.eigh.
        # To save time, we can project onto a lower dimensional space?
        # Or just run it and see.
        
        # Actually, 5280 is large.
        # Let's reduce the pool size to make it feasible for dense eigendecomposition.
        # C(11, 4) * 16 = 5280.
        # Maybe we can restrict to a subset of signs?
        # Or use a different spectral method?
        
        # Let's try to compute the top eigenvector using power iteration, which is O(N^2) per iteration.
        # 5280^2 = 27M. 100 iterations = 2.7B ops. Might be ~10-20s in NumPy.
        
        # Power Iteration for leading eigenvector of symmetric matrix A_comp.
        x = np.random.randn(n_cand)
        x /= np.linalg.norm(x)
        
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            Ax = A_comp @ x
            x_new = Ax
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new /= norm
            if np.linalg.norm(x_new - x) < 1e-8:
                break
            x = x_new
            
        # The leading eigenvector x gives a relaxation.
        # We select vertices with the largest |x_i| values.
        # But we need them to be mutually compatible.
        # The eigenvector doesn't guarantee compatibility.
        # However, we can use it as a heuristic for ordering in a greedy algorithm.
        
        # Order candidates by |x_i| descending.
        order = np.argsort(-np.abs(x))
        
        # Greedy Independent Set on Conflict Graph using this order.
        avail_mask = np.ones(n_cand, dtype=bool)
        selected = []
        
        for v in order:
            if not avail_mask[v]:
                continue
            selected.append(v)
            # Remove v and its conflict neighbors from avail_mask
            # Conflict neighbors are where A_comp[v] == 0 (and v != u)
            # So conflict_mask = ~A_comp[v]
            # But A_comp[v, v] = 0, so v is in conflict_mask.
            conflict_mask = ~A_comp[v]
            avail_mask[conflict_mask] = False
            
        # Verify and return
        if not selected:
            return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
            
        final_vecs = candidates[selected]
        return final_vecs

    except Exception as e:
        # Fallback to simple greedy if spectral fails
        print(f"Spectral failed: {e}")
        # Simple greedy
        avail_mask = np.ones(n_cand, dtype=bool)
        selected = []
        for v in range(n_cand):
            if not avail_mask[v]:
                continue
            selected.append(v)
            conflict_mask = ~A_comp[v]
            avail_mask[conflict_mask] = False
            
        if not selected:
            return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        final_vecs = candidates[selected]
        return final_vecs

# EVOLVE-BLOCK-END
