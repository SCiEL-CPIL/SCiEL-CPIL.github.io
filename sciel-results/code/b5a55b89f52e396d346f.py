import numpy as np
import itertools
import time
import scipy.sparse as sp

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate Type 1 vectors
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    
    # 2. Generate Type 2 vectors
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t2.append(v)
            
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    N = len(V_t2)
    
    # 3. Compute Conflict Matrix (Sparse)
    dots = V_t2 @ V_t2.T
    # Create sparse matrix from dense bool? Or build directly?
    # Building directly is faster if we know structure, but dense dot is fast.
    # 5280x5280 dense dot is ~0.1s.
    # Conversion to sparse is fast.
    conflict_dense = dots > 2
    np.fill_diagonal(conflict_dense, False)
    conflict_sparse = sp.csr_matrix(conflict_dense)
    
    # Precompute neighbors
    # neighbors[i] is array of indices j such that conflict[i,j] is True
    neighbors = [conflict_sparse[i].indices for i in range(N)]
    
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    n_restarts = 2000 # Can afford more restarts with sparse ops
    
    for r in range(n_restarts):
        if time.time() - start_time > 115:
            break
            
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        # Degrees in the alive subgraph
        # Initial degrees
        degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=np.int32)
        # Note: degrees change as we remove nodes.
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            # Find min degree among alive
            # This is O(N) scan
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            # Get neighbors of chosen that are still alive
            nbrs = neighbors[chosen]
            # Filter to alive
            # nbrs is an array of ints
            # We need to remove chosen and nbrs from alive
            
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            # Only remove if alive?
            # alive[to_remove] = False works even if already False
            
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            # A node u loses degree if its neighbor v is removed.
            # We removed `to_remove`.
            # For each u in alive, deg[u] -= count of neighbors in to_remove
            
            # Efficient update:
            # Iterate over removed nodes? No, too slow.
            # Iterate over alive nodes?
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # For each removed node r, decrement degree of its alive neighbors
                # This is sum of adjacency rows of removed nodes, restricted to rem
                # Using sparse matrix:
                # sub_matrix = conflict_sparse[rem][:, to_remove]
                # deg_decrement = sub_matrix.sum(axis=1)
                
                # This creates a sparse matrix slice.
                # Might be slow.
                
                # Alternative:
                # Just recompute degrees?
                # degrees[rem] = conflict_sparse[rem][:, alive].sum(axis=1)
                # This is O(N * avg_deg).
                
                # Let's try the slice sum.
                # conflict_sparse is CSR.
                # Row slice is fast. Col slice is slow.
                # So conflict_sparse[:, to_remove] is slow.
                
                # Better:
                # Iterate over to_remove, and for each, get its neighbors, and decrement degrees of alive neighbors.
                # Since degree is low (~56), this is fast.
                
                for r_idx in to_remove:
                    if not alive[r_idx]: # Already processed? No, we just set it False.
                        # But we might have multiple to_remove.
                        # We need to decrement degrees of neighbors of r_idx.
                        pass
                    
                # Actually, we can just do:
                # For each u in rem:
                #   degrees[u] -= sum(1 for v in neighbors[u] if not alive[v])
                # This is O(N * deg).
                
                # Let's just recompute degrees for rem using sparse matrix multiplication?
                # No, simple loop is fine for N=5000.
                
                # Optimization:
                # The number of alive nodes decreases.
                # We can just recalculate degrees for the remaining alive nodes from scratch?
                # No, that's O(N^2).
                
                # Let's stick to the dense update if sparse is tricky.
                # With N=5280, dense update `conflicts[rem][:, to_remove].sum(axis=1)` is fast.
                # Let's use the dense matrix for updates, it's simpler and likely fast enough for 5000x5000.
                
                pass # Skip sparse optimization for now, use dense as in seed but optimized.
                
        # Revert to dense logic for safety and speed of implementation
        # The seed used dense. It worked.
        # I will just use dense but with the decomposition.

    # RE-IMPLEMENTATION WITH DENSE (SAFER)
    
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    n_restarts = 800
    
    for r in range(n_restarts):
        if time.time() - start_time > 115:
            break
            
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflict_dense.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = np.where(conflict_dense[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_dense[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        
        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
