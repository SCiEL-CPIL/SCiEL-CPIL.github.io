import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np) # 5280
    
    # Build adjacency list for vectors
    # Conflict if dot > 2
    # We can build this by iterating pairs of supports that share 3 elements.
    # Or just compute dots for all pairs? 5280^2 = 27M.
    # 27M dot products of length 11.
    # Vectorized: all_vectors_np @ all_vectors_np.T
    # This is 5280 x 5280 matrix.
    # 27M floats. ~200MB. Acceptable.
    
    dots = all_vectors_np @ all_vectors_np.T
    # Conflict if dot > 2
    # Note: diagonal is 4. We don't pick same node twice.
    # We want adj[i] = list of j such that dots[i,j] > 2.
    
    # To save memory and time, we can extract indices.
    # But building the list of lists is O(N^2) which is 27M checks.
    # In Python loop, this is slow.
    # Let's use NumPy to find indices.
    
    # conflict_mask = (dots > 2) & (np.arange(n_total)[:, None] != np.arange(n_total)[None, :])
    # This creates a huge boolean matrix.
    
    # Better: iterate by support.
    # Supports sharing 3 elements.
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    # Map support index to vector indices
    # Vectors for support s are at s*16 to s*16+15
    
    adj = [[] for _ in range(n_total)]
    
    # For each 3-subset, we have a list of supports containing it.
    # For each pair of supports (s1, s2) in this list, they share at least 3 elements.
    # Since they are distinct 4-subsets, they share exactly 3.
    # We need to check conflicts between vectors in s1 and s2.
    
    for tri, tets in tri_to_tet.items():
        # tets is list of support indices
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                
                base1 = s1 * 16
                base2 = s2 * 16
                
                # Get vectors
                v1 = all_vectors_np[base1:base1+16]
                v2 = all_vectors_np[base2:base2+16]
                
                # Compute dots between v1 and v2
                # 16x16 matrix
                d = v1 @ v2.T
                
                # Find conflicts: d > 2
                # For each i in 16, j in 16 where d[i,j] > 2:
                # adj[base1+i].append(base2+j)
                # adj[base2+j].append(base1+i)
                
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
    
    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        # Greedy Minimum Degree MIS
        while avail.any():
            # Find min degree among available
            # To avoid copying deg every time, we can use a masked approach
            # But deg is modified.
            # We need the min of deg where avail is True.
            
            # Optimization: 
            # If we keep a list of available indices, we can check their degrees.
            # But maintaining the list is hard.
            # Let's use np.where.
            
            # This is O(N) in C.
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            # Get degrees of available nodes
            # deg[avail_indices]
            # Find min
            # If ties, pick random one
            
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            
            # Find all indices with min_deg
            min_indices_rel = np.where(degs == min_deg)[0]
            
            # Pick one randomly
            pick_rel = min_indices_rel[rng.integers(len(min_indices_rel))]
            u = avail_indices[pick_rel]
            
            # Add u to solution
            selected[u] = True
            avail[u] = False
            
            # Update neighbors
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k = np.sum(selected)
        
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
            
        # Optional: Try to improve with local search?
        # Given time, skip for now.
        
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
