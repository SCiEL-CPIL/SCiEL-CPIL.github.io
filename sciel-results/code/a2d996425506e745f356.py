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
    n_supports = len(supports)  # 330
    
    # Build support adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    support_adj = [[] for _ in range(n_supports)]
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_adj[i].append(j)
                support_adj[j].append(i)
    
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
    n_total = len(all_vectors_np)  # 5280
    
    # Build vector-level conflict graph (adjacency list)
    # Two vectors conflict if their dot product > 2
    # We only need to check pairs of vectors whose supports share >= 3 elements
    
    # Create a set of support pairs that share 3 elements
    support_pairs = set()
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_pairs.add((min(i, j), max(i, j)))
    
    # Build vector-level adjacency list
    adj = [[] for _ in range(n_total)]
    
    for s1, s2 in support_pairs:
        base1 = s1 * 16
        base2 = s2 * 16
        
        v1 = all_vectors_np[base1:base1+16]
        v2 = all_vectors_np[base2:base2+16]
        
        d = v1 @ v2.T
        
        indices = np.argwhere(d > 2)
        for i, j in indices:
            idx1 = base1 + i
            idx2 = base2 + j
            adj[idx1].append(idx2)
            adj[idx2].append(idx1)
    
    # Convert adjacency list to sorted lists
    for i in range(n_total):
        adj[i].sort()
    
    # Precompute conflict mask as a sparse boolean matrix for fast vectorized checks
    # conflict_mask[i] is a boolean array of length n_total
    # Using a list of boolean arrays for memory efficiency
    # Actually, let's use a different approach: precompute for each node,
    # the set of its neighbors as a numpy boolean array
    
    # For efficient local search, precompute neighbor masks
    # This uses 5280 * 5280 / 8 bytes ≈ 3.5 MB, which is fine
    conflict_mask = np.zeros((n_total, n_total), dtype=bool)
    for i in range(n_total):
        if adj[i]:
            conflict_mask[i, adj[i]] = True
    
    # Also build a "compatible with selected" check:
    # A node i can be added if no selected node is in adj[i]
    # i.e., ~conflict_mask[i, selected].all() for selected nodes... 
    # Actually: i is compatible if conflict_mask[i, selected].sum() == 0
    
    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    # Strategy: Multiple approaches
    # 1. Greedy minimum-degree MIS (fast, many restarts)
    # 2. Vectorized local search
    
    max_greedy_restarts = 500
    
    # Phase 1: Greedy Minimum Degree MIS with random restarts
    for restart in range(max_greedy_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        # Greedy Minimum Degree MIS
        while avail.any():
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            
            min_rel = np.where(degs == min_deg)[0]
            
            pick_rel = min_rel[rng.integers(len(min_rel))]
            u = avail_indices[pick_rel]
            
            selected[u] = True
            avail[u] = False
            
            # Remove neighbors and update degrees
            neighbors = adj[u]
            for v in neighbors:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k = np.sum(selected)
        
        # Phase 2: Vectorized local search to improve the solution
        # Try to add nodes that are compatible with all selected nodes
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            # Find all unselected nodes that are compatible with all selected nodes
            # A node i is compatible if conflict_mask[i, selected].sum() == 0
            # i.e., no selected node is a neighbor of i
            
            # Vectorized check: for each unselected node, check if it conflicts with any selected node
            # conflict_mask[unselected, selected].any(axis=1) gives True for conflicting nodes
            # We want nodes where this is False
            
            unselected = np.where(~selected)[0]
            if len(unselected) == 0:
                break
            
            # Check which unselected nodes conflict with selected nodes
            # conflict_mask[i, selected] for i in unselected
            # We can do: conflict_mask[np.ix_(unselected, np.where(selected)[0])].any(axis=1)
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                # No selected nodes, add all
                selected[unselected] = True
                improved = True
                continue
            
            conflicts = conflict_mask[np.ix_(unselected, sel_indices)].any(axis=1)
            # Nodes where conflicts is False can be added
            addable = unselected[~conflicts]
            
            if len(addable) > 0:
                selected[addable] = True
                improved = True
        
        k = np.sum(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
    
    # Phase 3: Try swap-based local search on the best solution
    if best_selected is not None and time.time() - start_time < deadline - 20:
        for _ in range(30):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            selected = best_selected.copy()
            k = np.sum(selected)
            
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                break
            
            # Try removing a random selected node and re-adding compatible ones
            for _ in range(10):
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                
                rem = sel_indices[rng.integers(len(sel_indices))]
                selected[rem] = False
                
                # Vectorized: find all nodes compatible with current selected
                sel_indices_now = np.where(selected)[0]
                unselected = np.where(~selected)[0]
                
                if len(unselected) == 0:
                    continue
                
                if len(sel_indices_now) == 0:
                    selected[unselected] = True
                else:
                    conflicts = conflict_mask[np.ix_(unselected, sel_indices_now)].any(axis=1)
                    addable = unselected[~conflicts]
                    if len(addable) > 0:
                        selected[addable] = True
                
                k_new = np.sum(selected)
                if k_new > k:
                    k = k_new
                    if k > best_k:
                        best_k = k
                        best_selected = selected.copy()
                    break
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
