import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # Build support adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Build conflict adjacency using support structure
    # Two vectors conflict if their dot product > 2
    # Only need to check pairs of vectors whose supports share >= 3 elements
    
    adj = [[] for _ in range(n_total)]
    
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                
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
    
    # Build conflict bitsets for fast compatibility checks
    # conflict_mask[i] is a boolean array where conflict_mask[i][j] = True if i conflicts with j
    conflict_mask = np.zeros((n_total, n_total), dtype=bool)
    for i in range(n_total):
        if adj[i]:
            conflict_mask[i, adj[i]] = True
    
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    # Phase 1: Greedy Minimum Degree MIS with random restarts
    max_greedy_restarts = 200
    
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
            
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k = np.sum(selected)
        
        # Phase 2: Fast local search using conflict bitsets
        # Try to add nodes that are compatible with all selected nodes
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            # Find unselected nodes that don't conflict with any selected node
            # A node i is addable if no selected node conflicts with i
            # conflict_mask[i, selected] should be all False
            
            unselected = ~selected
            # For each unselected node, check if it conflicts with any selected node
            # This is: conflict_mask[unselected_indices, :].any(axis=1) & selected
            
            # More efficient: check if conflict_mask[i] & selected has any True
            # We can do this vectorized
            unselected_indices = np.where(unselected)[0]
            if len(unselected_indices) == 0:
                break
            
            # For each unselected node, check conflicts with selected nodes
            # conflict_mask[unselected_indices][:, selected] -> shape (n_unsel, n_sel)
            # This could be large, so let's do it in chunks
            chunk_size = 500
            for start in range(0, len(unselected_indices), chunk_size):
                end = min(start + chunk_size, len(unselected_indices))
                chunk = unselected_indices[start:end]
                
                # Check if any selected node conflicts with these
                # conflict_mask[chunk, :] & selected -> (chunk_size, n_total)
                # We want: ~conflict_mask[chunk, :].any(axis=1) for selected positions only
                # Simpler: check if conflict_mask[chunk][:, selected].any(axis=1) is False
                sel_indices = np.where(selected)[0]
                if len(sel_indices) == 0:
                    # No selected nodes, can add all unselected
                    for idx in chunk:
                        selected[idx] = True
                        improved = True
                    break
                
                # Check conflicts
                conflicts_with_selected = conflict_mask[chunk][:, sel_indices].any(axis=1)
                addable = ~conflicts_with_selected
                
                for i, add in enumerate(addable):
                    if add:
                        selected[chunk[i]] = True
                        improved = True
        
        k = np.sum(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
        
        # Phase 3: Swap-based local search
        # Try removing one selected node and see if we can add more than one
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            continue
        
        for swap_iter in range(30):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                break
            
            # Try removing a random selected node
            rem = sel_indices[rng.integers(len(sel_indices))]
            selected[rem] = False
            
            # Try to add compatible nodes
            improved = True
            while improved:
                improved = False
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                
                unselected = ~selected
                unselected_indices = np.where(unselected)[0]
                if len(unselected_indices) == 0:
                    break
                
                sel_indices_now = np.where(selected)[0]
                if len(sel_indices_now) == 0:
                    for idx in unselected_indices:
                        selected[idx] = True
                        improved = True
                    break
                
                chunk_size = 500
                for start in range(0, len(unselected_indices), chunk_size):
                    end = min(start + chunk_size, len(unselected_indices))
                    chunk = unselected_indices[start:end]
                    
                    conflicts_with_selected = conflict_mask[chunk][:, sel_indices_now].any(axis=1)
                    addable = ~conflicts_with_selected
                    
                    for i, add in enumerate(addable):
                        if add:
                            selected[chunk[i]] = True
                            improved = True
            
            k_new = np.sum(selected)
            if k_new > k:
                k = k_new
                if k > best_k:
                    best_k = k
                    best_selected = selected.copy()
                break  # Restart with better solution
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
