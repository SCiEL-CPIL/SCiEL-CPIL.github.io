import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115
    
    # We will try vectors with 5 non-zero entries of +/- 1.
    # Norm is sqrt(5).
    # Condition: ||u - v||^2 >= 5.
    
    subsets = list(itertools.combinations(range(D), 5))
    n_subsets = len(subsets)
    
    # Precompute masks for subsets
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Generate all vectors
    # vector_id = subset_id * 32 + sign_id
    # sign_id is 0..31 representing the 5 signs
    vectors = []
    subset_indices = [] # maps vector_id to subset_id
    sign_indices = []   # maps vector_id to sign_id
    
    for sid, s in enumerate(subsets):
        for sign_id in range(32):
            vid = sid * 32 + sign_id
            v = [0] * D
            # Map sign_id bits to positions in s
            # bit 0 of sign_id -> pos s[0], etc
            for i in range(5):
                if (sign_id >> i) & 1:
                    v[s[i]] = 1
                else:
                    v[s[i]] = -1
            vectors.append(v)
            subset_indices.append(sid)
            sign_indices.append(sign_id)
            
    N = len(vectors)
    
    # Build conflict graph
    # Conflict if ||u - v||^2 < 5
    # We can precompute for each pair of subsets whether they can have ANY conflict.
    # If min possible dist^2 >= 5, no conflict.
    # If max possible dist^2 < 5, all conflict.
    # Otherwise, check specific signs.
    
    # Precompute subset-pair relationships
    # rel[i][j] = (min_dist_sq, max_dist_sq) for supports i, j?
    # Actually, just build adjacency list directly.
    # N is 14784. N^2 is ~200M. Too slow in Python.
    # We need to iterate over subset pairs.
    
    adj = [set() for _ in range(N)]
    
    # Iterate over pairs of subsets
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            inter = (masks[i] & masks[j]).bit_count()
            
            # Calculate min and max squared distance
            # dist^2 = 2*(5-inter) + sum_{k in I} (u_k - v_k)^2
            # (u_k - v_k)^2 is 0 if signs match, 4 if signs differ.
            
            min_base = 2 * (5 - inter)
            max_base = 2 * (5 - inter) + 4 * inter
            
            if min_base >= 5:
                # All pairs are compatible
                continue
            if max_base < 5:
                # All pairs are incompatible
                for si in range(32):
                    vi = i * 32 + si
                    for sj in range(32):
                        vj = j * 32 + sj
                        adj[vi].add(vj)
                        adj[vj].add(vi)
                continue
            
            # Mixed case: need to check specific signs
            # Get positions in intersection
            # To be fast, we can just check all 32x32 pairs?
            # 1024 checks per subset pair.
            # Number of subset pairs is ~ 462*461/2 = 106k.
            # 106k * 1024 = 108M operations. Might be too slow in Python.
            
            # Optimization:
            # We only care about signs on the intersection I.
            # Let proj_i(s) be the signs of s on I.
            # Conflict condition depends only on proj_i and proj_j.
            
            # Find intersection positions
            inter_pos = []
            for p in range(D):
                if (masks[i] & (1 << p)) and (masks[j] & (1 << p)):
                    inter_pos.append(p)
            
            # Map global position to index in s[i] and s[j]
            # idx_in_i = position of p in subsets[i]
            # idx_in_j = position of p in subsets[j]
            
            # Precompute mapping for this pair
            map_i = {}
            map_j = {}
            for k, p in enumerate(subsets[i]):
                map_i[p] = k
            for k, p in enumerate(subsets[j]):
                map_j[p] = k
            
            # For each sign pair, check conflict
            # We can group signs by their projection on I
            # proj_i_val is a bitmask of length len(I)
            
            # Let's just loop 32x32 but with early exit?
            # Or precompute a 32x32 boolean matrix for this pair of supports?
            # Since I size varies, we can't reuse easily.
            
            # Better: Iterate si, sj.
            # Check if dist^2 < 5.
            
            for si in range(32):
                vi = i * 32 + si
                # Extract signs on I for vi
                # We need to compare with sj
                for sj in range(32):
                    vj = j * 32 + sj
                    # Calculate dist^2
                    # dist^2 = 2*(5-inter) + 4 * (number of sign mismatches on I)
                    mismatches = 0
                    for p in inter_pos:
                        # Get sign of vi at p
                        # p is at index map_i[p] in subsets[i]
                        # sign is bit map_i[p] of si
                        si_bit = (si >> map_i[p]) & 1
                        sj_bit = (sj >> map_j[p]) & 1
                        if si_bit != sj_bit:
                            mismatches += 1
                            
                    d2 = 2 * (5 - inter) + 4 * mismatches
                    if d2 < 5:
                        adj[vi].add(vj)
                        adj[vj].add(vi)
                        
    # Convert adj to lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    # Greedy MIS
    # Sort nodes by degree (ascending)
    # Pick node, remove it and its neighbors
    
    # To handle time, we do multiple restarts with random tie-breaking
    best_size = 0
    best_set = set()
    
    # Heuristic: 
    # 1. Greedy
    # 2. Local Search (1-swap)
    
    def greedy_mis(adj_list, N, deadline):
        remaining = set(range(N))
        selected = set()
        
        # Order nodes by degree in remaining graph?
        # Dynamic degree is expensive.
        # Static degree is a good proxy.
        degrees = [len(adj_list[i]) for i in range(N)]
        order = sorted(range(N), key=lambda x: degrees[x])
        
        for u in order:
            if time.time() > deadline:
                break
            if u not in remaining:
                continue
            
            selected.add(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in adj_list[u]:
                if v in remaining:
                    remaining.remove(v)
                    
        return selected
        
    def local_search(selected, adj_list, N, deadline, max_iter=1000):
        selected = set(selected)
        in_selected = np.zeros(N, dtype=bool)
        for u in selected:
            in_selected[u] = True
            
        # Precompute: for each node NOT in selected, 
        # gain = 1 - (number of neighbors in selected)
        # We want to add node u if gain > 0?
        # Actually, adding u removes all its neighbors from selected.
        # Change in size = 1 - count(neighbors(u) in selected)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add a node
            # Sort candidates by gain (desc)
            candidates = []
            for u in range(N):
                if in_selected[u]:
                    continue
                # Count neighbors in selected
                cnt = 0
                for v in adj_list[u]:
                    if in_selected[v]:
                        cnt += 1
                gain = 1 - cnt
                if gain > 0:
                    candidates.append((gain, u))
            
            if not candidates:
                break
                
            # Pick best candidate
            candidates.sort(reverse=True)
            best_gain, u = candidates[0]
            
            # Add u, remove neighbors
            new_selected = set(selected)
            new_selected.add(u)
            for v in adj_list[u]:
                if v in new_selected:
                    new_selected.remove(v)
                    
            if len(new_selected) > len(selected):
                selected = new_selected
                in_selected = np.zeros(N, dtype=bool)
                for u in selected:
                    in_selected[u] = True
                improved = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = set(selected)
                    
        return selected

    # Run greedy + local search multiple times
    # Since graph is static, we can use random shuffling for greedy
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Randomized greedy
        remaining = set(range(N))
        selected = set()
        
        # Shuffle order
        order = list(range(N))
        random.shuffle(order)
        
        for u in order:
            if time.time() > deadline - 5:
                break
            if u not in remaining:
                continue
            
            selected.add(u)
            remaining.remove(u)
            
            for v in adj_list[u]:
                if v in remaining:
                    remaining.remove(v)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
        # Local search
        if time.time() < deadline - 10:
            ls_set = local_search(selected, adj_list, N, deadline - 5)
            if len(ls_set) > best_size:
                best_size = len(ls_set)
                best_set = ls_set
                
    # Construct final vectors
    final_vectors = []
    for vid in best_set:
        final_vectors.append(vectors[vid])
        
    if not final_vectors:
        # Fallback: return a small valid set
        # e.g. basis vectors? No, norm 1. Dist 1. OK.
        # But we want high score.
        # Just return one vector
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    return np.array(final_vectors, dtype=np.int64)
