import time
import random
import math
from itertools import combinations

def _is_regular(mask, adj_masks):
    """Check if induced subgraph defined by mask is regular."""
    if mask == 0:
        return False
    # Get vertices
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m -= lsb
    
    if not verts:
        return False
        
    # Check degrees
    first_deg = (adj_masks[verts[0]] & mask).bit_count()
    for v in verts[1:]:
        deg = (adj_masks[v] & mask).bit_count()
        if deg != first_deg:
            return False
    return True

def _count_regular_subgraphs(adj_masks, n, sizes):
    """Count regular induced subgraphs for given sizes."""
    count = 0
    for k in sizes:
        # Iterate all subsets of size k
        # For efficiency, we can use combinations
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular(mask, adj_masks):
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_adj_masks = []
    
    # Precomputation for incremental updates is complex to set up dynamically for each n.
    # Given the time limit, a simpler but effective approach is:
    # 1. Try to find a valid graph for n=22, 21, 20 using Simulated Annealing on size-6 violations.
    # 2. Verification is done only when size-6 count is 0.
    
    for target_n in [22, 21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        found = False
        
        # Precompute subset lists for incremental check?
        # For n=22, subsets of size 6 containing (u,v) is 4845.
        # We can generate them on the fly if it's fast enough, or precompute.
        # Precomputing for all pairs is better.
        
        # Let's try a simpler heuristic first: Random restarts with local search on size-6.
        # If local search fails, try SA.
        
        # We will use a cost function: number of regular subgraphs of size 6.
        
        # To speed up, we precompute the list of masks for each pair (u,v) for size 6.
        # masks_6[u][v] = list of masks (int) of size 6 containing u and v.
        # This allows O(1) lookup of affected subsets.
        
        masks_6 = [[None] * n for _ in range(n)]
        
        # Precomputation time:
        # For each pair (u,v), generate C(n-2, 4) masks.
        # For n=22, C(20,4) = 4845.
        # 231 pairs * 4845 = ~1.1M masks.
        # This takes a few seconds in Python.
        
        if time.time() > deadline - 5:
            continue
            
        for u in range(n):
            for v in range(u + 1, n):
                other_verts = [x for x in range(n) if x != u and x != v]
                masks_6[u][v] = []
                masks_6[v][u] = masks_6[u][v] # Symmetric list, but we need to be careful with reference?
                # Actually, we just need the list of masks.
                # Generating combinations of 4 from 20.
                for comb in combinations(other_verts, 4):
                    mask = (1 << u) | (1 << v)
                    for x in comb:
                        mask |= (1 << x)
                    masks_6[u][v].append(mask)
                # Note: masks_6[v][u] should be the same list of masks.
                # We can just set it to the same object or copy.
                # Since we only read, sharing is fine.
                masks_6[v][u] = masks_6[u][v]
                
        # Now run SA
        # Initial random graph
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    
        # Calculate initial cost
        current_cost = 0
        # We can calculate by summing over all pairs? No, that double counts.
        # Just iterate all subsets of size 6.
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular(mask, adj_masks):
                current_cost += 1
                
        T = 10.0
        min_T = 0.1
        alpha = 0.999
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Calculate delta cost
            # Flip edge (u,v)
            bit_v = 1 << v
            bit_u = 1 << u
            
            # Temporarily flip
            adj_masks[u] ^= bit_v
            adj_masks[v] ^= bit_u
            
            # Check affected subsets (size 6 containing u,v)
            delta = 0
            for mask in masks_6[u][v]:
                # We need to know if it was regular BEFORE and AFTER.
                # But we already flipped.
                # So "After" is current state.
                # "Before" is state with edge flipped back.
                # This is inefficient.
                
                # Better:
                # 1. Check if mask is regular in CURRENT (flipped) state.
                # 2. Check if mask is regular in ORIGINAL state.
                
                # Let's do it properly:
                # Save original state of u, v
                orig_u = adj_masks[u]
                orig_v = adj_masks[v]
                
                # Current state is flipped.
                # Let's check regularity in current state.
                is_reg_after = _is_regular(mask, adj_masks)
                
                # Check regularity in original state (temporarily revert)
                adj_masks[u] = orig_u
                adj_masks[v] = orig_v
                is_reg_before = _is_regular(mask, adj_masks)
                
                # Restore flipped state
                adj_masks[u] = orig_u ^ bit_v
                adj_masks[v] = orig_v ^ bit_u
                
                if is_reg_after and not is_reg_before:
                    delta += 1
                elif not is_reg_after and is_reg_before:
                    delta -= 1
            
            # Decide acceptance
            if delta <= 0:
                current_cost += delta
            else:
                if random.random() < math.exp(-delta / T):
                    current_cost += delta
                else:
                    # Revert
                    adj_masks[u] ^= bit_v
                    adj_masks[v] ^= bit_u
            
            T *= alpha
            if T < min_T:
                T = min_T
                
            if current_cost == 0:
                # Potential solution!
                # Verify full condition (all sizes >= 6)
                valid = True
                for k in range(6, n + 1):
                    for comb in combinations(range(n), k):
                        mask = 0
                        for v in comb:
                            mask |= (1 << v)
                        if _is_regular(mask, adj_masks):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_n = n
                    best_adj_masks = adj_masks
                    found = True
                    break
        
        if found:
            break
            
    if best_n == 0:
        # Fallback: return a small valid graph
        # n=5 is always valid (no subset of size 6)
        best_n = 5
        best_adj_masks = [0] * 5
        # Add some edges to make it non-trivial? Doesn't matter for validity.
        
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_masks[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
