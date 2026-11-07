"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    n = 256
    num_v = n - 1  # 255 vertices, indexed 0..254, vertex i = number i+2
    weights = [i + 2 for i in range(num_v)]
    
    # Precompute coprime_mask[i]: bit j set if i and j are coprime
    # Note: coprime_mask[i] does NOT include bit i
    coprime_mask = [0] * num_v
    for i in range(num_v):
        val_i = i + 2
        mask = 0
        for j in range(num_v):
            if i == j:
                continue
            val_j = j + 2
            if gcd(val_i, val_j) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask
        
    # Precompute non_coprime_mask[i]: bit j set if i and j are NOT coprime (i != j)
    # This is the conflict graph adjacency
    non_coprime_mask = [0] * num_v
    for i in range(num_v):
        # All vertices except i
        all_except_i = ((1 << num_v) - 1) ^ (1 << i)
        # Non-coprime are those in all_except_i but NOT in coprime_mask[i]
        non_coprime_mask[i] = all_except_i & ~coprime_mask[i]

    best_weight = [0]
    best_set = [set()]
    
    def greedy_clique_cover_bound(candidates_mask):
        """
        Upper bound for MWIS on candidates_mask.
        Equivalent to coloring the complement graph.
        Partitions candidates into cliques of the conflict graph (sets of pairwise non-coprime numbers).
        Sum of max weight in each clique is the bound.
        """
        remaining = candidates_mask
        bound = 0
        while remaining:
            # Form one clique (color class in complement)
            avail = remaining
            max_w_in_clique = 0
            
            while avail:
                # Pick vertex with highest weight from avail
                temp = avail
                best_idx = -1
                best_w = -1
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    w = weights[idx]
                    if w > best_w:
                        best_w = w
                        best_idx = idx
                    temp &= temp - 1
                
                if best_idx < 0:
                    break
                    
                max_w_in_clique = best_w
                
                # Remove best_idx and all its coprime neighbors from avail
                # Coprime neighbors are those NOT in the clique (since clique requires non-coprime)
                # So we keep only best_idx and its non-coprime neighbors in avail for this clique?
                # No, we are forming the clique. 
                # If u is coprime to best_idx, u cannot be in this clique.
                # So we remove u from avail.
                # Vertices to remove: best_idx AND all u such that gcd(u, best_idx) == 1.
                # These are exactly the bits in coprime_mask[best_idx].
                # Also remove best_idx itself.
                remove_mask = (1 << best_idx) | coprime_mask[best_idx]
                avail &= ~remove_mask
                
            bound += max_w_in_clique
            remaining = avail
            
        return bound

    def branch_and_bound(candidates_mask, current_weight, current_set):
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set[0] = set(current_set)
            return
        
        # Upper bound
        ub = current_weight + greedy_clique_cover_bound(candidates_mask)
        if ub <= best_weight[0]:
            return
        
        # Pick vertex to branch on
        # Heuristic: Highest weight
        temp = candidates_mask
        best_idx = -1
        best_w = -1
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            w = weights[idx]
            if w > best_w:
                best_w = w
                best_idx = idx
            temp &= temp - 1
        
        if best_idx < 0:
            return
        
        # Branch 1: Include best_idx
        # New candidates: those coprime to best_idx (and still in candidates)
        # Note: best_idx is removed because coprime_mask[best_idx] doesn't have bit best_idx
        new_candidates = candidates_mask & coprime_mask[best_idx]
        branch_and_bound(new_candidates, current_weight + best_w, current_set | {best_idx})
        
        # Branch 2: Exclude best_idx
        new_candidates = candidates_mask & ~(1 << best_idx)
        branch_and_bound(new_candidates, current_weight, current_set)
        
    # Start with all vertices
    all_mask = (1 << num_v) - 1
    branch_and_bound(all_mask, 0, set())
    
    # Always include 1
    result = [1]
    for idx in best_set[0]:
        result.append(idx + 2)
        
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
