import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses bitmask-accelerated local search with multiple random starts.
    """
    m = 205
    deadline = time.time() + 580.0  # Safety margin for 600s limit
    
    # 1. Compute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitmasks
    # conflict_mask[i] is a bitmask where bit j is 1 if (i-j) is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Precompute compatible_mask[i] = ~conflict_mask[i] & ((1 << m) - 1) & ~(1 << i)
    full_mask = (1 << m) - 1
    compatible_mask = [0] * m
    for i in range(m):
        compatible_mask[i] = full_mask ^ conflict_mask[i] ^ (1 << i)

    def get_conflicts(selected_mask, candidate):
        """
        Returns True if candidate conflicts with any node in selected_mask.
        """
        return (selected_mask & conflict_mask[candidate]) != 0

    def greedy_random(seed):
        """
        Generate a random greedy solution.
        """
        random.seed(seed)
        selected_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for c in candidates:
            if not get_conflicts(selected_mask, c):
                selected_mask |= (1 << c)
                
        return selected_mask

    def local_search(start_mask):
        """
        Improve a solution using 1-add, 2-swap, 3-swap, and 1-swap moves.
        """
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        no_improvement_count = 0
        max_iterations = 5000
        shake_count = 0
        rng = random.Random()
        
        while no_improvement_count < 100 and max_iterations > 0 and time.time() < deadline:
            max_iterations -= 1
            improved = False
            
            # 1. Try 1-Add: Add a node that doesn't conflict
            # Find any node not in current_mask that is compatible with all in current_mask
            # compatible_with_all = full_mask
            # for i in indices(current_mask): compatible_with_all &= compatible_mask[i]
            # candidates = compatible_with_all & ~current_mask
            # if candidates: pick one
            
            # Optimization: just iterate and check
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if not get_conflicts(current_mask, c):
                        current_mask |= (1 << c)
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
            
            if improved:
                no_improvement_count = 0
                continue
            
            # 2. Try 2-Swap: Remove one node i, add two nodes j, k
            current_indices = [i for i in range(m) if current_mask & (1 << i)]
            
            found_2swap = False
            for i in current_indices:
                mask_without_i = current_mask & ~(1 << i)
                
                # Find candidates compatible with mask_without_i
                # candidate c is valid if (mask_without_i & conflict_mask[c]) == 0
                # Equivalent to: c is in compatible_mask[k] for all k in mask_without_i
                # Let's compute the intersection of compatible_mask for all k in mask_without_i
                # Then intersect with ~current_mask
                
                # Start with all nodes
                valid_cands = full_mask
                for k in current_indices:
                    if k == i:
                        continue
                    valid_cands &= compatible_mask[k]
                
                # Remove nodes already in current_mask
                valid_cands &= ~current_mask
                
                # Now valid_cands contains nodes that can be added if i is removed.
                # We need to find two such nodes j, k that are compatible with each other.
                # i.e., an edge in the compatibility graph induced by valid_cands.
                
                # Get list of candidates
                cand_list = [c for c in range(m) if valid_cands & (1 << c)]
                
                if len(cand_list) < 2:
                    continue
                    
                # Check for compatible pair in cand_list
                # If cand_list is small, O(L^2) is fine.
                # If large, we can break early.
                pair_found = False
                for u_idx in range(len(cand_list)):
                    j = cand_list[u_idx]
                    # Check if j is compatible with any other in cand_list
                    # compatible_mask[j] has bits for all compatible nodes.
                    # Intersect with valid_cands (excluding j itself? compatible_mask[j] doesn't have j)
                    # So if (compatible_mask[j] & valid_cands) != 0, there is a compatible node.
                    if (compatible_mask[j] & valid_cands):
                        # Find k
                        # We just need one k.
                        # The first bit set in (compatible_mask[j] & valid_cands)
                        k_mask = compatible_mask[j] & valid_cands
                        # Get index of first set bit
                        # k = (k_mask & -k_mask).bit_length() - 1
                        k = (k_mask & -k_mask).bit_length() - 1
                        
                        # Perform swap
                        new_mask = mask_without_i | (1 << j) | (1 << k)
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        found_2swap = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        pair_found = True
                        break
                if pair_found:
                    break
            
            if found_2swap:
                no_improvement_count = 0
                continue

            # 3. Try 3-Swap: Remove two nodes i, l, add three nodes j, k, v
            # This is more expensive. Only try if 2-swap fails.
            # Iterate pairs (i, l) in current_indices
            found_3swap = False
            if len(current_indices) >= 2:
                for u_idx in range(len(current_indices)):
                    for v_idx in range(u_idx + 1, len(current_indices)):
                        i = current_indices[u_idx]
                        l = current_indices[v_idx]
                        mask_without_il = current_mask & ~(1 << i) & ~(1 << l)
                        
                        # Find candidates compatible with mask_without_il
                        valid_cands = full_mask
                        for k in current_indices:
                            if k == i or k == l:
                                continue
                            valid_cands &= compatible_mask[k]
                        
                        valid_cands &= ~current_mask
                        
                        cand_list = [c for c in range(m) if valid_cands & (1 << c)]
                        
                        if len(cand_list) < 3:
                            continue
                        
                        # Need to find 3 compatible nodes in cand_list.
                        # This is finding a triangle in the compatibility graph induced by cand_list.
                        # Or just 3 nodes that are pairwise compatible.
                        # Greedy: pick j from cand_list.
                        # Pick k from cand_list compatible with j.
                        # Pick v from cand_list compatible with j and k.
                        
                        # To avoid O(L^3), we can do:
                        # For each j in cand_list:
                        #   candidates_k = (compatible_mask[j] & valid_cands)
                        #   For each k in candidates_k:
                        #      candidates_v = (compatible_mask[j] & compatible_mask[k] & valid_cands)
                        #      if candidates_v:
                        #          v = first bit
                        #          success
                        
                        # Limit search depth to avoid slowness
                        for j in cand_list:
                            cands_k = compatible_mask[j] & valid_cands
                            # Iterate bits in cands_k
                            # If cands_k is 0, continue
                            # To speed up, check if cands_k has at least 2 bits? No, k and v must be distinct.
                            # Just iterate.
                            
                            # Optimization: If cands_k has many bits, we can just pick one and check if it has a neighbor.
                            # But we need to be sure.
                            
                            # Let's just try to find one.
                            # If cands_k is not 0:
                            #   Pick first k.
                            #   Check if (compatible_mask[j] & compatible_mask[k] & valid_cands) is not 0.
                            #   If yes, pick v.
                            #   This might fail if the first k doesn't have a neighbor, but another k does.
                            #   So we might need to iterate k.
                            
                            # Given small size, iterate k.
                            for k in range(m):
                                if not (cands_k & (1 << k)):
                                    continue
                                cands_v = compatible_mask[j] & compatible_mask[k] & valid_cands
                                if cands_v:
                                    v = (cands_v & -cands_v).bit_length() - 1
                                    # Found j, k, v
                                    new_mask = mask_without_il | (1 << j) | (1 << k) | (1 << v)
                                    current_mask = new_mask
                                    current_size += 1
                                    improved = True
                                    found_3swap = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_mask = current_mask
                                    break
                            if found_3swap:
                                break
                        if found_3swap:
                            break
                if found_3swap:
                    no_improvement_count = 0
                    continue

            # 4. Try 1-Swap: Remove one node i, add one node j (size stays same)
            # This helps escape local optima.
            if current_indices:
                i = rng.choice(current_indices)
                mask_without_i = current_mask & ~(1 << i)
                
                # Find a random j not in current_mask that is compatible with mask_without_i
                valid_js = []
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    if not (mask_without_i & conflict_mask[j]):
                        valid_js.append(j)
                
                if valid_js:
                    j = rng.choice(valid_js)
                    current_mask = mask_without_i | (1 << j)
                    # Size is same
                    no_improvement_count = 0
                    continue
            
            # If no moves found, we are stuck.
            no_improvement_count += 1
            
            # If stuck for a long time, perform a random restart from a perturbed version
            if no_improvement_count > 50:
                # Perturb: remove 1-3 random nodes
                num_remove = rng.randint(1, 3)
                if current_size > num_remove:
                    indices_to_remove = rng.sample([i for i in range(m) if current_mask & (1 << i)], num_remove)
                    for idx in indices_to_remove:
                        current_mask &= ~(1 << idx)
                    current_size = bin(current_mask).count('1')
                    no_improvement_count = 0 # Reset to allow more search
                    shake_count += 1
                    if shake_count > 10:
                        break # Give up on this start
                
        return best_mask, best_size

    # Main Search Loop
    best_global_mask = 0
    best_global_size = 0
    
    # Start with a few deterministic/random seeds
    num_starts = 100 # Increased starts
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Use different seeds
        seed = i * 12345 + 67890
        init_mask = greedy_random(seed)
        
        # Local search
        improved_mask, improved_size = local_search(init_mask)
        
        if improved_size > best_global_size:
            best_global_size = improved_size
            best_global_mask = improved_mask
            
        # Early termination if we reach a theoretical upper bound?
        # Max independent set size is not known exactly, but if we get > 15, it's likely near optimal.
        # Let's just run until deadline.

    # Convert best mask to array
    result = []
    for i in range(m):
        if best_global_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
