import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118  # 2s margin for 120s limit
    
    total_words = q ** n
    
    # Precompute neighbors as bitsets
    # Word i is represented by its index in base-q expansion
    # neighbor_mask[i] is a Python int where bit j is 1 if word j is within distance 1 of word i
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    neighbor_masks = [0] * total_words
    # Also store the list of neighbors for potential alternative implementations if needed, 
    # but bitsets are faster for union/intersection.
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 1 << i
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
        neighbor_masks[i] = mask

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            # Find word with max gain
            best_gain = 0
            best_candidates = []
            
            # Optimization: If number of uncovered words is small, we can be smarter
            # But for now, simple max gain
            for i in range(total_words):
                gain = bin(uncovered & neighbor_masks[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            chosen = best_candidates[random.randint(0, len(best_candidates) - 1)]
            code.append(chosen)
            uncovered &= ~neighbor_masks[chosen]
            
        return code

    def reduce_code(code):
        code = list(code)
        if not code:
            return code
            
        # Calculate coverage mask for all words
        # We will iteratively remove redundant words
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant
                # Union of neighbors of all other words
                others_mask = 0
                for x in code:
                    if x != c:
                        others_mask |= neighbor_masks[x]
                
                if (neighbor_masks[c] & ~others_mask) == 0:
                    # c is redundant
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to randomize order
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Calculate mask covered by all other words
                others_mask = 0
                for x in code:
                    if x != c_remove:
                        others_mask |= neighbor_masks[x]
                
                # Check if c_remove is redundant
                if (neighbor_masks[c_remove] & ~others_mask) == 0:
                    code.pop(i)
                    improved = True
                    break
                
                # Check for swap
                holes = neighbor_masks[c_remove] & ~others_mask
                if holes:
                    # Find a word d not in code that covers all holes
                    # d must be in the intersection of reverse_cover[h] for h in holes
                    # But we don't have reverse_cover precomputed as sets for bitsets easily.
                    # Let's iterate through all possible d not in code.
                    # To speed up, we can check candidates that intersect holes.
                    
                    # Optimization: Only consider d that covers at least one hole
                    # Actually, d MUST cover ALL holes.
                    # Let's find candidates from the first hole
                    # We need a list of words covering each word.
                    # Precomputing reverse_cover lists is useful here.
                    
                    pass # Placeholder, handled below with precomputed lists
                    
        return code

    # Precompute reverse cover: for each word w, list of codewords that cover w
    # This is expensive to store as lists of ints, but total_words is 2187.
    # reverse_cover[j] = [i for i in range(total_words) if bit j is set in neighbor_masks[i]]
    # This is 2187 lists.
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        mask = neighbor_masks[i]
        # Iterate set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            j = lsb.bit_length() - 1
            reverse_cover[j].append(i)
            m ^= lsb

    def local_search_optimized(code, deadline):
        code = list(code)
        if not code:
            return code
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Calculate mask covered by all other words
                others_mask = 0
                for x in code:
                    if x != c_remove:
                        others_mask |= neighbor_masks[x]
                
                # Check if c_remove is redundant
                if (neighbor_masks[c_remove] & ~others_mask) == 0:
                    code.pop(i)
                    improved = True
                    break
                
                # Check for swap
                holes = neighbor_masks[c_remove] & ~others_mask
                if holes:
                    # Find candidates that cover all holes
                    # Start with candidates covering the first hole
                    # Get index of first set bit in holes
                    lsb = holes & -holes
                    h_idx = lsb.bit_length() - 1
                    candidates = set(reverse_cover[h_idx])
                    candidates.discard(c_remove)
                    
                    # Intersect with reverse cover of other holes
                    m = holes
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if j == h_idx:
                            pass
                        else:
                            candidates &= set(reverse_cover[j])
                            if not candidates:
                                break
                        m ^= lsb
                    
                    if candidates:
                        # Choose best candidate
                        # Heuristic: Minimize "waste" or maximize "synergy"
                        # Let's just pick one randomly or the one with highest gain in other areas?
                        # For now, pick one that allows the swap.
                        # To be deterministic and potentially better, pick the one that covers the most "critical" words?
                        # Critical word: covered by only 1 codeword in the current set (excluding c_remove? No, including).
                        # Let's just pick a random one to explore space.
                        cand_list = list(candidates)
                        best_cand = cand_list[random.randint(0, len(cand_list) - 1)]
                        
                        # Perform swap
                        code[i] = best_cand
                        improved = True
                        break
                        
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        # Remove k random words
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
            
        # Repair greedily
        covered_mask = 0
        for c in code:
            covered_mask |= neighbor_masks[c]
            
        uncovered = (1 << total_words) - 1
        uncovered &= ~covered_mask
        
        while uncovered and time.time() < deadline:
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                gain = bin(uncovered & neighbor_masks[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            chosen = best_candidates[random.randint(0, len(best_candidates) - 1)]
            code.append(chosen)
            uncovered &= ~neighbor_masks[chosen]
            
        return code

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # 1. Greedy
        code = greedy_construct(seed)
        
        # 2. Reduce
        code = reduce_code(code)
        
        # 3. Local Search
        if time.time() < deadline - 5:
            code = local_search_optimized(code, deadline - 2)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # 4. Perturb and Repair
        if time.time() < deadline - 10 and best_code:
            # Perturb best code
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 5)
            
            # Reduce and Local Search perturbed
            perturbed = reduce_code(perturbed)
            if time.time() < deadline - 5:
                perturbed = local_search_optimized(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
                
        seed += 1
        
    if best_code is None:
        best_code = greedy_construct(0)
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
