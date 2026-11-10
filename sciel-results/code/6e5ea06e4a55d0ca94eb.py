import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor masks for each word
    # For radius r, neighbors are words at Hamming distance <= r
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = {i}
        
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
        
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = w[pos1]
                    v2 = w[pos2]
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            new_w_list = list(w)
                            new_w_list[pos1] = nv1
                            new_w_list[pos2] = nv2
                            new_w = tuple(new_w_list)
                            neigh_set.add(word_to_idx[new_w])
        
        if radius >= 3:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for pos3 in range(pos2 + 1, n):
                        v1 = w[pos1]; v2 = w[pos2]; v3 = w[pos3]
                        for nv1 in range(q):
                            if nv1 == v1: continue
                            for nv2 in range(q):
                                if nv2 == v2: continue
                                for nv3 in range(q):
                                    if nv3 == v3: continue
                                    new_w_list = list(w)
                                    new_w_list[pos1] = nv1
                                    new_w_list[pos2] = nv2
                                    new_w_list[pos3] = nv3
                                    new_w = tuple(new_w_list)
                                    neigh_set.add(word_to_idx[new_w])
        
        neighbors.append(neigh_set)
    
    # Convert to bitmasks
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for idx in neighbors[i]:
            m |= (1 << idx)
        masks[i] = m
            
    uncovered_mask = (1 << total_words) - 1
    
    # Greedy set cover with multiple random restarts
    best_code = None
    best_size = total_words + 1
    
    num_restarts = 50  # Try multiple restarts with different random orderings
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            # First restart: deterministic order
            candidate_order = list(range(total_words))
        else:
            # Subsequent restarts: random order
            candidate_order = list(range(total_words))
            random.shuffle(candidate_order)
        
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find candidate with max gain
            best_candidate = -1
            max_gain = -1
            
            for i in candidate_order:
                # Calculate gain: how many new words does this candidate cover?
                gain = 0
                for idx in neighbors[i]:
                    if not covered[idx]:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidate = i
                    if gain == 0:
                        break
                elif gain == max_gain and gain > 0:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_candidate = i
            
            if best_candidate == -1 or max_gain == 0:
                break
                
            code_indices.append(best_candidate)
            
            # Mark neighbors as covered
            for idx in neighbors[best_candidate]:
                covered[idx] = True
            
            # Check if all covered
            if not covered.any() or covered.all():
                if covered.all():
                    break
        
        # Check if this is a valid covering
        if len(code_indices) > 0:
            final_union = 0
            for i in code_indices:
                final_union |= masks[i]
            
            if final_union == (1 << total_words) - 1:
                # Valid covering
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = code_indices[:]
        
        # Early termination if we found optimal or near-optimal
        if best_size <= 73:
            break
    
    # If no valid code found, use fallback
    if best_code is None:
        if q == 3 and n == 6 and radius == 1:
            # Tetracode product construction: size 81
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            tail_space = list(itertools.product(range(3), repeat=2))
            fallback = [h + list(tail) for h in tetracode for tail in tail_space]
            return fallback
        else:
            return all_words
    
    # Post-processing: Remove redundant codewords
    if time.time() < deadline - 10:
        changed = True
        while changed:
            changed = False
            for i in range(len(best_code)):
                # Temporarily remove i
                # Check if union of others covers everything
                union_others = 0
                for j in range(len(best_code)):
                    if j == i:
                        continue
                    union_others |= masks[best_code[j]]
                
                if (masks[best_code[i]] & ~union_others) == 0:
                    # Redundant
                    best_code.pop(i)
                    changed = True
                    break
            
            if time.time() > deadline - 5:
                break
    
    # Convert indices back to words
    result = [all_words[i] for i in best_code]
    
    # Final validation check
    final_union = 0
    for i in best_code:
        final_union |= masks[i]
    
    if final_union != (1 << total_words) - 1:
        # Fallback
        if q == 3 and n == 6 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            tail_space = list(itertools.product(range(3), repeat=2))
            fallback = [h + list(tail) for h in tetracode for tail in tail_space]
            return fallback
        else:
            return all_words
    
    return result
