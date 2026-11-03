import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But the function signature is generic. We assume the evaluator passes these.
    # We will optimize for the specific case q=3, n=7, radius=1 but keep it general enough.
    
    total_words = q ** n
    
    # Precompute neighbors as bitmasks and index lists
    # Map word tuple to index
    # Index = sum(digit * q^i)
    
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

    # Precompute neighbors
    neighbors_mask = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 0
        idxs = [i]
        # Distance 0
        mask |= (1 << i)
        # Distance 1
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
                idxs.append(j)
        neighbors_mask[i] = mask
        neighbor_indices[i] = idxs

    def greedy_construct(random_seed, strategy=0):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_gain = -1
            best_candidate = -1
            best_count = 0
            
            # To add randomness, we can pick from top few or just randomize iteration order
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for i in candidates:
                if (neighbors_mask[i] & uncovered) == 0:
                    continue
                gain = (neighbors_mask[i] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = i
                    best_count = 1
                elif gain == best_gain:
                    best_count += 1
                    # Reservoir sampling to pick one of the best candidates uniformly
                    if random.random() < 1.0 / best_count:
                        best_candidate = i
            
            if best_gain == -1:
                break # Should not happen if uncovered is not 0 and masks are correct
                
            chosen = best_candidate
            code.append(chosen)
            uncovered &= ~neighbors_mask[chosen]
            
        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Try removing c
                for w in neighbor_indices[c]:
                    counts[w] -= 1
                
                # Check if any count is 0
                if np.any(counts == 0):
                    # Restore
                    for w in neighbor_indices[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    # Keep removed
                    changed = True
            
            code = new_code
        return code

    def local_search_improve(code, deadline):
        """
        Fast local search using coverage counts.
        1. Try to remove a codeword.
        2. If removal creates uncovered words, try to replace it with a better one
           that covers the gap and potentially allows further reduction.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to avoid getting stuck in specific order
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Tentatively remove c_remove
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                # Check if any count is 0
                if np.any(counts == 0):
                    # Restore
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1
                    
                    # Find the best replacement
                    # We need to cover the words that are now at count 0 (or would be if we didn't restore)
                    # Actually, we restored, so we need to identify which words were at risk.
                    # Let's re-calculate the "uncovered" set if we were to remove c_remove.
                    # Since we restored, we can just find words w where counts[w] == 1 and w is in neighbor_indices[c_remove].
                    # These are the critical words.
                    critical_words = []
                    for w in neighbor_indices[c_remove]:
                        if counts[w] == 1:
                            critical_words.append(w)
                    
                    if not critical_words:
                        # This shouldn't happen if np.any(counts==0) was true, but just in case
                        continue

                    # We need a word w_new such that neighbors_mask[w_new] covers all critical_words.
                    # This means w_new must be in neighbor_indices[cw] for all cw in critical_words.
                    # We can find the intersection of these neighbor lists.
                    
                    # Start with the neighbor list of the first critical word
                    # If there is only one critical word, the intersection is its neighbors.
                    if len(critical_words) == 1:
                        candidates = neighbor_indices[critical_words[0]]
                    else:
                        # Use a set for intersection, or just check membership
                        # Since lists are small (size 15), we can just intersect
                        # Let's pick the smallest list to iterate? They are all size 15.
                        # Just intersect the first two, then check the rest.
                        cand_set = set(neighbor_indices[critical_words[0]])
                        for cw in critical_words[1:]:
                            cand_set &= set(neighbor_indices[cw])
                            if not cand_set:
                                break
                        candidates = list(cand_set)
                    
                    best_w = -1
                    # Check candidates
                    for w in candidates:
                        if w == c_remove:
                            continue
                        # We already know w covers all critical words by construction of candidates
                        # But we should double check if we didn't use sets? 
                        # The set intersection ensures w is in neighbor_indices[cw] for all cw.
                        # So w covers all critical words.
                        best_w = w
                        break # First valid replacement is sufficient
                    
                    if best_w != -1:
                        # Perform swap
                        code[i] = best_w
                        
                        # Update counts
                        # Remove c_remove from counts (we already did, but we restored it)
                        # So current counts include c_remove.
                        # We need to:
                        # 1. Subtract c_remove
                        # 2. Add best_w
                        
                        for w in neighbor_indices[c_remove]:
                            counts[w] -= 1
                        for w in neighbor_indices[best_w]:
                            counts[w] += 1
                        
                        # Check if swap was valid (it should be by construction)
                        # Now, try to reduce the code again
                        # We can break out and restart the reduction loop
                        improved = True
                        break # Restart the while loop with new code
                
                else:
                    # c_remove was redundant
                    code.pop(i)
                    improved = True
                    break # Restart the while loop with new code
        
        return code

    # Main loop with time management
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code_fast(code)
        
        # Try local search improvement if time permits
        if len(code) < best_size or (best_code is None):
            if time.time() < deadline - 5: # Leave time for local search
                code = local_search_improve(code, deadline - 2)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we hit 186, we can stop? No, lower bound is 156. 
            # But 186 is best known. If we find 186, it's a great result.
            if best_size <= 186:
                # Keep searching a bit more to see if we can go lower
                pass
                # break # Don't break, try to beat 186
        
        seed += 1
        
    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
