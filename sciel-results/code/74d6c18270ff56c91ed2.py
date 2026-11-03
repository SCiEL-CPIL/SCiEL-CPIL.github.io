import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute neighbors (Hamming distance <= radius)
    # neighbors[i] is a list of indices of words within distance <= radius of word i
    # reverse_cover[j] is a list of indices of codewords that cover word j
    neighbors = [[] for _ in range(total_space)]
    reverse_cover = [[] for _ in range(total_space)]
    
    # For radius 1, neighbors are self + all single-bit changes
    for i in range(total_space):
        w = all_words[i]
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    neighbors[i].append(j)
                    reverse_cover[j].append(i)

    full_mask = (1 << total_space) - 1
    # Bitmasks for coverage for fast intersection
    cover_masks = [0] * total_space
    for i in range(total_space):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        cover_masks[i] = mask

    def greedy_cover(randomized=False, alpha=1.0):
        """
        Greedy set cover with optional randomization.
        alpha controls the randomness: higher alpha means more random choice among top candidates.
        """
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            # To speed up, we could maintain a priority queue, but for N=2187, linear scan is okay
            # However, doing this repeatedly is slow. 
            # Optimization: Only consider codewords that have high potential.
            # For now, simple linear scan with early exit if gain is 0.
            
            # Randomization strategy: 
            # 1. Find max gain.
            # 2. Collect all with gain == max_gain.
            # 3. If randomized, pick randomly from top 'k' or use softmax.
            
            max_gain = 0
            top_candidates = []
            
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    top_candidates = [i]
                elif gain == max_gain and gain > 0:
                    top_candidates.append(i)
            
            if max_gain == 0:
                break # Should not happen if space is connected, but safety
                
            if randomized:
                # Pick from top candidates, weighted by gain? 
                # Simple: just pick random from top candidates.
                # To make it more "simulated annealing" like, we could pick from candidates with gain > 0.5 * max_gain
                threshold = max_gain * 0.8 # Consider candidates with at least 80% of max gain
                viable = [c for c in top_candidates if (cover_masks[c] & uncovered).bit_count() >= threshold]
                if not viable:
                    viable = top_candidates
                best_idx = random.choice(viable)
            else:
                best_idx = top_candidates[0]
                
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    def local_search_reduce(code):
        """
        Local search to minimize code size.
        1. Remove redundant codewords.
        2. Swap codewords if it reduces size or maintains size with better structure.
        """
        if not code:
            return code
            
        current_code = set(code)
        
        # Build cover_count: cover_count[w] = number of codewords covering w
        cover_count = [0] * total_space
        for idx in current_code:
            for w in neighbors[idx]:
                cover_count[w] += 1
                
        improved = True
        max_iterations = 1000
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            # 1. Try to remove redundant codewords
            # A codeword c is redundant if all words it covers are covered by at least one other codeword.
            # i.e., for all w in neighbors[c], cover_count[w] > 1.
            
            # We iterate over a copy of the list to allow deletion
            code_list = list(current_code)
            random.shuffle(code_list) # Randomize order to break symmetry
            
            removed_any = False
            for idx in code_list:
                if idx not in current_code:
                    continue
                    
                removable = True
                for w_idx in neighbors[idx]:
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                
                if removable:
                    # Decrement cover_count
                    for w_idx in neighbors[idx]:
                        cover_count[w_idx] -= 1
                    current_code.remove(idx)
                    improved = True
                    removed_any = True
                    break # Restart loop with updated counts
            
            if removed_any:
                continue
                
            # 2. Try to swap codewords
            # If we cannot remove any, try to find a swap that reduces the number of "private" words
            # or simply finds a valid swap if it leads to a smaller set (which is impossible in a swap 1-for-1)
            # Actually, a swap 1-for-1 keeps the size same. 
            # To reduce size, we need to remove 1 and add 0 (impossible if not redundant) 
            # OR remove 1 and add 1, but then later remove another? 
            # Standard approach: 
            # Find a codeword c with private words. 
            # Find a candidate c' that covers all private words of c.
            # If we swap c with c', size remains same. 
            # However, this might make c' redundant later, or make other codewords redundant.
            
            # Better strategy for size reduction:
            # If no removal is possible, we are in a local minimum for size.
            # We need to perturb.
            
            # For now, just try to find a swap that might help future removals.
            # Identify codewords with private words.
            swaps_found = False
            for idx in code_list:
                if idx not in current_code:
                    continue
                
                # Find private words of idx
                private_words = []
                for w_idx in neighbors[idx]:
                    if cover_count[w_idx] == 1:
                        private_words.append(w_idx)
                        
                if not private_words:
                    # This shouldn't happen if we just checked for removal, 
                    # but due to shuffle/order, maybe? No, if cover_count[w] > 1 for all w, it was removable.
                    # So if we are here, there is at least one private word.
                    pass
                
                if private_words:
                    # Find candidates that cover ALL private words
                    # Start with reverse_cover of first private word
                    candidates = set(reverse_cover[private_words[0]])
                    for w in private_words[1:]:
                        candidates &= set(reverse_cover[w])
                        
                    # Exclude current code
                    candidates -= current_code
                    
                    if candidates:
                        # Pick a random candidate
                        new_idx = random.choice(list(candidates))
                        
                        # Check if this swap is "good". 
                        # Ideally, new_idx should cover words that are heavily covered by others, 
                        # or simply just do it to break symmetry.
                        
                        # Perform swap
                        # Remove idx
                        for w_idx in neighbors[idx]:
                            cover_count[w_idx] -= 1
                        current_code.remove(idx)
                        
                        # Add new_idx
                        for w_idx in neighbors[new_idx]:
                            cover_count[w_idx] += 1
                        current_code.add(new_idx)
                        
                        improved = True
                        swaps_found = True
                        break # Restart loop with updated counts
            
            if not improved:
                # No removal, no swap found? 
                # This means for every codeword, there is no single codeword outside the set 
                # that covers all its private words.
                # We are stuck. Return current best.
                break
                
        return list(current_code)

    def iterated_local_search(initial_code):
        current_code = set(local_search_reduce(initial_code))
        best_code = list(current_code)
        best_size = len(best_code)
        
        # Simulated Annealing / Iterated Local Search
        temperature = 10.0
        min_temperature = 1.0
        cooling = 0.95
        
        while time.time() < deadline - 5.0:
            # Perturbation
            # Remove k random codewords
            k = random.randint(1, 3)
            if len(current_code) <= k:
                k = len(current_code) // 2
            if k == 0:
                break
                
            to_remove = random.sample(list(current_code), k)
            perturbed = list(current_code - set(to_remove))
            
            # Repair
            # Compute uncovered
            covered_mask = 0
            for c in perturbed:
                covered_mask |= cover_masks[c]
            uncovered = full_mask & ~covered_mask
            
            if uncovered:
                # Greedy repair
                code = perturbed[:]
                while uncovered:
                    best_gain = 0
                    candidates = []
                    for i in range(total_space):
                        if i in code:
                            continue
                        gain = (cover_masks[i] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [i]
                        elif gain == best_gain and gain > 0:
                            candidates.append(i)
                    if best_gain == 0:
                        break
                    chosen = random.choice(candidates)
                    code.append(chosen)
                    uncovered &= ~cover_masks[chosen]
            else:
                code = perturbed[:]
                
            # Optimize
            repaired = local_search_reduce(code)
            
            # Acceptance
            new_size = len(repaired)
            if new_size < best_size:
                best_code = repaired
                best_size = new_size
                current_code = set(best_code)
                # Reset temperature on improvement?
                temperature = min(temperature + 2.0, 10.0)
            else:
                # Accept worse with probability based on temperature
                delta = new_size - best_size
                if delta == 0:
                    # Same size, accept to explore
                    if random.random() < 0.5:
                        current_code = set(repaired)
                else:
                    # Worse
                    prob = 1.0 / (1.0 + (1.0 - (1.0 / (1.0 + delta * temperature))))
                    # Simpler: prob = exp(-delta / temperature)
                    prob = 2.71828 ** (-delta / temperature)
                    if random.random() < prob:
                        current_code = set(repaired)
                    else:
                        # Revert to best? Or keep perturbed? 
                        # Usually keep best to avoid drifting too far.
                        current_code = set(best_code)
                        
            temperature *= cooling
            if temperature < min_temperature:
                temperature = min_temperature
                
        return best_code

    # Main Loop
    best_code = None
    best_size = float('inf')
    
    # Initial deterministic greedy
    init_code = greedy_cover(randomized=False)
    opt_code = local_search_reduce(init_code)
    best_code = opt_code
    best_size = len(best_code)
    
    # Iterated Local Search on the best found so far
    if time.time() < deadline - 10.0:
        ils_code = iterated_local_search(best_code)
        if len(ils_code) < best_size:
            best_code = ils_code
            best_size = len(best_code)
            
    # Randomized Restarting
    # Try different random seeds for greedy
    while time.time() < deadline - 10.0:
        # Randomized greedy
        rand_code = greedy_cover(randomized=True)
        rand_code = local_search_reduce(rand_code)
        
        if len(rand_code) < best_size:
            best_code = rand_code
            best_size = len(best_code)
            # If we improved, do a quick ILS on this new best
            if time.time() < deadline - 15.0:
                improved_ils = iterated_local_search(best_code)
                if len(improved_ils) < best_size:
                    best_code = improved_ils
                    best_size = len(best_code)
        else:
            # If not improved, maybe do a short ILS on this candidate to see if it leads to a better basin?
            # Only if we have time
            if time.time() < deadline - 20.0 and random.random() < 0.1:
                short_ils = iterated_local_search(rand_code)
                if len(short_ils) < best_size:
                    best_code = short_ils
                    best_size = len(best_code)

    # Final polish: try to remove any remaining redundancies in the best code
    final_code = local_search_reduce(best_code)
    
    return [all_words[i] for i in final_code]

if __name__ == "__main__":
    # Self-test for q=3, n=7, radius=1
    # construct(3, 7, 1)
    pass
