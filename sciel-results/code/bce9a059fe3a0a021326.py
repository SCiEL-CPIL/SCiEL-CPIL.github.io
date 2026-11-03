import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    cover_size = 1 + n * (q - 1)
    
    # Precompute neighbors using numpy for speed
    # Each word is an index from 0 to total_words-1
    # neighbors[i] is a list of indices within Hamming distance <= radius
    
    # Build neighbor lists
    neighbors = []
    for i in range(total_words):
        w = []
        idx = i
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                j = 0
                for val in new_w:
                    j = j * q + val
                idxs.append(j)
        neighbors.append(idxs)
    
    # Convert to numpy arrays for faster operations
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbors]
    
    # Build reverse cover: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbors[i]:
            reverse_cover[j].append(i)
    
    # Convert reverse cover to numpy arrays
    reverse_cover_arrays = [np.array(rc, dtype=np.int32) for rc in reverse_cover]
    
    # Precompute cover bitsets as Python ints
    cover_bitsets = []
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        cover_bitsets.append(bs)
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = (1 << total_words) - 1
        code = []
        
        while uncovered_mask != 0:
            # Find codeword with max gain
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                gain = bin(uncovered_mask & cover_bitsets[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code
    
    def reduce_code(code):
        # Use counts array for efficiency
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbor_arrays[c]] -= 1
                if np.any(counts[neighbor_arrays[c]] == 0):
                    counts[neighbor_arrays[c]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code
    
    def local_search_remove_swap(code, deadline):
        """Try to remove each codeword; if it creates uncovered words, try to swap with one that covers all of them."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                nbrs = neighbor_arrays[c_remove]
                counts[nbrs] -= 1
                
                # Check if any word becomes uncovered
                uncovered_words = nbrs[counts[nbrs] == 0]
                
                if len(uncovered_words) == 0:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    counts[nbrs] += 1
                    
                    # Try to find a swap: a codeword that covers all uncovered_words
                    # Intersection of reverse_cover for each uncovered word
                    candidate_set = set(reverse_cover[uncovered_words[0]])
                    for w in uncovered_words[1:]:
                        candidate_set &= set(reverse_cover[w])
                    
                    candidate_set.discard(c_remove)
                    if candidate_set:
                        best_w = random.choice(list(candidate_set))
                        # Remove c_remove, add best_w
                        counts[nbrs] -= 1
                        counts[neighbor_arrays[best_w]] += 1
                        code[i] = best_w
                        improved = True
                        break
        
        return code
    
    def local_search_critical(code, deadline):
        """Targeted local search: find words with count==1, try to reassign their coverage."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Find all words with count == 1
            critical_words = np.where(counts == 1)[0]
            
            if len(critical_words) == 0:
                break
            
            critical_words_list = critical_words.tolist()
            random.shuffle(critical_words_list)
            
            for w in critical_words_list:
                if time.time() >= deadline:
                    break
                
                # Find which codeword covers w
                covering_code = None
                for c in code:
                    if w in neighbors[c]:
                        covering_code = c
                        break
                
                if covering_code is None:
                    continue
                
                # Try to find another codeword that also covers w
                # and is in the code
                alternatives = [c for c in code if c != covering_code and w in neighbors[c]]
                
                if alternatives:
                    # If there's an alternative, we might be able to remove covering_code
                    # if no other word depends only on it
                    alt = random.choice(alternatives)
                    # Check if removing covering_code leaves any word uncovered
                    # (other than w, which is covered by alt)
                    can_remove = True
                    for w2 in neighbors[covering_code]:
                        if w2 == w:
                            continue
                        if counts[w2] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        # Remove covering_code
                        counts[neighbor_arrays[covering_code]] -= 1
                        code.remove(covering_code)
                        improved = True
                        break
        
        return code
    
    def repair_greedy(code, deadline):
        """Greedily add codewords to cover any uncovered words."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        
        while time.time() < deadline:
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Find best codeword to add
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                gain = np.sum(counts[neighbor_arrays[i]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            counts[neighbor_arrays[chosen]] += 1
        
        return code
    
    def perturb_repair(code, k, deadline):
        """Remove k random codewords and repair."""
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        code = repair_greedy(code, deadline)
        return code
    
    def full_local_search(code, deadline):
        """Combine multiple local search strategies."""
        code = local_search_remove_swap(code, deadline)
        code = local_search_critical(code, deadline)
        code = local_search_remove_swap(code, deadline)
        code = reduce_code(code)
        return code
    
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = full_local_search(code, deadline - 10)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search with varied perturbations
    seed = 1
    while time.time() < deadline - 5:
        # Different perturbation strategies
        strategy = seed % 3
        
        if strategy == 0:
            # Small random removal + repair
            k = random.randint(1, 2)
            perturbed_code = perturb_repair(best_code, k, deadline - 5)
        elif strategy == 1:
            # Larger random removal + repair
            k = random.randint(3, 5)
            perturbed_code = perturb_repair(best_code, k, deadline - 5)
        else:
            # Swap-based perturbation: remove 2, add 2 random, then repair
            perturbed_code = list(best_code)
            if len(perturbed_code) >= 2:
                remove_idx = random.sample(range(len(perturbed_code)), 2)
                for idx in sorted(remove_idx, reverse=True):
                    perturbed_code.pop(idx)
                # Add 2 random codewords
                for _ in range(2):
                    perturbed_code.append(random.randint(0, total_words - 1))
            perturbed_code = repair_greedy(perturbed_code, deadline - 5)
        
        # Local search on perturbed code
        perturbed_code = full_local_search(perturbed_code, deadline - 5)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # Fresh restarts
        if seed % 8 == 0 and time.time() < deadline - 20:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = full_local_search(fresh_code, deadline - 10)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    result = []
    for idx in best_code:
        w = []
        idx = idx
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        result.append(w)
    
    return result
