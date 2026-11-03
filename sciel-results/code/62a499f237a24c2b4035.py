import time
import random

def construct(q, n, radius):
    total_words = q ** n
    cover_size = 1 + n * (q - 1)
    
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

    # Precompute neighbors as list of lists
    neighbors = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        neighbors.append(idxs)
    
    # Bitset representation: for each codeword, a Python int whose bits represent covered words
    cover_bitsets = []
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        cover_bitsets.append(bs)
    
    # Reverse: for each word, which codewords cover it (as a list)
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbors[i]:
            reverse_cover[j].append(i)
    
    # Precompute: for each word, the set of codeword indices that cover it (as frozenset for fast intersection)
    reverse_cover_sets = [frozenset(rc) for rc in reverse_cover]
    
    # Precompute neighbor bitsets for fast uncovered checks
    neighbor_bitsets = cover_bitsets  # same thing
    
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
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                for w in neighbors[c]:
                    counts[w] -= 1
                if any(counts[w] == 0 for w in neighbors[c]):
                    for w in neighbors[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_remove_swap(code, deadline):
        """Try to remove each codeword; if it creates uncovered words, try to swap with one that covers all of them."""
        code = list(code)
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                for w in neighbors[c_remove]:
                    counts[w] -= 1
                
                # Check if any word becomes uncovered
                uncovered_words = [w for w in neighbors[c_remove] if counts[w] == 0]
                
                if not uncovered_words:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for w in neighbors[c_remove]:
                        counts[w] += 1
                    
                    # Try to find a swap: a codeword that covers all uncovered_words
                    # Use precomputed frozensets for fast intersection
                    if len(uncovered_words) == 1:
                        candidates = reverse_cover_sets[uncovered_words[0]]
                    else:
                        candidates = reverse_cover_sets[uncovered_words[0]]
                        for w in uncovered_words[1:]:
                            candidates = candidates & reverse_cover_sets[w]
                            if not candidates:
                                break
                    
                    if candidates and c_remove not in candidates:
                        best_w = random.choice(list(candidates))
                        # Remove c_remove, add best_w
                        for w in neighbors[c_remove]:
                            counts[w] -= 1
                        for w in neighbors[best_w]:
                            counts[w] += 1
                        code[i] = best_w
                        improved = True
                        break
        
        return code

    def local_search_critical(code, deadline):
        """Targeted local search: find words with count==1, try to reassign their coverage."""
        code = list(code)
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        # Build a mapping: for each codeword index in code, track which words it exclusively covers
        # More efficient: use the reverse_cover to quickly find alternatives
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Find all words with count == 1
            critical_words = [w for w in range(total_words) if counts[w] == 1]
            
            if not critical_words:
                break
            
            random.shuffle(critical_words)
            
            for w in critical_words:
                if time.time() >= deadline:
                    break
                
                # The unique codeword covering w must be in reverse_cover[w]
                # Find which one is actually in code
                covering_code = None
                for c in reverse_cover[w]:
                    if c in code:
                        covering_code = c
                        break
                
                if covering_code is None:
                    continue
                
                # Try to find another codeword in code that also covers w
                alternatives = [c for c in reverse_cover[w] if c != covering_code and c in code]
                
                if alternatives:
                    # Check if removing covering_code leaves any other word uncovered
                    can_remove = True
                    for w2 in neighbors[covering_code]:
                        if w2 == w:
                            continue
                        if counts[w2] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        for w2 in neighbors[covering_code]:
                            counts[w2] -= 1
                        code.remove(covering_code)
                        improved = True
                        break
        
        return code

    def repair_greedy(code, deadline):
        """Greedily add codewords to cover any uncovered words using bitsets."""
        code = list(code)
        # Use bitset for uncovered words
        uncovered_mask = (1 << total_words) - 1
        for c in code:
            uncovered_mask &= ~cover_bitsets[c]
        
        while time.time() < deadline:
            if uncovered_mask == 0:
                break
            
            # Find best codeword to add: max new coverage
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
        """Combine multiple local search strategies with time-aware allocation."""
        remaining = deadline - time.time()
        t1 = time.time() + remaining * 0.4
        t2 = time.time() + remaining * 0.7
        t3 = time.time() + remaining * 0.9
        
        code = local_search_remove_swap(code, t1)
        code = local_search_critical(code, t2)
        code = local_search_remove_swap(code, t3)
        code = reduce_code(code)
        # Final tight pass
        code = local_search_remove_swap(code, deadline)
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = full_local_search(code, deadline - 15)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search with varied perturbations
    seed = 1
    while time.time() < deadline - 8:
        # Different perturbation strategies
        strategy = seed % 4
        
        if strategy == 0:
            # Small random removal + repair
            k = random.randint(1, 2)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 1:
            # Larger random removal + repair
            k = random.randint(3, 5)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 2:
            # Swap-based perturbation: remove 2, add 2 random, then repair
            perturbed_code = list(best_code)
            if len(perturbed_code) >= 2:
                remove_idx = random.sample(range(len(perturbed_code)), 2)
                for idx in sorted(remove_idx, reverse=True):
                    perturbed_code.pop(idx)
                for _ in range(2):
                    perturbed_code.append(random.randint(0, total_words - 1))
            perturbed_code = repair_greedy(perturbed_code, deadline - 8)
        else:
            # Single swap perturbation: remove 1, add best replacement
            perturbed_code = list(best_code)
            if len(perturbed_code) >= 1:
                remove_idx = random.randint(0, len(perturbed_code) - 1)
                perturbed_code.pop(remove_idx)
                # Find best codeword to add back
                uncovered_mask = (1 << total_words) - 1
                for c in perturbed_code:
                    uncovered_mask &= ~cover_bitsets[c]
                if uncovered_mask != 0:
                    best_gain = 0
                    best_cand = None
                    for i in range(total_words):
                        gain = bin(uncovered_mask & cover_bitsets[i]).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = i
                    if best_cand is not None:
                        perturbed_code.append(best_cand)
                else:
                    # All covered, just add a random one to create diversity
                    perturbed_code.append(random.randint(0, total_words - 1))
            perturbed_code = repair_greedy(perturbed_code, deadline - 8)
        
        # Local search on perturbed code
        perturbed_code = full_local_search(perturbed_code, deadline - 8)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # Fresh restarts less frequently
        if seed % 12 == 0 and time.time() < deadline - 25:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = full_local_search(fresh_code, deadline - 15)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    # Final polish: one more tight local search on the best code
    if time.time() < deadline - 3:
        final_code = full_local_search(best_code, deadline - 2)
        if len(final_code) < best_size:
            best_size = len(final_code)
            best_code = final_code

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
