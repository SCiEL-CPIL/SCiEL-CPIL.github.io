import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
    N = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbor masks and lists
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    for c_idx in range(N):
        c_word = all_words[c_idx]
        mask = 1 << c_idx
        neighbor_lists[c_idx].append(c_idx)
        
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
        else:
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
                        
        neighbor_masks[c_idx] = mask
    
    # Precompute: for each word, which codewords cover it (inverse mapping)
    # This helps with criticality-based greedy
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)

    def greedy_cover_criticality():
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # Track coverage count for each word
        cover_count = np.zeros(N, dtype=np.int32)
        
        # Precompute criticality for all words
        criticality = np.array([len(word_to_coders[w]) for w in range(N)], dtype=np.int32)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Get list of uncovered word indices
            uncovered_words = []
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                uncovered_words.append(w_idx)
                temp ^= lsb
            
            if not uncovered_words:
                break
            
            # Find word with minimum criticality among uncovered
            min_crit = N + 1
            target_word = -1
            
            for w_idx in uncovered_words:
                crit = criticality[w_idx]
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:  # Can't get better
                    break
            
            if target_word == -1:
                break
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                
                # Count how many new words this codeword would cover
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                # Fallback: pick any codeword that covers target_word
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
            
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords - vectorized check"""
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Vectorized check: any word in c's neighborhood with cover_count == 1?
                if not np.any(cover_counts[neighbor_lists[c]] == 1):
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def two_opt_local_search(code):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each codeword
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                
                c_remove = code[i]
                nl_remove = neighbor_lists[c_remove]
                
                # Find words that would become uncovered if we remove c_remove
                newly_uncovered = [w for w in nl_remove if cover_counts[w] == 1]
                
                if not newly_uncovered:
                    # Can remove without adding
                    code.pop(i)
                    for w in nl_remove:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    improved = True
                    break
                
                # Try to find a single codeword that covers all newly_uncovered
                # and is not already in code
                best_add = -1
                best_gain = -1
                
                # For each newly uncovered word, find codewords that cover it
                # We need a codeword that covers ALL newly_uncovered words
                candidate_codewords = set()
                for w in newly_uncovered:
                    for c in word_to_coders[w]:
                        if c not in code_set:
                            candidate_codewords.add(c)
                
                if not candidate_codewords:
                    continue
                
                # Limit candidates to speed up: prioritize by potential gain
                # Sample at most 50 candidates if there are too many
                candidates = list(candidate_codewords)
                if len(candidates) > 50:
                    random.shuffle(candidates)
                    candidates = candidates[:50]
                
                # Check each candidate
                for c_add in candidates:
                    # Check if c_add covers all newly_uncovered
                    covers_all = True
                    for w in newly_uncovered:
                        if (neighbor_masks[c_add] >> w) & 1 == 0:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Count how many new words c_add would cover (beyond newly_uncovered)
                    new_coverage = np.sum(cover_counts[neighbor_lists[c_add]] == 0)
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_add = c_add
                
                if best_add != -1:
                    # Remove c_remove and add best_add
                    code.pop(i)
                    for w in nl_remove:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    
                    code.append(best_add)
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1
                    code_set.add(best_add)
                    
                    improved = True
                    break
        
        return code
    
    def cellular_automaton_polish(code, cover_counts, max_steps=50):
        """
        Discrete-time cellular automaton on the code structure.
        Each step, we look at codewords with low "deficit" (words they uniquely cover)
        and try to swap them with codewords that have high deficit, allowing
        global structural reorganization that escapes single-swap local minima.
        """
        code = list(code)
        code_set = set(code)
        cover_counts = cover_counts.copy()
        
        # Compute deficit for each codeword: number of words uniquely covered
        deficits = np.zeros(len(code), dtype=np.int32)
        for i, c in enumerate(code):
            deficits[i] = np.sum(cover_counts[neighbor_lists[c]] == 1)
        
        improved = True
        step = 0
        while improved and step < max_steps and time.time() < deadline - 5:
            improved = False
            step += 1
            
            # Sort codewords by deficit (lowest first - most "redundant" first)
            order = np.argsort(deficits)
            
            for i in order:
                if time.time() > deadline - 5:
                    break
                
                c_remove = code[i]
                bp_remove = neighbor_lists[c_remove]
                
                # Find words that would become uncovered
                newly_uncovered = [w for w in bp_remove if cover_counts[w] == 1]
                
                if not newly_uncovered:
                    # Pure removal
                    code.pop(i)
                    for w in bp_remove:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    # Recompute deficits
                    for j in range(len(code)):
                        deficits[j] = np.sum(cover_counts[neighbor_lists[code[j]]] == 1)
                    improved = True
                    break
                
                # Find candidate replacements
                candidate_set = None
                for w in newly_uncovered:
                    cc = set(word_to_coders[w])
                    if candidate_set is None:
                        candidate_set = cc
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                
                # Limit candidates to speed up
                if len(candidates) > 50:
                    random.shuffle(candidates)
                    candidates = candidates[:50]
                
                # Score candidates: prefer those with high deficit (to maintain coverage)
                best_swap = None
                best_score = -1
                
                for c_add in candidates:
                    # Check if c_add covers all newly_uncovered
                    covers_all = True
                    for w in newly_uncovered:
                        if (neighbor_masks[c_add] >> w) & 1 == 0:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Score: number of words c_add uniquely covers (high = good)
                    unique_cover = np.sum(cover_counts[neighbor_lists[c_add]] == 0)
                    # Also consider overlap with existing code
                    overlap = np.sum(cover_counts[neighbor_lists[c_add]] > 0)
                    score = unique_cover * 2 - overlap  # Net gain
                    
                    if score > best_score:
                        best_score = score
                        best_swap = c_add
                
                if best_swap is not None:
                    # Perform swap
                    code[i] = best_swap
                    code_set.discard(c_remove)
                    code_set.add(best_swap)
                    
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    for w in neighbor_lists[best_swap]:
                        cover_counts[w] += 1
                    
                    # Recompute deficits
                    for j in range(len(code)):
                        deficits[j] = np.sum(cover_counts[neighbor_lists[code[j]]] == 1)
                    
                    improved = True
                    break
        
        return code, cover_counts
    
    # Run criticality-based greedy
    best_code = greedy_cover_criticality()
    best_code = local_search_remove(best_code)
    best_code = two_opt_local_search(best_code)
    
    # Try multiple random restarts if time permits
    best_size = len(best_code)
    
    for restart in range(1, 30):
        if time.time() > deadline - 15:
            break
        
        # Randomized greedy: shuffle the order of considering codewords
        random.seed(restart * 42)
        code = greedy_cover_criticality()
        code = local_search_remove(code)
        code = two_opt_local_search(code)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    # Apply cellular automaton polish to the best code
    if time.time() < deadline - 15:
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in best_code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        polished_code, _ = cellular_automaton_polish(best_code, cover_counts, max_steps=50)
        
        if len(polished_code) < best_size:
            best_code = polished_code
            best_size = len(best_code)
        
        # Second polish pass with more steps if time permits
        if time.time() < deadline - 8 and len(polished_code) == best_size:
            cover_counts2 = np.zeros(N, dtype=np.int32)
            for c in polished_code:
                for w in neighbor_lists[c]:
                    cover_counts2[w] += 1
            polished_code2, _ = cellular_automaton_polish(polished_code, cover_counts2, max_steps=50)
            if len(polished_code2) < best_size:
                best_code = polished_code2
                best_size = len(best_code)
    
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [all_words[i] for i in best_code]
    return result
