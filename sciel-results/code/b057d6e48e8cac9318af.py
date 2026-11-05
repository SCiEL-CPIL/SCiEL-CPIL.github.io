"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    """
    if q == 3 and n == 6 and radius == 1:
        return _cover_ternary_6_1_optimized()
    else:
        return _generic_greedy_cover(q, n, radius)

def _cover_ternary_6_1_optimized():
    """
    Optimized solver for K_3(6,1) using Greedy Set Cover with Enhanced Local Search.
    """
    alphabet = range(3)
    all_words = list(itertools.product(alphabet, repeat=6))
    n_words = len(all_words) # 729
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets (indices of words covered by each word)
    # Each word covers itself + 12 neighbors (6 positions * 2 changes)
    cover_sets = []
    for w in all_words:
        covered = [word_to_idx[w]]
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.append(word_to_idx[tuple(new_w)])
        cover_sets.append(set(covered))
        
    full_set = set(range(n_words))
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = set(full_set)
        codewords = []
        
        while uncovered:
            best_word = -1
            best_count = -1
            candidates = []
            for i in range(n_words):
                count = len(cover_sets[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = i
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            if len(candidates) > 1:
                best_word = random.choice(candidates)
                
            codewords.append(best_word)
            uncovered -= cover_sets[best_word]
            
        return codewords

    def local_search(codewords):
        """
        Enhanced Local Search:
        1. Remove redundant codewords.
        2. Try to replace codewords to improve efficiency (reduce overlap).
        """
        # 1. Remove redundant codewords
        counts = [0] * n_words
        for cw in codewords:
            for idx in cover_sets[cw]:
                counts[idx] += 1
                
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords)):
                cw = codewords[i]
                can_remove = True
                for idx in cover_sets[cw]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in cover_sets[cw]:
                        counts[idx] -= 1
                    codewords.pop(i)
                    improved = True
                    break

        # 2. Replacement Search
        # We try to replace a codeword c with a non-codeword w.
        # The new set must be a valid cover.
        # Heuristic: Minimize the sum of coverage sizes? Or maximize "unique" coverage?
        # Let's try to minimize the total number of covered instances (sum of counts).
        # If we replace c with w, and the cover remains valid, we check if the new total count is lower.
        
        in_set = set(codewords)
        
        # We iterate a few times to allow cascading improvements
        for _ in range(10):
            improvement_made = False
            # Update counts
            counts = [0] * n_words
            for cw in codewords:
                for idx in cover_sets[cw]:
                    counts[idx] += 1
            
            # Try replacing each codeword
            for i in range(len(codewords)):
                cw = codewords[i]
                # Calculate current total overlap
                # If we remove cw, some words become uncovered (count 0).
                # We need to find a w that covers all those '0' words.
                
                # Words that would become uncovered if we remove cw
                # These are words where counts[idx] == 1 and idx in cover_sets[cw]
                critical_words = []
                for idx in cover_sets[cw]:
                    if counts[idx] == 1:
                        critical_words.append(idx)
                
                if not critical_words:
                    # cw is redundant, should have been removed in step 1
                    continue
                
                # Find a candidate w not in codewords that covers all critical_words
                # And ideally, w is "efficient"
                best_w = -1
                best_metric = float('inf')
                
                # To speed this up, we can restrict candidates to words that cover at least one critical word
                # Actually, w must cover ALL critical words.
                
                # Iterate through all potential w
                for w_idx in range(n_words):
                    if w_idx in in_set:
                        continue
                    
                    # Check if w covers all critical words
                    covers_all = True
                    for cw_idx in critical_words:
                        if cw_idx not in cover_sets[w_idx]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Metric: How many NEW overlaps does w create?
                    # Or simply: Does the new set have a lower total sum of counts?
                    # Sum of counts = sum_{idx} counts[idx]
                    # If we remove cw, sum decreases by |cover_sets[cw]|.
                    # If we add w, sum increases by |cover_sets[w]|.
                    # But we also change the counts of the words covered by w.
                    
                    # Let's just check if the new set is valid (it is, by construction of critical_words)
                    # and if it reduces the "waste".
                    # Waste = sum(counts[idx]) - n_words (since each word needs to be covered at least once)
                    # Minimizing sum(counts[idx]) is equivalent to minimizing waste.
                    
                    # Let's estimate the new sum.
                    # Current sum S.
                    # Remove cw: S' = S - |cover_sets[cw]|
                    # Add w: S'' = S' + |cover_sets[w]|
                    # However, this is an overestimate if w overlaps with others? No, sum of counts is just the sum of sizes of the sets.
                    # Wait, sum(counts[idx]) = sum_{cw in C} |cover_sets[cw]|.
                    # So minimizing sum of sizes is the metric.
                    
                    size_w = len(cover_sets[w_idx])
                    size_cw = len(cover_sets[cw])
                    
                    # If size_w < size_cw, it's a good swap in terms of size.
                    # But all sets have size 13! So this metric is constant.
                    # We need a better metric.
                    # Metric: Number of words covered by EXACTLY one codeword (efficiency).
                    # Or: Minimize the number of "extra" covers.
                    
                    # Let's try a different heuristic:
                    # Randomly pick a valid w.
                    # If we find a valid w, we can swap.
                    # To improve, we can try multiple w and pick the one that results in the most "removable" codewords in the next step?
                    # That's expensive.
                    
                    # Simpler: Just accept the first valid swap that doesn't increase the size (which is constant) 
                    # but maybe we can track "potential" for removal.
                    
                    # Let's just do a random valid swap to escape local minima.
                    # If we find any valid w, swap.
                    best_w = w_idx
                    break # Take the first valid one to keep it fast
                    
                if best_w != -1:
                    # Perform swap
                    codewords[i] = best_w
                    in_set.remove(cw)
                    in_set.add(best_w)
                    improvement_made = True
                    # Remove redundancies immediately after swap
                    counts = [0] * n_words
                    for cw2 in codewords:
                        for idx in cover_sets[cw2]:
                            counts[idx] += 1
                    # Quick redundancy check
                    for j in range(len(codewords)):
                        cw2 = codewords[j]
                        can_remove = True
                        for idx in cover_sets[cw2]:
                            if counts[idx] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            for idx in cover_sets[cw2]:
                                counts[idx] -= 1
                            codewords.pop(j)
                            in_set.remove(cw2)
                            break

            if not improvement_made:
                break

        return codewords

    best_codewords = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 110 # Leave 10s for evaluation
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < timeout:
        seed = num_restarts
        cw = greedy_cover(seed)
        cw = local_search(cw)
        
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            if best_size <= 73:
                break
                
        num_restarts += 1
        
    if best_codewords is None:
        best_codewords = greedy_cover(0)
        best_codewords = local_search(best_codewords)
        
    result = [list(all_words[i]) for i in best_codewords]
    return result

def _generic_greedy_cover(q, n, radius):
    """Generic greedy set cover for arbitrary q, n, radius."""
    alphabet = range(q)
    all_words = list(itertools.product(alphabet, repeat=n))
    n_words = len(all_words)
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets
    covered_sets = []
    for w in all_words:
        covered = []
        for candidate in all_words:
            dist = sum(1 for a, b in zip(w, candidate) if a != b)
            if dist <= radius:
                covered.append(candidate)
        covered_sets.append(set(word_to_idx[c] for c in covered))
    
    # Greedy set cover
    uncovered = set(range(n_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(n_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
            if best_count == len(uncovered):
                break
        
        codewords.append(best_word)
        uncovered -= covered_sets[best_word]
    
    # Try to remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            temp_codewords = codewords[:i] + codewords[i+1:]
            covered = set()
            for cw in temp_codewords:
                covered |= covered_sets[cw]
            if covered == set(range(n_words)):
                codewords = temp_codewords
                improved = True
                break
    
    result = [list(all_words[i]) for i in codewords]
    return result
# EVOLVE-BLOCK-END
