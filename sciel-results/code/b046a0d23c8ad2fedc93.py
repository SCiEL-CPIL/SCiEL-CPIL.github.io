"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search starting from a greedy feasible covering.
    Optimizes by iteratively swapping codewords to maximize unique coverage reduction.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbors using vectorized approach
    # For each word, compute all words within Hamming distance <= radius
    neighbors = [None] * total_words
    
    # Precompute single-position changes
    if radius >= 1:
        # For each position and each value change, compute the delta
        pos_deltas = []
        for pos in range(n):
            for val in range(q):
                # delta for changing position pos to val
                # We need to handle this carefully
                pass
    
    # Simpler: precompute neighbors for each word
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh_set = {w}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh_set.add(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            neigh_set.add(word_to_int(new_word, q))
        neighbors[w] = neigh_set
    
    # Greedy initialization with randomization
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    remaining = list(range(total_words))
    random.shuffle(remaining)
    
    while np.sum(~covered) > 0:
        best_word = -1
        best_new_coverage = -1
        
        # Sample a subset for speed if too many remaining
        check_list = remaining if len(remaining) <= 1000 else random.sample(remaining, 1000)
        
        for w in check_list:
            new_cov = np.sum(~covered[list(neighbors[w])])
            if new_cov > best_new_coverage:
                best_new_coverage = new_cov
                best_word = w
                if new_cov == np.sum(~covered):
                    break
        
        if best_word == -1:
            # Fallback: check all remaining
            for w in remaining:
                new_cov = np.sum(~covered[list(neighbors[w])])
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
        
        if best_word == -1:
            break
        
        codewords.append(best_word)
        remaining.remove(best_word)
        for u in neighbors[best_word]:
            covered[u] = True
    
    # Coverage count array
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        coverage_count[list(neighbors[cw])] += 1
    
    def try_remove_redundant():
        removed_any = False
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                cw = codewords[i]
                if np.min(coverage_count[list(neighbors[cw])]) > 1:
                    coverage_count[list(neighbors[cw])] -= 1
                    codewords.pop(i)
                    changed = True
                    removed_any = True
                    break
        return removed_any
    
    try_remove_redundant()
    
    start_time = time.time()
    time_limit = 115
    cw_set = set(codewords)
    
    # Track stagnation
    stagnant_count = 0
    best_size = len(codewords)
    
    while time.time() - start_time < time_limit:
        if try_remove_redundant():
            stagnant_count = 0
            best_size = len(codewords)
            continue
        
        if len(codewords) == 0:
            break
        
        # Prioritize codewords with most unique coverage
        unique_counts = []
        for i, cw in enumerate(codewords):
            unique_counts.append((np.sum(coverage_count[list(neighbors[cw])] == 1), i))
        
        # Sort by unique count descending, then randomize among top
        unique_counts.sort(reverse=True)
        top_candidates = [idx for _, idx in unique_counts[:min(20, len(unique_counts))]]
        random.shuffle(top_candidates)
        
        swapped = False
        for idx in top_candidates:
            old_cw = codewords[idx]
            
            # Find words uniquely covered by old_cw
            would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
            
            if len(would_uncover) == 0:
                # Should have been caught, but safety
                coverage_count[list(neighbors[old_cw])] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                swapped = True
                break
            
            # Generate candidates: neighbors of would_uncover words
            candidates = set()
            for u in would_uncover[:10]:  # Limit to first 10 for speed
                candidates.update(neighbors[u])
            candidates.discard(old_cw)
            candidates -= cw_set
            
            # Also add some random candidates
            for _ in range(3):
                candidates.add(random.randint(0, total_words - 1))
            
            candidates = list(candidates)
            random.shuffle(candidates)
            
            for new_cw in candidates:
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                
                # Check if new_cw covers all would_uncover
                if all(u in neighbors[new_cw] for u in would_uncover):
                    # Perform swap
                    coverage_count[list(neighbors[old_cw])] -= 1
                    coverage_count[list(neighbors[new_cw])] += 1
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                    swapped = True
                    stagnant_count = 0
                    break
            
            if swapped:
                break
        
        if not swapped:
            stagnant_count += 1
            # If stuck for a while, try a random perturbation
            if stagnant_count > 50 and len(codewords) > 5:
                # Remove a random codeword and re-add a random one that helps
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
                
                if len(would_uncover) > 0:
                    # Find a random word that covers most of would_uncover
                    best_new = None
                    best_cov = 0
                    for _ in range(50):
                        candidate = random.randint(0, total_words - 1)
                        if candidate in cw_set or candidate == old_cw:
                            continue
                        cov = sum(1 for u in would_uncover if u in neighbors[candidate])
                        if cov > best_cov:
                            best_cov = cov
                            best_new = candidate
                    
                    if best_new and best_cov >= len(would_uncover) * 0.8:
                        coverage_count[list(neighbors[old_cw])] -= 1
                        coverage_count[list(neighbors[best_new])] += 1
                        codewords[idx] = best_new
                        cw_set.discard(old_cw)
                        cw_set.add(best_new)
                        stagnant_count = 0
    
    try_remove_redundant()
    
    result = [int_to_word(cw, n, q) for cw in codewords]
    return result
# EVOLVE-BLOCK-END
