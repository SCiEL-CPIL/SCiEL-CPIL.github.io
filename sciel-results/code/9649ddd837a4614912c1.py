"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search starting from a greedy feasible covering,
    then optimizes by iteratively swapping codewords to maximize coverage.
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
    
    all_words = list(range(total_words))
    
    # Precompute neighbors as lists for faster iteration
    neighbors = [None] * total_words
    for w in all_words:
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
        neighbors[w] = list(neigh_set)
    
    # Greedy initialization with random tie-breaking
    covered = [False] * total_words
    codewords = []
    uncovered_count = total_words
    remaining = list(all_words)
    random.shuffle(remaining)
    
    while uncovered_count > 0 and remaining:
        best_word = -1
        best_new_coverage = -1
        for w in remaining:
            new_cov = 0
            for u in neighbors[w]:
                if not covered[u]:
                    new_cov += 1
            if new_cov > best_new_coverage:
                best_new_coverage = new_cov
                best_word = w
            if new_cov == uncovered_count:
                break
        if best_word == -1:
            break
        codewords.append(best_word)
        remaining.remove(best_word)
        for u in neighbors[best_word]:
            if not covered[u]:
                covered[u] = True
                uncovered_count -= 1
    
    coverage_count = [0] * total_words
    for cw in codewords:
        for u in neighbors[cw]:
            coverage_count[u] += 1
    
    # Helper to try removing redundant codewords
    def try_remove_redundant():
        removed_any = False
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                cw = codewords[i]
                can_remove = True
                for u in neighbors[cw]:
                    if coverage_count[u] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for u in neighbors[cw]:
                        coverage_count[u] -= 1
                    codewords.pop(i)
                    changed = True
                    removed_any = True
                    break
        return removed_any
    
    try_remove_redundant()
    
    start_time = time.time()
    time_limit = 115
    
    cw_set = set(codewords)
    
    # Local search: try swaps that maximize net coverage gain
    while time.time() - start_time < time_limit:
        if try_remove_redundant():
            continue
        
        if len(codewords) == 0:
            break
        
        # Pick a random codeword to try replacing
        idx = random.randint(0, len(codewords) - 1)
        old_cw = codewords[idx]
        
        # Count how many words are uniquely covered by old_cw
        unique_count = 0
        for u in neighbors[old_cw]:
            if coverage_count[u] == 1:
                unique_count += 1
        
        if unique_count == 0:
            # Redundant, should have been caught
            for u in neighbors[old_cw]:
                coverage_count[u] -= 1
            codewords.pop(idx)
            cw_set.discard(old_cw)
            continue
        
        # Find candidate new codewords
        # Strategy: try neighbors of old_cw first, then random samples
        candidates = set()
        
        # Add neighbors of old_cw
        for u in neighbors[old_cw]:
            candidates.add(u)
        
        # Add some random candidates to escape local optima
        num_random = min(20, total_words // 10)
        for _ in range(num_random):
            candidates.add(random.randint(0, total_words - 1))
        
        # Remove existing codewords from candidates
        candidates -= cw_set
        candidates.discard(old_cw)
        
        if not candidates:
            continue
        
        # Evaluate each candidate: compute net gain in coverage
        best_candidate = None
        best_gain = -1
        
        for new_cw in candidates:
            # Calculate how many new words would be covered
            new_coverage = 0
            for u in neighbors[new_cw]:
                if coverage_count[u] == 0:
                    new_coverage += 1
            
            # Calculate how many unique words would be lost
            lost_unique = 0
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1 and u not in neighbors[new_cw]:
                    lost_unique += 1
            
            gain = new_coverage - lost_unique
            
            if gain > best_gain:
                best_gain = gain
                best_candidate = new_cw
        
        # Accept if gain is positive or zero (zero gain swaps can help escape local optima)
        if best_candidate is not None and best_gain >= 0:
            # Perform the swap
            for u in neighbors[old_cw]:
                coverage_count[u] -= 1
            for u in neighbors[best_candidate]:
                coverage_count[u] += 1
            codewords[idx] = best_candidate
            cw_set.discard(old_cw)
            cw_set.add(best_candidate)
        
        # Occasionally try a perturbation: remove a random codeword and re-add
        if random.random() < 0.1 and len(codewords) > 1:
            rem_idx = random.randint(0, len(codewords) - 1)
            rem_cw = codewords[rem_idx]
            
            # Check if we can remove it without breaking coverage
            can_remove = True
            for u in neighbors[rem_cw]:
                if coverage_count[u] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                for u in neighbors[rem_cw]:
                    coverage_count[u] -= 1
                codewords.pop(rem_idx)
                cw_set.discard(rem_cw)
                
                # Try to re-add a better codeword
                # Find the most uncovered word
                max_uncovered = 0
                best_add = -1
                for w in range(total_words):
                    if coverage_count[w] == 0:
                        # This word is uncovered, find a codeword that covers it
                        pass
                
                # Greedy re-add: find codeword that covers most uncovered words
                best_add_cw = -1
                best_new_cov = -1
                sample_size = min(50, total_words)
                for _ in range(sample_size):
                    w = random.randint(0, total_words - 1)
                    if w in cw_set:
                        continue
                    new_cov = 0
                    for u in neighbors[w]:
                        if coverage_count[u] == 0:
                            new_cov += 1
                    if new_cov > best_new_cov:
                        best_new_cov = new_cov
                        best_add_cw = w
                
                if best_add_cw != -1:
                    for u in neighbors[best_add_cw]:
                        coverage_count[u] += 1
                    codewords.append(best_add_cw)
                    cw_set.add(best_add_cw)
                else:
                    # Restore
                    for u in neighbors[rem_cw]:
                        coverage_count[u] += 1
                    codewords.insert(rem_idx, rem_cw)
                    cw_set.add(rem_cw)
    
    try_remove_redundant()
    
    result = [int_to_word(cw, n, q) for cw in codewords]
    return result
# EVOLVE-BLOCK-END
