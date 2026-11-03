"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts.
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
    
    # Precompute neighbors as numpy arrays for faster set operations
    neighbors = {}
    for w in all_words:
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
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
                            new_int = word_to_int(new_word, q)
                            if new_int not in neigh:
                                neigh.append(new_int)
        neighbors[w] = set(neigh)
    
    # Precompute neighbor lists for faster iteration
    neighbor_lists = {w: list(neigh) for w, neigh in neighbors.items()}
    
    def greedy_init():
        covered = np.zeros(total_words, dtype=np.int32)
        codewords = []
        uncovered_count = total_words
        remaining = set(all_words)
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            # Randomize order to break ties
            rem_list = list(remaining)
            random.shuffle(rem_list)
            
            for w in rem_list:
                neigh_arr = neighbor_lists[w]
                new_cov = 0
                for u in neigh_arr:
                    if covered[u] == 0:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                if new_cov == uncovered_count:
                    break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            remaining.discard(best_word)
            for u in neighbor_lists[best_word]:
                if covered[u] == 0:
                    covered[u] = 1
                    uncovered_count -= 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbor_lists[cw]:
                coverage_count[u] += 1
                
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbor_lists[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbor_lists[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        no_improve_count = 0
        best_len = len(codewords)
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                no_improve_count = 0
                best_len = len(codewords)
                continue
                
            if len(codewords) == 0:
                break
            
            if no_improve_count > 200:
                # Try a random restart within the current solution
                if random.random() < 0.1:
                    # Remove a random codeword and re-add to escape local optimum
                    if len(codewords) > 1:
                        idx = random.randint(0, len(codewords) - 1)
                        old_cw = codewords[idx]
                        for u in neighbor_lists[old_cw]:
                            coverage_count[u] -= 1
                        codewords.pop(idx)
                        cw_set.discard(old_cw)
                        no_improve_count = 0
                        continue
            
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would become uncovered if we remove old_cw
            would_uncover = []
            for u in neighbor_lists[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                # Can remove without uncovering
                for u in neighbor_lists[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                no_improve_count = 0
                best_len = len(codewords)
                continue
            
            # Find best replacement
            # Candidates: words that cover all would_uncover words
            # Start with neighbors of the first uncovered word, then expand
            candidate_set = set()
            for u in would_uncover[:min(3, len(would_uncover))]:
                candidate_set.update(neighbors[u])
            
            # Also add some random candidates to escape local optima
            if random.random() < 0.3:
                for _ in range(min(50, total_words)):
                    candidate_set.add(random.randint(0, total_words - 1))
            
            best_new_cw = -1
            best_score = -1
            best_gain = -1
            
            # Evaluate candidates
            for new_cw in candidate_set:
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                
                # Check if new_cw covers all would_uncover
                valid = True
                for u in would_uncover:
                    if u not in neighbors[new_cw]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Score: number of newly covered words (words covered by new_cw but not old_cw)
                # that are currently covered (redundancy gain)
                gain = 0
                for u in neighbor_lists[new_cw]:
                    if u not in neighbors[old_cw]:
                        if coverage_count[u] >= 1:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_cw = new_cw
                    best_score = gain
            
            if best_new_cw != -1:
                for u in neighbor_lists[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbor_lists[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
                no_improve_count = 0
                best_len = len(codewords)
            else:
                no_improve_count += 1
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    best_len = float('inf')
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        if remaining_time <= 0:
            break
        
        # Greedy is fast, so give most time to optimize
        iter_time = min(18, remaining_time - 3)
        if iter_time <= 0:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
