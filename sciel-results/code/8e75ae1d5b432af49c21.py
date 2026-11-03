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
    
    # Precompute neighbors
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
    
    def greedy_init():
        covered = [False] * total_words
        codewords = []
        codewords_set = set()
        uncovered_set = set(all_words)
        
        while uncovered_set:
            # Pick a random uncovered word to anchor the selection
            u = random.choice(list(uncovered_set))
            
            best_w = -1
            best_cov = -1
            
            # The best codeword must cover u, so it must be in neighbors[u]
            for w in neighbors[u]:
                if w in codewords_set:
                    continue
                
                cov = 0
                for v in neighbors[w]:
                    if v in uncovered_set:
                        cov += 1
                
                if cov > best_cov:
                    best_cov = cov
                    best_w = w
            
            if best_w == -1:
                # Fallback (should rarely happen if space is connected)
                best_w = random.choice(list(neighbors[u]))
            
            codewords.append(best_w)
            codewords_set.add(best_w)
            
            # Update uncovered set
            for v in neighbors[best_w]:
                if v in uncovered_set:
                    uncovered_set.discard(v)
                    
        return codewords
    
    def optimize(codewords, time_limit):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
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
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
                
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            would_uncover = []
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find best candidate from neighbors of the first uncovered word
            if would_uncover:
                first_uncovered = would_uncover[0]
                potential_candidates = list(neighbors[first_uncovered])
                random.shuffle(potential_candidates)
                
                best_new_cw = -1
                best_score = -1
                
                for new_cw in potential_candidates:
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
                    
                    # Score: Maximize redundancy gain
                    score = 0
                    for u in neighbors[new_cw]:
                        if u not in neighbors[old_cw]:
                            if coverage_count[u] >= 1:
                                score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[best_new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = best_new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(best_new_cw)
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    best_len = float('inf')
    
    while time.time() - start_time < time_limit:
        # Greedy is now very fast. We can do many restarts.
        # Allocate a small fixed time for optimization per restart, 
        # or dynamic based on remaining time.
        
        # If we have lots of time, do longer optimizations.
        # If we are close to limit, do quick optimizations.
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        # Dynamic time allocation: 
        # Early on, try longer optimizations. Later, try shorter ones to get more samples.
        # Let's try a mix.
        # 50% chance: long run (10s), 50% chance: short run (2s)
        if random.random() < 0.5:
            opt_time = min(10, remaining_time - 2)
        else:
            opt_time = min(2, remaining_time - 2)
            
        if opt_time <= 0:
            continue
            
        cw = greedy_init()
        cw = optimize(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
            
        # If we found a very good code (close to lower bound), we can spend more time polishing it?
        # The current structure doesn't support "polishing" the best code specifically, 
        # but since optimize is stochastic, running it again on a copy might help.
        # However, copying is expensive.
        # Let's just rely on multiple restarts.
        
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
