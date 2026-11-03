"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search: greedy init + iterative swap optimization.
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
    
    # Precompute neighbors efficiently
    neighbors = {}
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = set()
        neigh.add(w)
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.add(word_to_int(new_word, q))
        
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
                            neigh.add(word_to_int(new_word, q))
        
        neighbors[w] = neigh
    
    def greedy_init():
        """Fast greedy: pick word covering most uncovered, repeat."""
        covered = bytearray(total_words)  # 0 or 1
        codewords = []
        uncovered = total_words
        remaining = list(range(total_words))
        random.shuffle(remaining)
        
        while uncovered > 0:
            best_word = -1
            best_gain = -1
            
            for w in remaining:
                gain = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        gain += 1
                        if gain >= best_gain + 1:  # early stop if can't beat
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    if best_gain == uncovered:
                        break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            remaining.remove(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered -= 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Iterative swap: try removing/replacing codewords to reduce count."""
        coverage = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            """Remove any codeword that doesn't uniquely cover anything."""
            removed = False
            changed = True
            while changed:
                changed = False
                idxs = list(range(len(codewords)))
                random.shuffle(idxs)
                for i in idxs:
                    cw = codewords[i]
                    unique_cover = False
                    for u in neighbors[cw]:
                        if coverage[u] == 1:
                            unique_cover = True
                            break
                    if not unique_cover:
                        for u in neighbors[cw]:
                            coverage[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed = True
                        break
            return removed
        
        try_remove_redundant()
        start = time.time()
        
        while time.time() - start < time_limit:
            if try_remove_redundant():
                continue
            if not codewords:
                break
            
            # Pick a random codeword to try replacing
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would become uncovered
            would_uncover = [u for u in neighbors[old_cw] if coverage[u] == 1]
            
            if not would_uncover:
                # Can remove safely
                for u in neighbors[old_cw]:
                    coverage[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Try to find a replacement that covers all would_uncover
            # Sample candidates from union of neighbors of would_uncover words
            candidate_pool = set()
            for u in would_uncover:
                candidate_pool.update(neighbors[u])
            candidate_pool.discard(old_cw)
            
            if not candidate_pool:
                continue
            
            candidates = list(candidate_pool)
            random.shuffle(candidates)
            best_new = -1
            best_score = -1
            
            for new_cw in candidates[:50]:  # limit candidates for speed
                if new_cw in cw_set:
                    continue
                # Check if new_cw covers all would_uncover
                if not all(u in neighbors[new_cw] for u in would_uncover):
                    continue
                # Score: how many new words does it cover that old didn't?
                score = 0
                for u in neighbors[new_cw]:
                    if u not in neighbors[old_cw] and coverage[u] > 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_new = new_cw
            
            if best_new != -1:
                for u in neighbors[old_cw]:
                    coverage[u] -= 1
                for u in neighbors[best_new]:
                    coverage[u] += 1
                codewords[idx] = best_new
                cw_set.discard(old_cw)
                cw_set.add(best_new)
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts with time budget
    start_time = time.time()
    total_time = 115
    best = []
    
    while time.time() - start_time < total_time - 3:
        iter_budget = min(15, total_time - (time.time() - start_time))
        if iter_budget <= 1:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_budget - 1)
        
        if not best or len(cw) < len(best):
            best = cw
        
        if time.time() - start_time > total_time - 5:
            break
    
    return [int_to_word(cw, n, q) for cw in best]
# EVOLVE-BLOCK-END
