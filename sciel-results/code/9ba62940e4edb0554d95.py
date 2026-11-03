"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Combines sampling-based greedy initialization with swap-based local search.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    deadline = time.time() + 115  # Safety margin before 120s kill

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

    # Precompute neighbors as sets for fast membership and intersection
    neighbors = [None] * total_words
    
    if radius == 1:
        for w in range(total_words):
            neigh = {w}
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.add(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = neigh
    elif radius == 0:
        for w in range(total_words):
            neighbors[w] = {w}
    else:
        for w in range(total_words):
            neigh = {w}
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.add(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.add(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = neigh

    # Reverse neighbors: for each word u, which codewords cover it
    rev_neighbors = [set() for _ in range(total_words)]
    for w in range(total_words):
        for u in neighbors[w]:
            rev_neighbors[u].add(w)

    def greedy_init():
        """Sampling-based greedy: pick candidates from neighborhoods of uncovered words."""
        covered = [False] * total_words
        codewords = []
        uncovered = set(range(total_words))
        
        while uncovered:
            # Sample uncovered words to generate candidates
            if len(uncovered) > 200:
                sample = random.sample(list(uncovered), min(50, len(uncovered)))
            else:
                sample = list(uncovered)
            
            # Collect candidate codewords from neighborhoods of sampled uncovered words
            candidates = set()
            for u in sample:
                candidates.update(rev_neighbors[u])
            
            if not candidates:
                # Fallback
                u = next(iter(uncovered))
                candidates = set(rev_neighbors[u])
            
            # Evaluate candidates by new coverage
            best_word = -1
            best_new_cov = -1
            
            for w in candidates:
                new_cov = 0
                for u2 in neighbors[w]:
                    if u2 in uncovered:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if best_new_cov >= len(neighbors[w]):
                        break
            
            if best_word == -1:
                u = next(iter(uncovered))
                best_word = next(iter(rev_neighbors[u]))
                best_new_cov = 1
            
            codewords.append(best_word)
            for u2 in neighbors[best_word]:
                if u2 in uncovered:
                    uncovered.discard(u2)
        
        return codewords

    def optimize(codewords, time_limit):
        """Local search: remove redundant, swap to create redundancy."""
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def remove_redundant():
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

        remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if remove_redundant():
                continue
            
            if not codewords:
                break
            
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            
            # Find words uniquely covered by old_cw
            private_words = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
            
            if not private_words:
                # Can remove directly
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find replacement covering all private_words
            potential = set(neighbors[private_words[0]])
            for pw in private_words[1:]:
                potential.intersection_update(neighbors[pw])
                if not potential:
                    break
            
            if not potential:
                continue
            
            valid_candidates = [c for c in potential if c not in cw_set and c != old_cw]
            
            if not valid_candidates:
                continue
            
            random.shuffle(valid_candidates)
            # Limit candidates for speed
            valid_candidates = valid_candidates[:200]
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in valid_candidates:
                # Score: maximize overlap with already-covered words (creates redundancy)
                score = 0
                for u in neighbors[new_cw]:
                    if coverage_count[u] > 0:
                        score += coverage_count[u] - 1
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
        
        remove_redundant()
        return codewords

    def polish(codewords, time_limit):
        """Final polish: try all codewords for removal or swap with broader search."""
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        start_time = time.time()
        
        def try_remove_all():
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords)):
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
                        break
        
        try_remove_all()
        
        while time.time() - start_time < time_limit:
            if try_remove_all():
                continue
            
            if not codewords:
                break
            
            # Try each codeword in order
            for idx in range(len(codewords)):
                old_cw = codewords[idx]
                
                private_words = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
                
                if not private_words:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    break  # Restart loop since indices changed
                
                # Find replacement
                potential = set(neighbors[private_words[0]])
                for pw in private_words[1:]:
                    potential.intersection_update(neighbors[pw])
                    if not potential:
                        break
                
                if not potential:
                    continue
                
                valid_candidates = [c for c in potential if c not in cw_set and c != old_cw]
                
                if not valid_candidates:
                    continue
                
                # Score candidates
                best_new_cw = -1
                best_score = -1
                
                for new_cw in valid_candidates:
                    score = 0
                    for u in neighbors[new_cw]:
                        if coverage_count[u] > 0:
                            score += coverage_count[u] - 1
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
                    break  # Restart loop
        
        try_remove_all()
        return codewords

    # Multiple restarts
    start_time = time.time()
    best_codewords = []
    best_len = float('inf')
    
    # First restart gets more time
    iter_time = 30
    cw = greedy_init()
    if time.time() < deadline - 5:
        opt_time = min(25, deadline - time.time() - 5)
        if opt_time > 0:
            cw = optimize(cw, opt_time)
        if time.time() < deadline - 3:
            pol_time = min(10, deadline - time.time() - 3)
            if pol_time > 0:
                cw = polish(cw, pol_time)
    
    if len(cw) < best_len:
        best_len = len(cw)
        best_codewords = cw
    
    # Subsequent restarts with less time each
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        # Decreasing time allocation
        remaining = deadline - time.time() - 5
        if remaining < 15:
            break
        iter_time = min(15, remaining * 0.4)
        if iter_time < 5:
            break
        
        cw = greedy_init()
        opt_time = min(iter_time - 2, 10)
        if opt_time > 0 and time.time() < deadline - 5:
            cw = optimize(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
            # If we improved, do a quick polish
            if time.time() < deadline - 3:
                pol_time = min(5, deadline - time.time() - 3)
                if pol_time > 0:
                    polished = polish(best_codewords[:], pol_time)
                    if len(polished) < best_len:
                        best_len = len(polished)
                        best_codewords = polished
    
    # Final polish on best result
    if time.time() < deadline - 2 and best_codewords:
        pol_time = min(5, deadline - time.time() - 2)
        if pol_time > 0:
            polished = polish(best_codewords[:], pol_time)
            if len(polished) < best_len:
                best_len = len(polished)
                best_codewords = polished
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
