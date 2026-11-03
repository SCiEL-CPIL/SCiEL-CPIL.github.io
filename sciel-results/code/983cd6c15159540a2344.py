import random
import numpy as np
import time
from collections import defaultdict

def construct(q, n, radius):
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

    # Precompute neighbors
    neighbors = [None] * total_words
    if radius == 1:
        for w in range(total_words):
            word = int_to_word(w, n, q)
            neigh = {w}
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    word[pos] = new_val
                    neigh.add(word_to_int(word, q))
                word[pos] = orig
            neighbors[w] = neigh
    else:
        # General radius (likely not needed for this specific task instance q=3, n=8, r=1)
        for w in range(total_words):
            word = int_to_word(w, n, q)
            neigh = {w}
            # Add radius 1
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    word[pos] = new_val
                    neigh.add(word_to_int(word, q))
                word[pos] = orig
            # Add radius 2
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    orig1 = word[pos1]
                    for new_val1 in range(q):
                        if new_val1 == orig1:
                            continue
                        word[pos1] = new_val1
                        for pos2 in range(pos1 + 1, n):
                            orig2 = word[pos2]
                            for new_val2 in range(q):
                                if new_val2 == orig2:
                                    continue
                                word[pos2] = new_val2
                                neigh.add(word_to_int(word, q))
                            word[pos2] = orig2
                        word[pos1] = orig1
            neighbors[w] = neigh

    def greedy_init():
        covered = bytearray(total_words)
        uncovered_count = total_words
        codewords = []
        uncovered_set = set(range(total_words))
        
        while uncovered_count > 0:
            best_word = -1
            best_new = 0
            
            # Sample candidates from neighbors of uncovered words
            if len(uncovered_set) > 100:
                sample = random.sample(list(uncovered_set), 100)
            else:
                sample = list(uncovered_set)
            
            candidates = set()
            for u in sample:
                candidates.update(neighbors[u])
            
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new:
                    best_new = new_cov
                    best_word = w
            
            if best_word == -1:
                # Fallback
                u = next(iter(uncovered_set))
                best_word = next(iter(neighbors[u]))
                best_new = 1

            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered_count -= 1
                    uncovered_set.discard(u)
        
        return codewords

    def optimize(codewords, time_limit):
        coverage = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed = 0
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords)):
                    cw = codewords[i]
                    is_redundant = True
                    for u in neighbors[cw]:
                        if coverage[u] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for u in neighbors[cw]:
                            coverage[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed += 1
                        break
            return removed

        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # 1. Try to find a valid swap that increases redundancy
            # Score: minimize unique coverage count
            current_unique = sum(1 for c in coverage if c == 1)
            
            improvement_found = False
            # Sample a subset of codewords to try swapping to save time
            # If len is small, try all
            if len(codewords) < 100:
                indices_to_try = list(range(len(codewords)))
                random.shuffle(indices_to_try)
            else:
                indices_to_try = random.sample(range(len(codewords)), min(20, len(codewords)))
            
            for idx in indices_to_try:
                if idx >= len(codewords):
                    continue
                old_cw = codewords[idx]
                
                # Find unique words
                unique_words = [u for u in neighbors[old_cw] if coverage[u] == 1]
                
                if not unique_words:
                    # Redundant, remove it
                    for u in neighbors[old_cw]:
                        coverage[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    improvement_found = True
                    break
                
                # Find replacement
                # Candidate must cover all unique_words
                # Intersect neighbors of unique_words
                # Since unique_words is small (usually), we can just check candidates
                # Candidates: all words not in cw_set
                # To speed up, only check words that are in neighbors of some unique word?
                # Actually, the replacement MUST be in the intersection of neighbors of unique_words.
                
                # Heuristic: Pick a random candidate from the intersection
                # Or evaluate a few.
                
                # Let's find the intersection set
                # Start with neighbors of the first unique word
                candidates = set(neighbors[unique_words[0]])
                for u in unique_words[1:]:
                    candidates &= neighbors[u]
                
                candidates -= cw_set
                candidates.discard(old_cw)
                
                if not candidates:
                    continue
                    
                # Evaluate candidates to pick the best one
                # Best one: minimizes new unique words, or maximizes redundancy
                best_new_cw = -1
                best_score = -1
                
                # Limit candidates to check
                if len(candidates) > 50:
                    candidates = random.sample(list(candidates), 50)
                
                for new_cw in candidates:
                    # Check validity: covers all unique_words (guaranteed by intersection)
                    # Calculate new unique count impact
                    # We remove old_cw, add new_cw
                    # Words in neighbors[old_cw] lose 1 coverage
                    # Words in neighbors[new_cw] gain 1 coverage
                    
                    # Simplified score: 
                    # Count how many words in neighbors[new_cw] currently have coverage 1
                    # If they become 2, good.
                    # Count how many words in neighbors[new_cw] currently have coverage 0
                    # They become 1, bad (but they were uncovered? No, coverage is global)
                    # Wait, coverage is maintained. 
                    
                    # Let's just count the new unique words created by new_cw
                    # A word u in neighbors[new_cw] becomes unique if coverage[u] == 1 (before adding)
                    # But we also remove old_cw.
                    
                    # Simulate:
                    new_unique_count = 0
                    for u in neighbors[new_cw]:
                        # coverage[u] is current
                        # After removing old_cw:
                        cov_old = coverage[u]
                        if u in neighbors[old_cw]:
                            cov_old -= 1
                        # After adding new_cw:
                        cov_new = cov_old + 1
                        if cov_new == 1:
                            new_unique_count += 1
                    
                    # We want to MINIMIZE new_unique_count
                    # Also, we want to ensure no word becomes uncovered (cov_new == 0)
                    # Since we picked new_cw from intersection of neighbors of unique_words of old_cw,
                    # the unique_words are covered.
                    # What about other words?
                    # If a word was covered only by old_cw, it's in unique_words, so it's covered by new_cw.
                    # If a word was covered by old_cw and others, it remains covered.
                    # So validity is guaranteed.
                    
                    if new_unique_count < best_score or best_score == -1:
                        best_score = new_unique_count
                        best_new_cw = new_cw
                        
                        if new_unique_count == 0:
                            break # Perfect swap, no new unique words
                
                if best_new_cw != -1:
                    # Perform swap
                    for u in neighbors[old_cw]:
                        coverage[u] -= 1
                    for u in neighbors[best_new_cw]:
                        coverage[u] += 1
                    codewords[idx] = best_new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(best_new_cw)
                    improvement_found = True
                    break
            
            if not improvement_found:
                # Try to remove redundant again
                if try_remove_redundant() > 0:
                    continue
                
                # Perturbation: Iterated Local Search
                # Remove k random codewords and repair
                k = random.randint(3, 8)
                if len(codewords) > k:
                    to_remove = random.sample(codewords, k)
                    for cw in to_remove:
                        idx = codewords.index(cw)
                        for u in neighbors[cw]:
                            coverage[u] -= 1
                        codewords.pop(idx)
                        cw_set.discard(cw)
                    
                    # Repair
                    uncovered = [u for u in range(total_words) if coverage[u] == 0]
                    while uncovered:
                        # Greedily pick best codeword for uncovered
                        best_cw = -1
                        best_cov = 0
                        # Sample from neighbors of uncovered
                        sample_unc = random.sample(uncovered, min(10, len(uncovered)))
                        candidates = set()
                        for u in sample_unc:
                            candidates.update(neighbors[u])
                        
                        for cw in candidates:
                            if cw in cw_set:
                                continue
                            cov = 0
                            for u in neighbors[cw]:
                                if coverage[u] == 0:
                                    cov += 1
                            if cov > best_cov:
                                best_cov = cov
                                best_cw = cw
                            
                        if best_cw == -1:
                            # Fallback
                            u = uncovered[0]
                            for cw in neighbors[u]:
                                if cw not in cw_set:
                                    best_cw = cw
                                    break
                            if best_cw == -1:
                                best_cw = next(iter(neighbors[u]))
                                # If it's in set, we have a problem, but shouldn't happen if we removed enough
                                # Actually if best_cw is in set, it means it's already there, so it covers u?
                                # If coverage[u]==0, no codeword in set covers u.
                                # So best_cw cannot be in set.
                        
                        codewords.append(best_cw)
                        cw_set.add(best_cw)
                        for u in neighbors[best_cw]:
                            coverage[u] += 1
                            if coverage[u] == 1:
                                uncovered.remove(u) # This is slow O(N). Use set for uncovered?
                    
                    # Reset uncovered to set for efficiency? 
                    # The above repair loop is O(N * K * N) which is slow.
                    # Let's use a set for uncovered in the main loop too?
                    # For now, keep it simple. The space is small.
                
                # After perturbation, the solution is valid (repaired) but might be larger.
                # The local search loop will continue to try to shrink it.
        
        try_remove_redundant()
        return codewords

    start_time = time.time()
    time_limit = 115
    best_codewords = None
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 10:
            break
            
        # Allocate time for this restart
        iter_time = min(20, remaining - 5)
        
        cw = greedy_init()
        cw = optimize(cw, iter_time)
        
        if best_codewords is None or len(cw) < len(best_codewords):
            best_codewords = cw
            # Update best if we found a better one
            
        # If we are close to the end, use remaining time to polish the best
        if best_codewords and time.time() - start_time > time_limit - 20:
            final_time = time_limit - (time.time() - start_time) - 2
            if final_time > 5:
                best_codewords = optimize(best_codewords, final_time)
            break

    if best_codewords is None:
        best_codewords = greedy_init()
        
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
