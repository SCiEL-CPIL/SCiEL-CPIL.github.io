# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search starting from a greedy feasible covering.
    Optimizes by iteratively swapping one codeword for another to minimize the count.
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

    # Precompute neighbors for radius=1 (since the task is radius 1 usually, but let's be generic if radius is passed)
    # However, the task specifies radius. For q=3, n=8, radius=1.
    # If radius > 1, the neighbor list is huge. But the task contract implies specific q, n, radius.
    # For K_3(8,1), radius is 1.
    
    # We will assume radius=1 for optimization if possible, but handle general radius.
    # Given the constraints (120s), and q=3, n=8, total_words=6561.
    # If radius=1, neighbors are 1 + 8*2 = 17.
    # If radius=2, neighbors are 1 + 16 + 28*4 = 129.
    # We'll generate neighbors on the fly or precompute if small enough.
    
    # Precomputing all neighbors for 6561 words is feasible.
    neighbors = [None] * total_words
    for w in range(total_words):
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
    
    # Greedy Initialization
    covered = [False] * total_words
    codewords = []
    uncovered_count = total_words
    remaining = set(range(total_words))
    
    # Shuffle remaining to ensure randomization in greedy choice
    # Actually, greedy picks the one covering most *new* words.
    # To speed up, we can just iterate.
    
    while uncovered_count > 0:
        best_word = -1
        best_new_coverage = -1
        # Random tie-breaking
        # We can sample a few candidates if remaining is large, but 6561 is small.
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
        remaining.discard(best_word)
        for u in neighbors[best_word]:
            if not covered[u]:
                covered[u] = True
                uncovered_count -= 1
                
    # Maintain coverage counts
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
                    cw_set.discard(cw)
                    changed = True
                    removed_any = True
                    break
        return removed_any

    # Initial cleanup
    try_remove_redundant()
    
    start_time = time.time()
    time_limit = 115
    
    # Local Search
    # Strategy:
    # 1. Try to remove redundant codewords.
    # 2. If none, pick a codeword with minimal unique coverage (|U|).
    # 3. Try to find a replacement word that covers U.
    # 4. If found, swap.
    # 5. If stuck, try a more random selection or 2-swap (complex).
    
    no_improvement_count = 0
    
    while time.time() - start_time < time_limit:
        if try_remove_redundant():
            no_improvement_count = 0
            continue
            
        if len(codewords) == 0:
            break
            
        # Find codewords and their unique coverage
        # We want to pick a candidate to swap out.
        # Heuristic: Pick the one with smallest |U|.
        # If |U| == 0, it's redundant (handled above).
        
        candidates_to_swap = []
        for i, cw in enumerate(codewords):
            unique_words = []
            for u in neighbors[cw]:
                if coverage_count[u] == 1:
                    unique_words.append(u)
            if len(unique_words) > 0:
                candidates_to_swap.append((len(unique_words), i, unique_words))
                
        if not candidates_to_swap:
            # Should be handled by try_remove_redundant, but if all have 0 unique, they are redundant.
            # If we are here, it means try_remove_redundant failed to remove?
            # This can happen if coverage_count logic is slightly off or if we just finished a loop.
            # Force a full check?
            break
            
        # Sort by size of unique coverage
        candidates_to_swap.sort(key=lambda x: x[0])
        
        # Try the top few candidates (smallest unique coverage first)
        swap_done = False
        # Try up to 5 best candidates to increase success rate
        for _, idx, unique_words in candidates_to_swap[:5]:
            old_cw = codewords[idx]
            
            # Find a new_cw that covers all words in unique_words
            # new_cw must be within distance 1 of all u in unique_words
            # This means new_cw must be in the intersection of Ball(u, 1) for all u in unique_words
            
            # Start with candidates from the first word
            first_u = unique_words[0]
            possible_new_cws = set(neighbors[first_u])
            
            # Intersect with neighbors of other unique words
            for u in unique_words[1:]:
                if not possible_new_cws:
                    break
                possible_new_cws &= set(neighbors[u])
                
            if not possible_new_cws:
                continue
                
            # Filter out old_cw and existing codewords
            # We prefer a new_cw that is not already in cw_set to avoid duplicates?
            # Actually, if new_cw is in cw_set, we can't add it. We must pick a word NOT in cw_set.
            # Wait, if new_cw is in cw_set, we are effectively doing nothing or swapping with itself?
            # If new_cw == old_cw, no change.
            # If new_cw is another existing codeword, we can't have duplicates.
            # So we need new_cw not in cw_set.
            
            valid_new_cws = [cw for cw in possible_new_cws if cw not in cw_set]
            
            if not valid_new_cws:
                continue
                
            # Pick one randomly to avoid bias
            new_cw = random.choice(valid_new_cws)
            
            # Perform swap
            # Update coverage counts
            for u in neighbors[old_cw]:
                coverage_count[u] -= 1
            for u in neighbors[new_cw]:
                coverage_count[u] += 1
                
            # Update codewords list
            codewords[idx] = new_cw
            cw_set.discard(old_cw)
            cw_set.add(new_cw)
            
            swap_done = True
            no_improvement_count = 0
            break
            
        if not swap_done:
            no_improvement_count += 1
            # If stuck for a long time, we might need a perturbation.
            # For now, just keep looping, maybe randomizing the order of candidates more.
            # The sorting makes it deterministic. Let's add some randomness.
            # If no_improvement_count is high, shuffle the candidates before sorting?
            # Or just accept that we might be in a local minimum.
            # Given the small search space, 115s is a lot of iterations.
            # We can also try a "2-swap" if we get stuck?
            # Let's stick to 1-swap for now, but randomize the selection of which candidate to try if the top ones fail.
            
            if no_improvement_count > 1000:
                 # Randomly pick a candidate from the full list, not just the top 5
                 if candidates_to_swap:
                     _, idx, unique_words = random.choice(candidates_to_swap)
                     old_cw = codewords[idx]
                     first_u = unique_words[0]
                     possible_new_cws = set(neighbors[first_u])
                     for u in unique_words[1:]:
                         if not possible_new_cws:
                             break
                         possible_new_cws &= set(neighbors[u])
                         
                     if possible_new_cws:
                         valid_new_cws = [cw for cw in possible_new_cws if cw not in cw_set]
                         if valid_new_cws:
                             new_cw = random.choice(valid_new_cws)
                             for u in neighbors[old_cw]:
                                 coverage_count[u] -= 1
                             for u in neighbors[new_cw]:
                                 coverage_count[u] += 1
                             codewords[idx] = new_cw
                             cw_set.discard(old_cw)
                             cw_set.add(new_cw)
                             swap_done = True
                             no_improvement_count = 0

    try_remove_redundant()
    
    result = [int_to_word(cw, n, q) for cw in codewords]
    return result
# EVOLVE-BLOCK-END
