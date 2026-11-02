"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - time.time()

def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_q3_n7_r1()
    
    return _general_greedy(q, n, radius)

def _solve_q3_n7_r1():
    q, n, radius = 3, 7, 1
    total_words = 3 ** 7  # 2187
    
    # Precompute all words as flat base-3 integers
    # We represent each word as an integer 0..2186
    # word i corresponds to base-3 representation of i padded to 7 digits
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices within Hamming distance 1
    neighbors = [None] * total_words
    for i in range(total_words):
        # Get digits of i in base 3
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % 3)
            x //= 3
        # digits[0] is least significant (position 6 in standard notation)
        # But for Hamming distance it doesn't matter which position is which
        
        nb_set = {i}
        for pos in range(n):
            orig = digits[pos]
            for v in range(3):
                if v == orig:
                    continue
                # Change digit at pos
                new_x = i
                # Subtract old digit contribution
                old_contrib = orig * (3 ** pos)
                new_contrib = v * (3 ** pos)
                new_x = i - old_contrib + new_contrib
                nb_set.add(new_x)
        
        neighbors[i] = list(nb_set)
    
    # Greedy set cover with local search improvement
    # Phase 1: Greedy construction
    uncovered = [True] * total_words
    uncovered_count = total_words
    codewords = []
    
    # For speed, we'll track for each candidate how many uncovered it covers
    # But recomputing each time is simpler and fast enough for 2187 words
    
    while uncovered_count > 0 and _time_left() > 10.0:
        # Find best center: maximizes number of uncovered points covered
        best_center = -1
        max_covered = 0
        
        # Only consider centers that are themselves uncovered or cover uncovered points
        # For efficiency, consider all centers but skip those with 0 uncovered in ball
        for c in range(total_words):
            count = 0
            for nb in neighbors[c]:
                if uncovered[nb]:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_center = c
        
        if best_center == -1 or max_covered == 0:
            break
        
        codewords.append(best_center)
        for nb in neighbors[best_center]:
            if uncovered[nb]:
                uncovered[nb] = False
                uncovered_count -= 1
    
    # Phase 2: Local search - try to remove redundant codewords
    # Then try swaps to reduce size further
    
    # First, remove redundant codewords
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        new_codewords = []
        for i, c in enumerate(codewords):
            # Check if c is redundant: every point in ball(c) is covered by other codewords
            is_redundant = True
            for p in neighbors[c]:
                if uncovered[p]:
                    # p is only covered by c (since it's uncovered by others)
                    # Wait, uncovered[p] means p is NOT covered by any codeword
                    # But we've already covered everything, so uncovered should be all False
                    # Let me reconsider
                    pass
            
            # Better approach: for each point in ball(c), check if it's covered by some other codeword
            for p in neighbors[c]:
                covered_by_other = False
                for j, r in enumerate(codewords):
                    if j == i:
                        continue
                    # Check if p is in ball(r)
                    # We need to check Hamming distance between p and r
                    # Or use precomputed neighbors
                    if p in neighbors[r]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                changed = True
            else:
                new_codewords.append(c)
        
        codewords = new_codewords
    
    # Phase 3: Try to add a codeword and remove two (or more) to reduce total size
    # This is more complex. Let's try a simpler improvement:
    # For each pair of codewords, see if we can replace them with one that covers more
    # Actually, let's try: for each codeword c, try replacing it with a different center c'
    # such that the total number of uncovered points (if we remove c and add c') decreases
    
    # Alternative: try to find a codeword that can be removed if we add a different one
    # i.e., find c and c' such that ball(c) - ball(c') is covered by other codewords
    # and ball(c') adds more coverage than ball(c) loses
    
    # Let's do a simple 1-for-1 swap search
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i, c in enumerate(codewords):
            # Try to find a replacement c' != c that covers all of ball(c) that wasn't covered by others
            # and possibly covers additional points
            # First, find which points in ball(c) are only covered by c
            critical_points = []
            for p in neighbors[c]:
                only_covered_by_c = True
                for j, r in enumerate(codewords):
                    if j == i:
                        continue
                    if p in neighbors[r]:
                        only_covered_by_c = False
                        break
                if only_covered_by_c:
                    critical_points.append(p)
            
            if not critical_points:
                # c is redundant, remove it
                codewords.pop(i)
                improved = True
                break
            
            # Try to find c' that covers all critical_points
            # and has at least as many total uncovered points as c does
            best_c_prime = -1
            best_gain = 0
            current_coverage = 0
            for p in neighbors[c]:
                # Count how many other codewords cover p
                cnt = 0
                for j, r in enumerate(codewords):
                    if j != i and p in neighbors[r]:
                        cnt += 1
                if cnt == 0:
                    current_coverage += 1  # p is only covered by c
            
            # Search for c' among all words (or a subset)
            # To be efficient, only consider c' that covers all critical_points
            # This is expensive. Let's limit the search.
            # Try a random subset of candidates
            candidates = list(range(total_words))
            random.shuffle(candidates)
            for c_prime in candidates[:500]:  # Limit search
                if c_prime == c:
                    continue
                # Check if c' covers all critical points
                covers_all = True
                for p in critical_points:
                    if p not in neighbors[c_prime]:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Calculate gain: how many new points does c' cover that weren't covered before?
                # New points = ball(c') - union of balls of other codewords
                # But we're removing c, so we need to check what c' adds vs what c provided
                # Simpler: after removing c and adding c', is everything still covered?
                # And did we reduce the count? (We're doing 1-for-1 so count stays same)
                # The goal is to enable future removals. So we want c' to be "better" positioned.
                
                # Let's just check if the swap maintains coverage
                # Temporarily remove c, add c', check if all is covered
                # This is expensive. Skip for now.
                pass
        
        # If no improvement found in this pass, break
        if not improved:
            # Try a different strategy: remove one codeword and see if we can cover its unique points
            # by adjusting others? Too complex. Break.
            break
    
    # Convert indices back to words
    result = []
    for c in codewords:
        digits = []
        x = c
        for _ in range(n):
            digits.append(x % 3)
            x //= 3
        # Reverse to get most significant first? 
        # Actually, the order doesn't matter for the problem as long as consistent
        # But let's reverse to match standard notation
        result.append(digits[::-1])
    
    return result

def _general_greedy(q, n, radius):
    total_words = q ** n
    if total_words > 100000:
        # Too large, use simple heuristic
        return [[0] * n]
    
    # Generate all words
    all_words = []
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= q
        all_words.append(digits[::-1])
    
    # Precompute neighbors
    neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        neighbors[i].add(i)
        if radius == 1:
            for pos in range(n):
                for v in range(q):
                    if v == all_words[i][pos]:
                        continue
                    new_word = all_words[i][:]
                    new_word[pos] = v
                    # Find index of new_word
                    idx = 0
                    for val in new_word:
                        idx = idx * q + val
                    neighbors[i].add(idx)
    
    uncovered = [True] * total_words
    uncovered_count = total_words
    codewords = []
    
    while uncovered_count > 0 and _time_left() > 5.0:
        best_center = -1
        max_covered = 0
        for c in range(total_words):
            count = sum(1 for nb in neighbors[c] if uncovered[nb])
            if count > max_covered:
                max_covered = count
                best_center = c
        
        if best_center == -1:
            break
        
        codewords.append(best_center)
        for nb in neighbors[best_center]:
            if uncovered[nb]:
                uncovered[nb] = False
                uncovered_count -= 1
    
    return [all_words[c] for c in codewords]

# EVOLVE-BLOCK-END
