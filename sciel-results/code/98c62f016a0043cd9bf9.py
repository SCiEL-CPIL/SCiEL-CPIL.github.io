"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import heapq
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    total_words = q ** n  # 2187
    
    # Precompute neighbor lists (each word + its 14 neighbors at Hamming distance 1)
    all_neighbors = []
    for w_int in range(total_words):
        digits = []
        temp = w_int
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        neighbors = [w_int]
        for i in range(n):
            orig = digits[i]
            for new_val in range(q):
                if new_val != orig:
                    digits[i] = new_val
                    val = 0
                    for d in digits:
                        val = val * q + d
                    neighbors.append(val)
            digits[i] = orig
        all_neighbors.append(neighbors)
    
    # Also precompute: for each word, which codewords cover it (reverse mapping)
    # We'll build this incrementally
    
    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word
    
    # Phase 1: Greedy set cover
    # Start with no codewords, add the one that covers the most uncovered words
    covered = np.zeros(total_words, dtype=np.bool_)
    code_ints = []
    code_set = set()
    
    # For efficiency, track how many uncovered words each candidate covers
    # Initially all words are uncovered, so gain[c] = len(all_neighbors[c]) = 15 for all c
    # But we need to be smarter: use a heap
    
    # Greedy: repeatedly pick the word that covers the most currently-uncovered words
    remaining = total_words  # number of uncovered words
    
    while remaining > 0 and time.time() < deadline - 30:
        best_gain = -1
        best_word = -1
        
        # Scan all words to find the one with max uncovered coverage
        # This is O(total_words * 15) per iteration, but we can optimize
        for c in range(total_words):
            if c in code_set:
                continue
            gain = 0
            for nb in all_neighbors[c]:
                if not covered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = c
            if gain == 0:
                continue
        
        if best_gain <= 0:
            break
        
        # Add best_word to code
        code_ints.append(best_word)
        code_set.add(best_word)
        for nb in all_neighbors[best_word]:
            if not covered[nb]:
                covered[nb] = True
                remaining -= 1
    
    # Phase 2: Local search - try to remove redundant codewords
    # Rebuild coverage counts
    covered_count = np.zeros(total_words, dtype=np.int32)
    for c_int in code_ints:
        for nb in all_neighbors[c_int]:
            covered_count[nb] += 1
    
    # Try removing each codeword if it doesn't break coverage
    changed = True
    while changed and time.time() < deadline - 15:
        changed = False
        for idx in range(len(code_ints)):
            c_int = code_ints[idx]
            can_remove = True
            for nb in all_neighbors[c_int]:
                if covered_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for nb in all_neighbors[c_int]:
                    covered_count[nb] -= 1
                code_ints.pop(idx)
                code_set.discard(c_int)
                changed = True
                break  # Restart scan since indices changed
    
    # Phase 3: Swap-based local search
    # Try replacing pairs of codewords with one that covers more
    for _ in range(50000):
        if time.time() > deadline - 5:
            break
        
        if len(code_ints) < 2:
            break
        
        # Pick two random codewords to try removing
        i1 = random.randint(0, len(code_ints) - 1)
        i2 = random.randint(0, len(code_ints) - 1)
        if i1 == i2:
            continue
        
        c1 = code_ints[i1]
        c2 = code_ints[i2]
        
        # Temporarily remove both
        for nb in all_neighbors[c1]:
            covered_count[nb] -= 1
        for nb in all_neighbors[c2]:
            covered_count[nb] -= 1
        
        # Find words that could replace them (cover the newly uncovered words)
        # Words that cover at least one newly-uncovered word
        candidates = []
        for nb in all_neighbors[c1]:
            if covered_count[nb] == 0:
                candidates.append(nb)
        for nb in all_neighbors[c2]:
            if covered_count[nb] == 0:
                candidates.append(nb)
        
        if not candidates:
            # No uncovered words, so we can just remove one of them
            # Actually if no uncovered words, we should have been able to remove them in phase 2
            # Roll back
            for nb in all_neighbors[c1]:
                covered_count[nb] += 1
            for nb in all_neighbors[c2]:
                covered_count[nb] += 1
            continue
        
        # Try to find a single word that covers all newly uncovered words
        # or two words that together cover all
        newly_uncovered = set()
        for nb in all_neighbors[c1]:
            if covered_count[nb] == 0:
                newly_uncovered.add(nb)
        for nb in all_neighbors[c2]:
            if covered_count[nb] == 0:
                newly_uncovered.add(nb)
        
        # Try single replacement
        found_single = False
        for c in code_ints:
            if c == c1 or c == c2:
                continue
            covers_all = True
            for nb in newly_uncovered:
                if covered_count[nb] + (1 if nb in all_neighbors[c] else 0) <= 0:
                    # Check if c covers nb
                    pass
            # More efficient: check if c's neighbors cover all newly_uncovered
            c_neighbors = set(all_neighbors[c])
            if newly_uncovered.issubset(c_neighbors):
                # c can replace both c1 and c2
                # Remove c1, c2 and add c (if not already in code)
                if c in code_set:
                    # c is already in code, just remove c1 and c2
                    # But we already decremented covered_count for c1 and c2
                    # c's coverage is still counted, so we're good
                    code_ints.pop(i2 if i2 > i1 else i1)
                    code_ints.pop(i1)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    found_single = True
                    break
                else:
                    # Add c to code
                    for nb in all_neighbors[c]:
                        covered_count[nb] += 1
                    code_ints.pop(i2 if i2 > i1 else i1)
                    code_ints.pop(i1)
                    code_ints.append(c)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    code_set.add(c)
                    found_single = True
                    break
        
        if not found_single:
            # Roll back
            for nb in all_neighbors[c1]:
                covered_count[nb] += 1
            for nb in all_neighbors[c2]:
                covered_count[nb] += 1
    
    # Phase 4: Final cleanup - remove any remaining redundant codewords
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(code_ints)):
            c_int = code_ints[idx]
            can_remove = True
            for nb in all_neighbors[c_int]:
                if covered_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for nb in all_neighbors[c_int]:
                    covered_count[nb] -= 1
                code_ints.pop(idx)
                code_set.discard(c_int)
                changed = True
                break
    
    # Verify coverage and add any missing words
    for w in range(total_words):
        if covered_count[w] == 0 and time.time() < deadline - 1:
            code_ints.append(w)
            code_set.add(w)
            for nb in all_neighbors[w]:
                covered_count[nb] += 1
    
    return [int_to_word(c) for c in code_ints]
# EVOLVE-BLOCK-END
