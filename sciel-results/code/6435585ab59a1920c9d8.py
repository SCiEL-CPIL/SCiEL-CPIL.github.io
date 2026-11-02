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
    """
    if n == 0:
        return [[]]
    
    # Total number of words in the space
    total_words = q ** n
    
    # If the space is small enough, we can potentially find optimal or near-optimal
    # For our problem: q=3, n=8, radius=1, total_words = 6561
    
    # Precompute all words as integers (base-q representation)
    # Each word is an integer from 0 to total_words - 1
    # Convert between integer and list representation
    
    def int_to_word(val, n, q):
        """Convert integer to word (list of digits)"""
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        """Convert word (list of digits) to integer"""
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbors: for each word, list of words at Hamming distance <= radius
    # For radius=1, neighbors are the word itself plus words differing in exactly one position
    
    # For efficiency, precompute all words and their neighbors
    all_words = list(range(total_words))
    
    # Precompute neighbors for each word (distance <= radius)
    # This is memory intensive for large spaces, but for 6561 words it's fine
    neighbors = {}
    for w in all_words:
        word = int_to_word(w, n, q)
        neigh = [w]  # Include the word itself
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        if radius >= 2:
            # Add words at distance 2
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
    
    # Greedy initialization: build a feasible covering
    # Start with no codewords, repeatedly add the codeword that covers the most uncovered words
    
    # Track which words are covered
    covered = [False] * total_words
    codewords = []
    uncovered_count = total_words
    
    # Precompute for each potential codeword, which words it covers
    # We need to find the codeword that covers the most uncovered words
    
    # For efficiency, use a priority queue approach
    # But for 6561 words, a simpler approach works
    
    # Greedy: repeatedly pick the word that covers the most currently uncovered words
    remaining = set(all_words)
    
    while uncovered_count > 0:
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
            if new_cov == uncovered_count:  # Can't do better
                break
        if best_word == -1:
            break
        codewords.append(best_word)
        remaining.discard(best_word)
        for u in neighbors[best_word]:
            if not covered[u]:
                covered[u] = True
                uncovered_count -= 1
    
    # Now we have a feasible covering. Try to optimize via local search.
    # Strategy: try to remove or swap codewords to reduce the total count while maintaining coverage.
    
    # Count how many times each word is covered (by how many codewords)
    coverage_count = [0] * total_words
    for cw in codewords:
        for u in neighbors[cw]:
            coverage_count[u] += 1
    
    # Local search: try to remove codewords that are redundant
    # A codeword is redundant if removing it doesn't uncover any word
    
    # First, try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords) - 1, -1, -1):
            cw = codewords[i]
            # Check if removing this codeword leaves all words covered
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
                break  # Restart after removal
    
    # Now try swaps: replace one codeword with another to potentially allow further removals
    # This is more complex. Let's try a simpler approach: randomized local search
    
    # Try to improve by swapping codewords
    # For each codeword, try replacing it with a random other word and see if we can then remove others
    
    # Time limit: 120 seconds total, leave some for final conversion
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Local search with swaps
    # Represent codewords as a set for fast lookup
    cw_set = set(codewords)
    
    # Try iterative improvement
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Try to remove a redundant codeword
        if len(codewords) > 0:
            # Pick a random codeword to try removing
            idx = random.randint(0, len(codewords) - 1)
            cw = codewords[idx]
            can_remove = True
            for u in neighbors[cw]:
                if coverage_count[u] == 1:
                    can_remove = False
                    break
            if can_remove:
                for u in neighbors[cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(cw)
                continue
        
        # Try a swap: replace one codeword with another
        if len(codewords) > 0:
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Pick a random new codeword
            new_cw = random.randint(0, total_words - 1)
            if new_cw == old_cw or new_cw in cw_set:
                continue
            
            # Check if the swap is valid: after removing old_cw and adding new_cw, all words must be covered
            # First, check what would be uncovered if we remove old_cw
            would_uncover = []
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                # Old codeword is redundant, just remove it
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Check if new_cw covers all words that would be uncovered
            new_covers_all = True
            for u in would_uncover:
                if u not in neighbors[new_cw]:
                    new_covers_all = False
                    break
            
            if new_covers_all:
                # Perform the swap
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = new_cw
                cw_set.discard(old_cw)
                cw_set.add(new_cw)
    
    # Final cleanup: remove any remaining redundant codewords
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
                break
    
    # Convert back to list of lists
    result = [int_to_word(cw, n, q) for cw in codewords]
    
    # Verify coverage (optional, for safety)
    # covered_check = [False] * total_words
    # for cw in codewords:
    #     for u in neighbors[cw]:
    #         covered_check[u] = True
    # assert all(covered_check), "Coverage verification failed"
    
    return result
# EVOLVE-BLOCK-END
