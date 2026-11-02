"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # For this specific task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Sphere size: 1 + 7*(3-1) = 15
    # Lower bound: ceil(2187/15) = 146
    # Best known: 186

    # Precompute all words as integers 0..2186
    total = q ** n
    # Convert int to tuple of digits
    def to_tuple(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return tuple(reversed(digits))

    # Precompute neighbors (within Hamming distance <= radius) for each word
    # For radius=1, neighbors are the word itself plus all single-position changes
    # This gives us the "covered" set for each potential codeword
    
    # We'll use a greedy set cover approach with multiple restarts
    # Each codeword covers 1 + n*(q-1) = 15 words
    
    # Precompute coverage sets for all possible codewords
    # coverage[word_idx] = frozenset of all words within distance <= radius
    
    # For efficiency, precompute neighbor offsets
    # For radius 1: word w is covered by codeword c if dist(c,w) <= 1
    # Equivalently, w covers codeword c if dist(c,w) <= 1 (symmetric)
    
    # Let's precompute for each word index, the list of codeword indices that cover it
    # Or better: for each potential codeword, the set of words it covers
    
    # Actually, for greedy set cover, we want:
    # - universe = all 2187 words
    # - for each candidate codeword c, S(c) = {w : dist(c,w) <= radius}
    # - greedily pick c that maximizes |S(c) \ covered|
    
    # Precompute S(c) for all c
    # For radius=1, S(c) has size 1 + 7*2 = 15
    
    # Let's represent words as integers 0..2186
    # Precompute all neighbor sets
    
    # For each word c, compute all words within distance 1
    # dist <= 1 means: same word, or differ in exactly one position by +/-1 mod q
    
    # Efficient computation: for word c, neighbors are:
    # c itself, and for each position i, for each delta in {1,2} (mod 3),
    # change position i by delta
    
    # Let's precompute a list of (position, delta) pairs
    pos_delta_pairs = []
    for pos in range(n):
        for delta in range(1, q):
            pos_delta_pairs.append((pos, delta))
    
    # For each word index, precompute its coverage set as a bitmask or sorted list
    # With 2187 words, we can use a bitset of 2187 bits = 274 bytes per word
    # Total memory: 2187 * 274 ≈ 600KB, fine
    
    # Actually, let's use numpy bitsets for efficiency
    # coverage_mask[i] is a numpy array of bools (length total) indicating which words are covered by codeword i
    
    # But that's 2187 * 2187 = 4.7M booleans = 4.7MB, fine
    
    # Let's precompute
    # For each word c (as integer), compute the set of words it covers
    coverage = []
    for c in range(total):
        c_tuple = to_tuple(c)
        covered = set()
        covered.add(c)
        for pos, delta in pos_delta_pairs:
            new_list = list(c_tuple)
            new_list[pos] = (new_list[pos] + delta) % q
            # Convert back to integer
            new_int = 0
            for i in range(n):
                new_int = new_int * q + new_list[i]
            covered.add(new_int)
        coverage.append(frozenset(covered))
    
    # Greedy set cover with multiple restarts and randomization
    best_code = None
    best_size = total  # worst case
    
    # Multiple restarts with different random orderings
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Randomize the order of candidates
        candidates = list(range(total))
        random.shuffle(candidates)
        
        covered = set()
        code = []
        
        for step in range(total):
            if time.time() > deadline - 3:
                break
            
            # Find the candidate that covers the most uncovered words
            best_gain = -1
            best_idx = -1
            
            for c in candidates:
                if c in covered:
                    continue
                gain = len(coverage[c] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = c
                elif gain == best_gain:
                    # Tie-breaking: random
                    if random.random() < 0.5:
                        best_gain = gain
                        best_idx = c
            
            if best_gain <= 0:
                break
            
            covered.update(coverage[best_idx])
            code.append(best_idx)
            
            # Remove best_idx from candidates to avoid re-considering
            # Actually, we can keep it but skip if already covered... 
            # Better: just skip covered codewords in the loop above
            
            if len(covered) == total:
                break
        
        if len(covered) == total and len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        # Early termination if we hit the known lower bound
        if best_size <= 156:
            break
    
    # Convert back to list of digit lists
    if best_code is None:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    result = []
    for c in best_code:
        result.append(list(to_tuple(c)))
    
    return result
# EVOLVE-BLOCK-END
