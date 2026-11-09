import time
import random
import itertools
import numpy as np
from collections import defaultdict

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _popcount(x):
    return bin(x).count("1")

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _reed_muller_extended():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _generate_golay_code():
    """
    Generate the [24, 12, 8] Golay code.
    We use a known generator matrix in systematic form.
    """
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    codewords = []
    n_rows = 12
    n_cols = 24
    for mask in range(1 << n_rows):
        word = 0
        for i in range(n_rows):
            if (mask >> i) & 1:
                for j in range(n_cols):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords.append(word)
    return codewords

def _shorten_golay_to_18():
    """
    Shorten the Golay code by 6 positions to get a code of length 18.
    We look for 6 positions where we can force zeros to maximize the number of remaining codewords.
    The shortened code will have minimum distance at least 8 (since original is 8 and we remove 6 bits, 
    actually shortening reduces distance by at most the number of shortened positions if they were 1, 
    but here we select codewords that are 0 in those positions, so the distance between any two shortened 
    codewords is the same as in the original, which is >= 8. 
    Wait, if we shorten by setting 6 coordinates to 0, the new code is a subcode. 
    The distance between any two words in the subcode is the same as in the original code. 
    So the minimum distance remains 8.
    The length is 24 - 6 = 18.
    The size is the number of codewords with 0 in those 6 positions.
    """
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    golay_words = _generate_golay_code()
    
    best_shortened = []
    best_count = 0
    
    # Try all combinations of 6 columns to shorten
    # There are C(24, 6) = 134596 combinations. 
    # For each, we need to count how many Golay words have 0 in those positions.
    # This is fast enough.
    
    for cols in itertools.combinations(range(24), 6):
        count = 0
        for w in golay_words:
            ok = True
            for c in cols:
                if (w >> c) & 1:
                    ok = False
                    break
            if ok:
                count += 1
        if count > best_count:
            best_count = count
            # Reconstruct the shortened code
            remaining_cols = [c for c in range(24) if c not in cols]
            shortened = []
            for w in golay_words:
                if all(((w >> c) & 1) == 0 for c in cols):
                    new_word = 0
                    for i, c in enumerate(remaining_cols):
                        if (w >> c) & 1:
                            new_word |= (1 << i)
                    shortened.append(new_word)
            best_shortened = shortened
    
    return best_shortened

def _construct_steiner_system(n=18, k=6, t=3):
    """
    Construct a Steiner System S(3, 6, 18) or a maximal partial Steiner system.
    This is used to generate a constant weight code of weight 6 (or 12, etc.)
    Actually, we want a code of length 18, distance 6.
    A constant weight code A(18, w, 6) can be derived from a Steiner system S(3, w, 18) if it exists.
    However, S(3, 6, 18) does not exist (necessary conditions: 18 choose 3) / (6 choose 3) = 816/20 = 40.8, not integer).
    So we look for a maximal partial Steiner system.
    
    Alternatively, we can use the fact that the optimal code A_2(18,6) is known to be 256 or 512?
    Wait, the Reed-Muller code RM(1,4) extended has length 16, size 32.
    The Golay shortened code has length 18.
    If we shorten the [24,12,8] Golay code by 6 bits, we get a [18, k, 8] code.
    The maximum size of a [18, k, 8] code is bounded.
    If we can find a shortening that gives a large k, that's good.
    
    Let's try a different structural approach:
    1. Start with the zero vector.
    2. Use a greedy algorithm with a strong heuristic: 
       - Prefer vectors of weight 9 (middle weight).
       - Use a bitset representation for fast conflict checking.
       - Use backtracking to fill gaps.
    """
    pass

def _greedy_with_backtracking(N=18, min_dist=6, time_limit=50.0, seed=42):
    """
    Greedy algorithm with backtracking and symmetry breaking.
    """
    rng = random.Random(seed)
    
    # Precompute all possible words? No, 2^18 is 262144. That's manageable.
    # We can precompute the weight of each word.
    # We can also precompute the Hamming distance to the zero vector (which is the weight).
    
    # Strategy:
    # 1. Fix the first word to be 0.
    # 2. Generate a list of candidate words, sorted by some heuristic (e.g., weight).
    # 3. Greedily add words that are at distance >= min_dist from all existing words.
    # 4. If stuck, backtrack.
    
    # To make it faster, we use a bitset for "forbidden" words.
    # A word w is forbidden if there exists a word c in the code such that dist(w, c) < min_dist.
    # This is equivalent to the union of Hamming balls of radius min_dist-1 around each codeword.
    
    # Since N=18 and min_dist=6, the radius is 5.
    # The size of a Hamming ball of radius 5 in 18 dimensions is sum_{i=0}^5 C(18, i) = 1 + 18 + 153 + 816 + 3060 + 8568 = 12616.
    # This is small enough to precompute for each codeword? 
    # If we have 512 codewords, 512 * 12616 ~ 6.4 million operations. That's fine.
    
    # However, we don't need to precompute for all. We can just check candidates.
    
    # Let's use a simpler greedy approach with a good ordering.
    # Order candidates by weight, then by value.
    
    candidates = list(range(1 << N))
    # Sort by weight, then by value
    candidates.sort(key=lambda x: (bin(x).count('1'), x))
    
    code = [0]
    forbidden = set([0])
    
    # Precompute distances to 0? No, we need distances to all codewords.
    
    # To speed up, we can maintain a list of codewords and check each candidate against them.
    # With 512 codewords, checking 1 candidate takes 512 XORs and popcounts.
    # 262144 candidates * 512 checks = 134 million operations. Might be slow in Python.
    
    # Optimization: Use numpy for vectorized distance checks?
    # Or, use a bitset of forbidden words.
    
    # Let's try to build the forbidden set incrementally.
    # When we add a word w, we add all words at distance < min_dist to forbidden.
    
    # Precompute the neighbors of each word? 
    # We can precompute the list of words at distance <= 5 from 0. 
    # Then, for any word w, the neighbors are w XOR neighbors_of_0.
    
    neighbors_of_0 = []
    for w in range(1 << N):
        if bin(w).count('1') <= min_dist - 1:
            neighbors_of_0.append(w)
    
    # This list has ~12616 elements.
    
    def add_word(w):
        nonlocal forbidden
        for d in neighbors_of_0:
            forbidden.add(w ^ d)
    
    add_word(0)
    
    start_time = time.time()
    
    # Iterate through candidates
    for w in candidates:
        if time.time() - start_time > time_limit:
            break
        if w in forbidden:
            continue
        # Check if w is actually at distance >= min_dist from all codewords
        # Since we maintain the forbidden set correctly, w not in forbidden implies it's safe.
        # But wait, the forbidden set is the union of balls. 
        # If w is not in the union, then dist(w, c) >= min_dist for all c in code.
        # So we can just check `if w not in forbidden`.
        
        code.append(w)
        add_word(w)
        
        # If we reach a large size, we can stop or continue
        if len(code) >= 512:
            break
            
    return code

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()

    # 1. Try Golay Shortening
    # This gives a linear code of length 18, distance 8.
    # The size is likely around 128-256.
    golay_shortened = _shorten_golay_to_18()
    
    if not _check_min_distance(golay_shortened, MIN_DIST):
        raise ValueError("Golay shortening failed distance check")
    
    best = list(golay_shortened)
    
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    
    if time_remaining > 5.0:
        # 2. Greedy with Backtracking starting from the Golay shortened code
        # We need to adapt the greedy function to start from an existing code.
        
        # Let's modify the greedy function to accept an initial code.
        
        def greedy_from_initial(initial_code, N=18, min_dist=6, time_limit=50.0, seed=42):
            rng = random.Random(seed)
            candidates = list(range(1 << N))
            candidates.sort(key=lambda x: (bin(x).count('1'), x))
            
            code = list(initial_code)
            forbidden = set()
            
            neighbors_of_0 = []
            for w in range(1 << N):
                if bin(w).count('1') <= min_dist - 1:
                    neighbors_of_0.append(w)
            
            def add_word(w):
                for d in neighbors_of_0:
                    forbidden.add(w ^ d)
            
            for w in code:
                add_word(w)
                
            start_time = time.time()
            
            for w in candidates:
                if time.time() - start_time > time_limit:
                    break
                if w in forbidden:
                    continue
                code.append(w)
                add_word(w)
                
                if len(code) >= 512:
                    break
                    
            return code

        greedy_result = greedy_from_initial(best, N, MIN_DIST, time_limit=time_remaining - 2.0, seed=42)
        
        if len(greedy_result) > len(best) and _check_min_distance(greedy_result, MIN_DIST):
            best = greedy_result
            
    return best

if __name__ == "__main__":
    words = construct()
    print(f"Code size: {len(words)}")
    print(f"Min distance: {min(_hamming(a, b) for i, a in enumerate(words) for b in words[i+1:])}")
