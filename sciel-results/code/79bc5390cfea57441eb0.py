"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

MINIMUM_DISTANCE = 6
N = 18
TOTAL_WORDS = 1 << N  # 262144

def _seed_words():
    # First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # Every affine Boolean function on F_2^4 is a word: 32 words, distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _min_distance_to_code(word, code_np):
    # Bit-parallel minimum Hamming distance to any codeword using NumPy.
    if len(code_np) == 0:
        return MINIMUM_DISTANCE
    dists = np.bit_count(code_np ^ word)
    return int(dists.min())

def _try_add_word(word, code, code_np):
    if _min_distance_to_code(word, code_np) >= MINIMUM_DISTANCE:
        code.append(word)
        # Rebuild numpy array
        code_np = np.array(code, dtype=np.uint32)
        return True, code_np
    return False, code_np

def _greedy_construct(deadline):
    code = _seed_words()
    code_np = np.array(code, dtype=np.uint32)
    
    # We scan all candidates from 0 to 2**N - 1.
    # To be more effective, we can try a few different orders if time permits,
    # but a single full scan is a good baseline.
    # Since the seed is fixed, the result is deterministic.
    
    # Optimization: 
    # Instead of rebuilding the numpy array every time we add a word (which is O(M)),
    # we can pre-allocate a larger array and keep track of the current length.
    # However, np.array(code) is fast enough for M < 1000.
    # Let's stick to the simple approach first.
    
    # To improve the packing density, simple sequential scanning (0, 1, 2...) might be suboptimal.
    # A common heuristic is to scan in an order that maximizes the "distance" to the existing code,
    # but that requires computing distances for all candidates at each step, which is expensive.
    
    # Alternative: Randomized greedy with fixed seed?
    # The prompt says "deterministic is safer". But it also says "largest possible".
    # If we use a fixed seed for shuffling, it is deterministic.
    # Let's try a shuffled order with a fixed seed to see if it finds a better packing.
    # Actually, the current score is 6.78 (32 * 0.212?). No, score is value/seed.
    # Wait, the score is the number of codewords? 
    # "Score: number of codewords, maximized."
    # "The evaluator measures the raw value (higher is better). The search score is value/seed"
    # So if the seed program returns 32 words, and the score is 1.0, then the raw value is 32?
    # But the current program has a score of 6.7812.
    # 32 * 6.7812 = 217? That seems high.
    # Or is the "seed" in "value/seed" a different seed?
    # "1.0 equals the seed program". So if the seed program returns 32 words, the score is 1.0.
    # The current program has a score of 6.7812.
    # This implies the current program returns 32 * 6.7812 = 217 words?
    # Let's check the code. The code does a greedy scan.
    # If it finds 217 words, that's a significant improvement over 32.
    
    # Let's try to improve the greedy scan.
    # One issue with sequential scan is that it might get stuck in a local optimum.
    # A better approach might be to use a "backtracking" or "repair" step, but that's complex.
    
    # Let's try to speed up the inner loop further.
    # We can batch the candidates?
    # No, we need to check each candidate against the *current* code.
    
    # Let's try a different candidate order: 
    # Sort candidates by their weight (number of 1s) or something?
    # Or just use a fixed random permutation?
    
    # Let's try a fixed random permutation of candidates.
    rng = np.random.default_rng(42)
    candidates = rng.permutation(TOTAL_WORDS)
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        candidate = int(candidate)
        added, code_np = _try_add_word(candidate, code, code_np)
        # If added, code_np is updated.
        
    return code

def construct():
    # 90s wall-clock limit, keep safety margin.
    deadline = time.time() + 85.0
    
    result = _greedy_construct(deadline)
    
    return result
# EVOLVE-BLOCK-END
