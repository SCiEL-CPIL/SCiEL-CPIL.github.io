"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # Start with the Reed-Muller code (32 words, distance 8)
    # First-order Reed-Muller on 16 coordinates, extended by two zero bits
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    
    # Try to extend the code by adding new words
    # We'll use a greedy approach: try adding words one by one,
    # checking that each new word has distance >= 6 from all existing words.
    
    # Generate candidate words to try adding
    # Strategy: try all 2^18 words in order of increasing weight,
    # and for each, check if it can be added to the code.
    # This is expensive, so we'll use a smarter approach.
    
    # Alternative: use a known construction or try random/greedy extension
    
    # Let's try a greedy extension with random candidates
    import random
    
    best_code = rm_words[:]
    
    # Try to add more words using a greedy approach
    # We'll try random words and check if they fit
    rng = random.Random(42)
    
    # Precompute a set for fast lookup
    code_set = set(best_code)
    
    # Try to extend the code
    # For efficiency, we'll batch-check candidates
    
    # Generate candidates: all 18-bit words, but we'll sample them
    # Actually, let's try a more systematic approach:
    # Try all words with weight in a certain range
    
    # For now, let's try a simple greedy with random sampling
    max_iterations = 500000
    for i in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random candidate
        candidate = rng.getrandbits(18)
        
        if candidate in code_set:
            continue
        
        # Check distance to all existing codewords
        valid = True
        for cw in best_code:
            if (candidate ^ cw).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            best_code.append(candidate)
            code_set.add(candidate)
    
    # Also try a systematic search for words with specific weights
    # Words of weight 3, 4, 5, etc. that might fit
    
    # Let's also try to use the structure of the RM code to find extensions
    # The RM(1,4) code has the property that any two codewords differ in exactly 8 positions
    # (for the 16-bit part), so the 18-bit version has distance 8.
    
    # Try adding words that are "far" from the RM code
    # We can try words with certain patterns
    
    # Additional greedy pass with different random seed
    rng2 = random.Random(123)
    for i in range(200000):
        if time.time() > deadline - 3.0:
            break
        
        candidate = rng2.getrandbits(18)
        
        if candidate in code_set:
            continue
        
        valid = True
        for cw in best_code:
            if (candidate ^ cw).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            best_code.append(candidate)
            code_set.add(candidate)
    
    return best_code
# EVOLVE-BLOCK-END
