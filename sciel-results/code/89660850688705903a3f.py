"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 17
MIN_DISTANCE = 6

def construct():
    """
    Construct a binary code of length 17 with minimum distance 6.
    Uses a greedy randomized approach with multiple restarts to maximize codeword count.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    best_code = []
    
    # Precompute bit masks for quick distance checks
    # We'll use integer representation for speed
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    def can_add(word, code_set):
        """Check if word can be added to code_set maintaining min distance."""
        for existing in code_set:
            if hamming_dist(word, existing) < MIN_DISTANCE:
                return False
        return True
    
    def greedy_fill(code_list, random_order):
        """
        Greedily add words from random_order to code_list if they fit.
        Returns the extended code list.
        """
        code_set = set(code_list)
        current_code = list(code_list)
        
        for word in random_order:
            if word in code_set:
                continue
            if can_add(word, code_set):
                code_set.add(word)
                current_code.append(word)
        
        return current_code
    
    # Strategy: Start with random seeds and greedily fill
    # Try multiple restarts with different random orderings
    
    # First, generate all possible words (2^17 = 131072)
    all_words = list(range(2**N))
    
    # Try several different strategies:
    # 1. Random order
    # 2. Weight-ordered (low weight first, then high)
    # 3. Mixed approaches
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Start with all-zero word (weight 0)
            seed = [0]
        elif restart == 1:
            # Start with all-ones word
            seed = [(2**N) - 1]
        else:
            # Random seed
            seed = [random.randint(0, 2**N - 1)]
        
        # Create a shuffled list of all words, excluding seed
        candidates = [w for w in all_words if w not in seed]
        random.shuffle(candidates)
        
        # Greedy fill
        code = greedy_fill(seed, candidates)
        
        if len(code) > len(best_code):
            best_code = code
        
        # If we've found a large code, we can stop early
        if len(best_code) >= 100:
            break
    
    # Additional local search: try to improve by swapping
    # For now, return the best found
    return best_code

# EVOLVE-BLOCK-END
