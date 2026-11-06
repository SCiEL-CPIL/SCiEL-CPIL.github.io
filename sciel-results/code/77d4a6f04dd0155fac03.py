"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct a binary code of length 18 with minimum distance 6.
    Uses a greedy approach starting from Reed-Muller code and adding more words.
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = time.time() + 55  # Safety margin before 90s limit
    
    # Start with first-order Reed-Muller code (32 words, distance 8)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Greedy addition: try to add more codewords
    # For each candidate, check if it has distance >= MIN_DIST from all existing words
    def min_distance_to_code(candidate, existing_words):
        """Compute minimum Hamming distance from candidate to any word in existing_words."""
        min_dist = N + 1
        for w in existing_words:
            d = (candidate ^ w).bit_count()
            if d < min_dist:
                min_dist = d
                if min_dist < MIN_DIST:
                    break
        return min_dist
    
    # Try to add random codewords
    rng = random.Random(42)
    added_count = 0
    last_add_time = time.time()
    
    # Phase 1: Try systematic candidates based on patterns
    # Try all words with certain properties
    for weight in range(MIN_DIST, N + 1):
        if time.time() > DEADLINE:
            break
        # Generate some candidates with this weight
        for _ in range(1000):
            if time.time() > DEADLINE:
                break
            # Random candidate with approximately this weight
            positions = list(range(N))
            rng.shuffle(positions)
            candidate = 0
            for pos in positions[:weight]:
                candidate |= (1 << pos)
            
            if min_distance_to_code(candidate, words) >= MIN_DIST:
                words.append(candidate)
                added_count += 1
                last_add_time = time.time()
    
    # Phase 2: Random search for additional words
    attempts = 0
    max_attempts = 500000
    while attempts < max_attempts and time.time() < DEADLINE:
        attempts += 1
        # Generate random candidate
        candidate = rng.getrandbits(N)
        
        if min_distance_to_code(candidate, words) >= MIN_DIST:
            words.append(candidate)
            added_count += 1
            last_add_time = time.time()
    
    return words
# EVOLVE-BLOCK-END
