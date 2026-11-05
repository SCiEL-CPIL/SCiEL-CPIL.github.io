"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 18
MIN_DIST = 6

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def reed_muller_construction():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Every affine Boolean function on F_2^4 is a word: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def to_bit_matrix(words):
    """Convert list of integers to a 2D numpy array of bits."""
    arr = np.array([(w >> i) & 1 for w in words for i in range(N)])
    return arr.reshape(-1, N)

def min_distance_to_set(word, words_arr):
    """Compute minimum Hamming distance from word to any word in words_arr (numpy 2D array)."""
    word_bits = np.array([(word >> i) & 1 for i in range(N)])
    if len(words_arr) == 0:
        return N
    xor = words_arr ^ word_bits
    dists = xor.sum(axis=1)
    return int(dists.min())

def greedy_local_search(words, max_time=50.0, seed=42):
    """Local search: try to add new words by modifying existing ones or random candidates."""
    rng = random.Random(seed)
    words = list(words)
    words_arr = to_bit_matrix(words)
    start_time = time.time()
    
    # Precompute all possible single-bit flips for each word
    # Try adding words systematically
    
    added = 0
    attempts = 0
    max_attempts = 500000
    
    while time.time() - start_time < max_time and attempts < max_attempts:
        attempts += 1
        # Strategy 1: Try random word
        weight = rng.randint(0, 9)
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if min_distance_to_set(word, words_arr) >= MIN_DIST:
            words.append(word)
            words_arr = np.vstack([words_arr, np.array([(word >> i) & 1 for i in range(N)])])
            added += 1
            continue
        
        # Strategy 2: Try flipping 1-2 bits of an existing word
        if len(words) > 0 and rng.random() < 0.7:
            base_idx = rng.randint(0, len(words) - 1)
            base = words[base_idx]
            num_flips = rng.choice([1, 1, 2, 2, 3])
            flip_positions = rng.sample(range(N), num_flips)
            word = base
            for p in flip_positions:
                word ^= (1 << p)
            
            if word not in set(words):
                if min_distance_to_set(word, words_arr) >= MIN_DIST:
                    words.append(word)
                    words_arr = np.vstack([words_arr, np.array([(word >> i) & 1 for i in range(N)])])
                    added += 1
    
    return words

def construct():
    """Construct the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline (15s safety margin from 90s limit)
    
    # Start with the Reed-Muller construction (32 words)
    base_words = reed_muller_construction()
    best_words = list(base_words)
    
    # Try multiple local search attempts with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        result = greedy_local_search(base_words, max_time=remaining_time - 2, seed=seed)
        if len(result) > len(best_words):
            best_words = result
            # Also try to extend the best result found
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                extended = greedy_local_search(best_words, max_time=remaining_time - 2, seed=seed + 100)
                if len(extended) > len(best_words):
                    best_words = extended
    
    return best_words

# EVOLVE-BLOCK-END
