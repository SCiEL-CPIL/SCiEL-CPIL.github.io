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

def is_valid_word(candidate, words):
    """Check if candidate has distance >= MIN_DIST from all words in words."""
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True

def greedy_add(words, word_set, candidate):
    """Try to add candidate to words if valid. Returns True if added."""
    if candidate in word_set:
        return False
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    words.append(candidate)
    word_set.add(candidate)
    return True

def greedy_random_construction(seed=42, max_iter=1000000, deadline=None, base_words=None):
    """Greedy random construction to add words to a base code."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Generate random word with various weight distributions
        r = rng.random()
        if r < 0.1:
            weight = 0
        elif r < 0.25:
            weight = rng.choice([1, 1, 2, 2, 2])
        elif r < 0.5:
            weight = rng.choice([3, 3, 3, 4, 4, 4, 5])
        elif r < 0.75:
            weight = rng.choice([5, 6, 6, 7, 7, 8])
        elif r < 0.9:
            weight = rng.choice([8, 9, 9, 10, 10])
        else:
            weight = rng.randint(10, N)
        
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if word in word_set:
            continue
        
        if is_valid_word(word, words):
            words.append(word)
            word_set.add(word)
    
    return words

def greedy_with_local_search(seed=42, max_iter=500000, deadline=None, base_words=None):
    """Greedy construction with occasional local search perturbations."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Occasionally do a local search: try flipping bits of an existing word
        if rng.random() < 0.1 and len(words) > 0:
            base_idx = rng.randint(0, len(words) - 1)
            base = words[base_idx]
            # Try flipping 1-3 bits
            n_flips = rng.randint(1, 3)
            flip_positions = rng.sample(range(N), n_flips)
            candidate = base
            for p in flip_positions:
                candidate ^= (1 << p)
            if candidate not in word_set and is_valid_word(candidate, words):
                words.append(candidate)
                word_set.add(candidate)
        else:
            # Standard random generation
            weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9])
            positions = rng.sample(range(N), min(weight, N))
            word = 0
            for p in positions:
                word |= 1 << p
            
            if word in word_set:
                continue
            
            if is_valid_word(word, words):
                words.append(word)
                word_set.add(word)
    
    return words

def systematic_search(seed=42, deadline=None):
    """Systematic search over low-weight words, then medium-weight."""
    rng = random.Random(seed)
    words = []
    word_set = set()
    
    # First, add the all-zero word
    words.append(0)
    word_set.add(0)
    
    # Generate all words of weight 1 through 5
    # This is manageable: C(18,1)+C(18,2)+C(18,3)+C(18,4)+C(18,5) = 18+153+816+3060+8568 = 12615
    from itertools import combinations
    
    for weight in range(1, 6):
        if deadline and time.time() > deadline:
            break
        for positions in combinations(range(N), weight):
            if deadline and time.time() > deadline:
                break
            word = 0
            for p in positions:
                word |= 1 << p
            if word in word_set:
                continue
            if is_valid_word(word, words):
                words.append(word)
                word_set.add(word)
    
    # Then try random higher-weight words
    for i in range(200000):
        if deadline and time.time() > deadline:
            break
        weight = rng.randint(6, 12)
        positions = rng.sample(range(N), weight)
        word = 0
        for p in positions:
            word |= 1 << p
        if word in word_set:
            continue
        if is_valid_word(word, words):
            words.append(word)
            word_set.add(word)
    
    return words

def construct():
    """
    Constructs the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6.
    
    Strategy:
    1. Start with Reed-Muller base (32 words).
    2. Use multiple greedy strategies with different seeds and approaches.
    3. Return the best result within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin from 90s limit
    
    best_words = []
    
    # Try systematic search first (deterministic, may find good low-weight codes)
    if time.time() < deadline:
        words = systematic_search(seed=42, deadline=deadline)
        if len(words) > len(best_words):
            best_words = words
    
    # Try greedy random with multiple seeds
    seeds = [42, 123, 999, 2023, 777, 555, 111, 888, 333, 666, 1234, 5678, 9012, 3456, 7890]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=500000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
    
    # Try greedy with local search
    for seed in [42, 123, 999]:
        if time.time() > deadline:
            break
        current_words = greedy_with_local_search(seed=seed, max_iter=500000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
    
    # If still small, do a long greedy run
    if len(best_words) < 100 and time.time() < deadline:
        current_words = greedy_random_construction(seed=4242, max_iter=2000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
    
    # Final attempt: try to extend the best code found with more random search
    if best_words and time.time() < deadline:
        current_words = list(best_words)
        word_set = set(current_words)
        rng = random.Random(99999)
        for i in range(1000000):
            if time.time() > deadline:
                break
            weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12])
            positions = rng.sample(range(N), min(weight, N))
            word = 0
            for p in positions:
                word |= 1 << p
            if word in word_set:
                continue
            if is_valid_word(word, current_words):
                current_words.append(word)
                word_set.add(word)
        if len(current_words) > len(best_words):
            best_words = current_words
    
    # Ensure we return at least the Reed-Muller code if nothing better was found
    if not best_words:
        best_words = reed_muller_construction()
    
    return best_words

# EVOLVE-BLOCK-END
