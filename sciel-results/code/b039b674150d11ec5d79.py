"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Generate the 32-word Reed-Muller seed: RM(1,4) extended to 17 bits
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                if bit:
                    word |= (1 << point)
            seed_words.append(word)
    
    # Vectorized bit extraction
    def words_to_bits(words):
        n_words = len(words)
        bits = np.zeros((n_words, N), dtype=np.uint8)
        for i, w in enumerate(words):
            for j in range(N):
                if w & (1 << j):
                    bits[i, j] = 1
        return bits
    
    def int_to_bits(w):
        bits = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if w & (1 << j):
                bits[j] = 1
        return bits
    
    def min_dist_to_code(candidate_int, bits):
        cand_bits = int_to_bits(candidate_int)
        xor = bits ^ cand_bits
        dists = xor.sum(axis=1)
        return int(np.min(dists))
    
    def is_valid(candidate_int, bits):
        return min_dist_to_code(candidate_int, bits) >= MIN_DIST
    
    # Try to build a large code using greedy + local search
    def build_code(seed, time_limit):
        current_words = seed[:]
        current_bits = words_to_bits(current_words)
        
        best_words = current_words[:]
        best_size = len(current_words)
        
        attempts_no_add = 0
        max_no_add = 200
        
        while time.time() < time_limit:
            # Try to add a random word
            candidate = random.getrandbits(N)
            if is_valid(candidate, current_bits):
                current_words.append(candidate)
                current_bits = np.vstack([current_bits, int_to_bits(candidate)])
                
                if len(current_words) > best_size:
                    best_size = len(current_words)
                    best_words = current_words[:]
                
                attempts_no_add = 0
            else:
                attempts_no_add += 1
                
                # If stuck, try a swap or mutation
                if attempts_no_add > max_no_add:
                    idx = random.randint(0, len(current_words) - 1)
                    old_word = current_words[idx]
                    
                    # Try different mutation strategies
                    if random.random() < 0.4:
                        candidate = random.getrandbits(N)
                    elif random.random() < 0.7:
                        # Flip 1-2 bits
                        flips = random.randint(1, 2)
                        mask = 0
                        for _ in range(flips):
                            pos = random.randint(0, N - 1)
                            mask |= (1 << pos)
                        candidate = old_word ^ mask
                    else:
                        # Flip 3-4 bits
                        flips = random.randint(3, 4)
                        mask = 0
                        for _ in range(flips):
                            pos = random.randint(0, N - 1)
                            mask |= (1 << pos)
                        candidate = old_word ^ mask
                    
                    cand_bits = int_to_bits(candidate)
                    xor = current_bits ^ cand_bits
                    dists = xor.sum(axis=1)
                    bad_indices = np.where(dists < MIN_DIST)[0]
                    
                    if len(bad_indices) == 0:
                        # Can add
                        current_words.append(candidate)
                        current_bits = np.vstack([current_bits, cand_bits])
                        if len(current_words) > best_size:
                            best_size = len(current_words)
                            best_words = current_words[:]
                        attempts_no_add = 0
                    elif len(bad_indices) == 1 and bad_indices[0] == idx:
                        # Can swap
                        current_words[idx] = candidate
                        current_bits[idx] = cand_bits
                        attempts_no_add = 0
                    else:
                        # Try to find a word that's far from all bad ones
                        # Skip for now
                        pass
                    
                    attempts_no_add = 0  # Reset to allow more swaps
        
        return best_words, best_size
    
    # Try multiple random seeds/restarts to find the best code
    best_overall = seed_words[:]
    best_size_overall = len(seed_words)
    
    # First, try with the original seed
    words, size = build_code(seed_words, deadline - 5.0)
    if size > best_size_overall:
        best_size_overall = size
        best_overall = words
    
    # Try with perturbed seeds
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Generate a few alternative starting points
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Create a perturbed seed by adding some random words to the RM seed
            perturbed_seed = seed_words[:]
            for _ in range(5):
                w = random.getrandbits(N)
                # Check if it's valid with respect to current perturbed_seed
                bits = words_to_bits(perturbed_seed)
                if is_valid(w, bits):
                    perturbed_seed.append(w)
            
            time_left = deadline - 5.0 - time.time()
            if time_left > 2.0:
                words, size = build_code(perturbed_seed, deadline - 5.0)
                if size > best_size_overall:
                    best_size_overall = size
                    best_overall = words
    
    return best_overall
# EVOLVE-BLOCK-END
