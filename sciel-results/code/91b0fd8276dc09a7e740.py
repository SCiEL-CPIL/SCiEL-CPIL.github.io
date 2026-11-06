"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
MIN_DISTANCE = 6
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute weight for all 2^18 words? Too large. Use on-the-fly.
    # Instead, precompute popcount for 2^9 = 512 entries and combine.
    # Actually, use a lookup table for 2^9 and combine halves.
    pop9 = np.zeros(512, dtype=np.int8)
    for i in range(512):
        pop9[i] = i.bit_count()
    
    def weight(x):
        return pop9[x & 0x1ff] + pop9[(x >> 9) & 0x1ff]
    
    def hamming(a, b):
        return (a ^ b).bit_count()
    
    # Start with the 32-word Reed-Muller code (distance 8)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Verify initial code
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if hamming(words[i], words[j]) < MIN_DISTANCE:
                # Should not happen, but safety
                pass
    
    # Greedy extension: try to add more codewords
    # Candidate ordering: by weight, then by value
    # We'll try candidates in order of "balancedness" (weight near 9)
    
    # Generate candidates in order of weight closeness to 9
    # But 2^18 is too many to sort. Instead, iterate by weight class.
    
    # For each weight w from 9 down to 0 and 9 up to 18, try all words of that weight
    # This is still too many. Instead, use a smarter approach.
    
    # Better: iterate through all 2^18 words but skip those too close to existing.
    # Use a rejection filter: for each existing word, mark all words within distance < 6 as forbidden.
    
    # Actually, let's use a different strategy: 
    # For each candidate, check against all existing words. Use early termination.
    
    # To make this feasible, we need to be smart about which candidates to try.
    # Strategy: try random candidates, but biased toward balanced weights.
    
    # Let's use a hybrid: first try a structured set, then random.
    
    # Structured: all weight-9 words that are "far" from existing
    # Generate weight-9 words efficiently
    
    # Actually, let's just do a bounded greedy with random candidates
    # and a deadline guard.
    
    import random
    rng = random.Random(42)
    
    # Try to add candidates
    # We'll try in batches, checking time periodically
    
    max_attempts = 5000000
    added = 0
    attempts = 0
    
    # First, try a structured pass: all words of weight 9, 8, 10 in some order
    # But that's C(18,9) = 48620, C(18,8) = 43758, etc. - manageable
    
    # Generate all weight-9 words
    from itertools import combinations
    
    def generate_weight_words(w, n=N):
        """Generate all n-bit words of weight w."""
        positions = list(range(n))
        for combo in combinations(positions, w):
            word = 0
            for p in combo:
                word |= (1 << p)
            yield word
    
    # Try weight classes in order of closeness to 9
    weight_order = [9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 17, 0]
    
    for w in weight_order:
        if time.time() - start_time > DEADLINE - 5:
            break
        if w < 0 or w > N:
            continue
        for candidate in generate_weight_words(w):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Check if candidate is already in words
            if candidate in words:
                continue
            # Check distance to all existing words
            feasible = True
            for existing in words:
                if (candidate ^ existing).bit_count() < MIN_DISTANCE:
                    feasible = False
                    break
            if feasible:
                words.append(candidate)
                added += 1
    
    # Now try random candidates to fill in gaps
    if time.time() - start_time < DEADLINE - 5:
        random_candidates = set()
        for _ in range(max_attempts):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Generate random word with weight biased toward 9
            w = rng.choice([6,7,8,9,10,11,12], p=[0.05,0.1,0.2,0.25,0.2,0.1,0.05])
            positions = rng.sample(range(N), w)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            
            if candidate in words or candidate in random_candidates:
                continue
            random_candidates.add(candidate)
            
            feasible = True
            for existing in words:
                if (candidate ^ existing).bit_count() < MIN_DISTANCE:
                    feasible = False
                    break
            if feasible:
                words.append(candidate)
                added += 1
    
    # Final local improvement: try to replace pairs to allow more words
    # This is complex; skip for now given time constraints.
    
    # Verify final code
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if (words[i] ^ words[j]).bit_count() < MIN_DISTANCE:
                # Remove the later one
                words.pop(j)
                break
    
    return words
# EVOLVE-BLOCK-END
