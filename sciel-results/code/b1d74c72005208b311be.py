import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0

    N = 17
    D = 6

    # Precompute bit_count for all 2^17 values
    # Actually, let's use a lookup table for 8-bit chunks
    # Or just use Python's .bit_count() which is hardware-accelerated

    def rm_code():
        """Generate 32-word first-order Reed-Muller code on 16 coords."""
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    def greedy_extend(seed_words, rng, time_limit):
        """
        Greedy extension: repeatedly find the candidate with maximum
        minimum-distance to current codewords, add it if distance >= D.
        """
        current = list(seed_words)
        n = len(current)
        
        # For efficiency, we'll maintain the current set as a list
        # and score candidates by their min-distance to the set
        
        # Precompute: for each candidate, we need min over current of popcount(c ^ w)
        # This is expensive. Let's use a sampling-based approach.
        
        # Strategy: maintain a pool of "promising" candidates
        # Each round, sample a batch, score them, add the best
        
        best_min_dist = D  # threshold
        round_num = 0
        
        while time.time() < time_limit:
            round_num += 1
            # Sample candidates - use different ranges as we go
            # Early: sample broadly. Later: focus on unexplored regions.
            batch_size = 4096 if round_num < 100 else 2048
            
            # Generate random candidates
            if round_num < 50:
                # Broad random sampling
                candidates = [rng.randint(1, (1 << N) - 1) for _ in range(batch_size)]
            else:
                # Mix of random and mutations of existing words
                candidates = []
                for _ in range(batch_size // 2):
                    candidates.append(rng.randint(1, (1 << N) - 1))
                for _ in range(batch_size // 2):
                    base = rng.choice(current)
                    # Flip 2-4 random bits
                    mask = 0
                    for _ in range(rng.randint(2, 4)):
                        mask |= 1 << rng.randint(0, N - 1)
                    candidates.append(base ^ mask)
            
            best_cand = None
            best_score = 0
            
            for c in candidates:
                # Check if c satisfies distance constraint to all current words
                min_d = N + 1
                for w in current:
                    d = (c ^ w).bit_count()
                    if d < D:
                        min_d = 0
                        break
                    if d < min_d:
                        min_d = d
                
                if min_d >= D:
                    if min_d > best_score:
                        best_score = min_d
                        best_cand = c
            
            if best_cand is not None:
                current.append(best_cand)
            else:
                # No candidate found in this batch; try a different strategy
                # or break if we've done many rounds
                if round_num > 5000:
                    break
                # Try with a larger batch next time
                continue
        
        return current

    # Generate base RM code
    base_rm = rm_code()
    
    best_result = base_rm
    best_size = len(base_rm)
    
    rng = random.Random(12345)
    
    # Multiple restarts with different seeds
    restart_num = 0
    while time.time() < deadline - 3.0:
        restart_num += 1
        
        # Choose seed
        if restart_num % 4 == 1:
            seed = list(base_rm)
        elif restart_num % 4 == 2:
            # Random seed of size ~10
            seed = [0]
            for _ in range(15):
                w = rng.randint(1, (1 << N) - 1)
                ok = True
                for s in seed:
                    if (w ^ s).bit_count() < D:
                        ok = False
                        break
                if ok:
                    seed.append(w)
        elif restart_num % 4 == 3:
            # Weight-based seed: words with specific weights
            seed = []
            # Add words of weight 8 (middle)
            for _ in range(30):
                w = rng.randint(1, (1 << N) - 1)
                if w.bit_count() == 8:
                    ok = True
                    for s in seed:
                        if (w ^ s).bit_count() < D:
                            ok = False
                            break
                    if ok:
                        seed.append(w)
            if not seed:
                seed = [0]
        else:
            # Random seed of size ~5
            seed = [0]
            for _ in range(8):
                w = rng.randint(1, (1 << N) - 1)
                ok = True
                for s in seed:
                    if (w ^ s).bit_count() < D:
                        ok = False
                        break
                if ok:
                    seed.append(w)
        
        # Extend greedily
        result = greedy_extend(seed, rng, deadline - 2.0)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    return best_result
