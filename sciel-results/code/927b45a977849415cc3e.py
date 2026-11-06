import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # Start with a greedy construction
    # Use a good initial seed: try to build a large code from scratch
    
    def dist(a, b):
        return (a ^ b).bit_count()
    
    def can_add(candidate, code):
        """Check if candidate can be added to code (list of ints)."""
        for w in code:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    # Phase 0: Build initial code using greedy with smart ordering
    # Start with 0, then greedily add words
    
    code = [0]
    code_set = {0}
    
    # Try to add words in a structured way first
    # Words of weight 6, 7, 8 are good candidates for a distance-6 code
    
    # Generate all words of weight 6 (12376 of them) - too many to enumerate
    # Instead, use random with bias toward medium weight
    
    rng = random.Random(12345)
    
    # Fast greedy: try many random candidates
    attempts = 0
    max_attempts_per_phase = 500000
    
    while time_left() > 5.0:
        # Generate candidate
        # Bias toward weight 6-11
        weight = rng.randint(6, 11)
        positions = rng.sample(range(N), weight)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
        
        if candidate in code_set:
            continue
        
        if can_add(candidate, code):
            code.append(candidate)
            code_set.add(candidate)
            attempts = 0
        else:
            attempts += 1
            if attempts > max_attempts_per_phase:
                # Try swap
                if len(code) > 2:
                    idx = rng.randint(0, len(code) - 1)
                    old = code[idx]
                    code.pop(idx)
                    code_set.discard(old)
                    
                    # Try to add a new word
                    added = False
                    for _ in range(100):
                        weight = rng.randint(6, 11)
                        positions = rng.sample(range(N), weight)
                        candidate = 0
                        for p in positions:
                            candidate |= (1 << p)
                        if candidate not in code_set and can_add(candidate, code):
                            code.append(candidate)
                            code_set.add(candidate)
                            added = True
                            break
                    
                    if not added:
                        code.append(old)
                        code_set.add(old)
                    attempts = 0
    
    best_code = code[:]
    best_size = len(code)
    
    # Phase 1: Local search - try to add more with different strategies
    # Use numpy for faster distance checks
    
    def numpy_can_add(candidate, code_arr):
        """code_arr is (M, 17) uint8 array. Returns True if candidate fits."""
        cand = np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)
        dists = np.sum(code_arr ^ cand, axis=1)
        return bool(dists.min() >= MIN_DIST)
    
    code_arr = np.array([[w >> j & 1 for j in range(N)] for w in code], dtype=np.uint8)
    
    # Try to add more words with numpy-accelerated checks
    no_add = 0
    while time_left() > 2.0:
        # Generate candidate
        weight = rng.randint(5, 12)
        positions = rng.sample(range(N), weight)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
        
        if candidate in code_set:
            continue
        
        if numpy_can_add(candidate, code_arr):
            code.append(candidate)
            code_set.add(candidate)
            code_arr = np.vstack([code_arr, np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)])
            no_add = 0
            if len(code) > best_size:
                best_size = len(code)
                best_code = code[:]
        else:
            no_add += 1
            if no_add > 50000:
                # Try swap
                if len(code) > 5:
                    idx = rng.randint(0, len(code) - 1)
                    old = code[idx]
                    code.pop(idx)
                    code_set.discard(old)
                    code_arr = np.delete(code_arr, idx, axis=0)
                    
                    added = False
                    for _ in range(200):
                        weight = rng.randint(5, 12)
                        positions = rng.sample(range(N), weight)
                        candidate = 0
                        for p in positions:
                            candidate |= (1 << p)
                        if candidate not in code_set and numpy_can_add(candidate, code_arr):
                            code.append(candidate)
                            code_set.add(candidate)
                            code_arr = np.vstack([code_arr, np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)])
                            added = True
                            break
                    
                    if not added:
                        code.append(old)
                        code_set.add(old)
                        code_arr = np.vstack([code_arr, np.array([(old >> j) & 1 for j in range(N)], dtype=np.uint8)])
                    no_add = 0
    
    # Phase 2: Try to add words that are at distance 3 from existing words
    # (these are the "tight" candidates)
    while time_left() > 1.0 and len(code) < 350:
        base = code[rng.randint(0, len(code) - 1)]
        # Flip 3 random bits
        flips = rng.sample(range(N), 3)
        candidate = base
        for f in flips:
            candidate ^= (1 << f)
        
        if candidate in code_set:
            continue
        
        if numpy_can_add(candidate, code_arr):
            code.append(candidate)
            code_set.add(candidate)
            code_arr = np.vstack([code_arr, np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)])
            if len(code) > best_size:
                best_size = len(code)
                best_code = code[:]
    
    # Verify best_code
    if best_code:
        valid = True
        for i in range(len(best_code)):
            if not valid:
                break
            for j in range(i + 1, len(best_code)):
                if (best_code[i] ^ best_code[j]).bit_count() < MIN_DIST:
                    valid = False
                    break
        if valid:
            return best_code
    
    # Fallback
    if code:
        return code
    
    return [0]
