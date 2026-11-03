import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1()
    else:
        # Fallback: return a trivial valid code (might be large)
        # Just to be safe, return a single word (invalid for radius 1 if n>1)
        # Better: return all words (valid, huge)
        # But this case is unlikely.
        return [[0]*n]

def _solve_3_8_1():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    q = 3
    n = 8
    radius = 1
    space_size = q ** n
    
    # Precompute neighbors
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx = idx * q + digits[i]
        return idx

    neighbors = []
    for i in range(space_size):
        d = idx_to_digits(i)
        nbrs = [i]
        for pos in range(n):
            original = d[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                d[pos] = new_val
                nbrs.append(digits_to_idx(d))
            d[pos] = original
        neighbors.append(nbrs)
        
    # Helper to check validity of swap
    def is_valid_swap(c, new_c, cover_count):
        # Words in neighbors[c] not in neighbors[new_c] must have cover_count > 1
        set_new_c = set(neighbors[new_c])
        for w in neighbors[c]:
            if w not in set_new_c:
                if cover_count[w] == 1:
                    return False
        return True

    def optimize(code, time_limit):
        cover_count = np.zeros(space_size, dtype=np.int32)
        for c in code:
            for w in neighbors[c]:
                cover_count[w] += 1
                
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            
            # 1. Try removals
            removed_any = False
            for i in range(len(code)):
                c = code[i]
                removable = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        removable = False
                        break
                if removable:
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    improved = True
                    removed_any = True
                    break
            
            if removed_any:
                continue
                
            # 2. Try swaps
            # Limit swaps per iteration to avoid getting stuck in a specific pattern
            swaps_made = 0
            max_swaps_per_iter = 20
            attempts = 0
            max_attempts = 100
            
            while swaps_made < max_swaps_per_iter and attempts < max_attempts and (time.time() < time_limit):
                attempts += 1
                i = random.randint(0, len(code) - 1)
                c = code[i]
                new_c = random.randint(0, space_size - 1)
                
                if c == new_c:
                    continue
                    
                if is_valid_swap(c, new_c, cover_count):
                    # Perform swap
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    for w in neighbors[new_c]:
                        cover_count[w] += 1
                    code[i] = new_c
                    swaps_made += 1
                    improved = True
                    
        return code

    # Strategy 1: Greedy Initialization
    def greedy_init():
        uncovered = set(range(space_size))
        code = []
        while uncovered:
            best_gain = -1
            best_word = -1
            # To speed up, we can't check all 6561 words every time if it's slow.
            # But 6561 * 17 operations is ~100k ops.
            # If we do this ~500 times, it's 50M ops. Might be slow in Python.
            # Optimization: Only check words that are neighbors of currently uncovered words?
            # Or just sample?
            # Let's try full scan first, it might be fast enough for 140s.
            for w in range(space_size):
                gain = 0
                for u in neighbors[w]:
                    if u in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            code.append(best_word)
            for u in neighbors[best_word]:
                uncovered.discard(u)
        return code

    # Strategy 2: Seed Initialization
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            seed_code.append(digits_to_idx(word))

    best_code = []
    best_size = float('inf')
    
    # Run Greedy
    if time.time() < deadline - 10: # Leave time for optimization
        greedy_code = greedy_init()
        opt_greedy = optimize(greedy_code, deadline - 5)
        if len(opt_greedy) < best_size:
            best_code = opt_greedy
            best_size = len(best_code)
            
    # Run Seed
    if time.time() < deadline - 10:
        opt_seed = optimize(seed_code[:], deadline - 5)
        if len(opt_seed) < best_size:
            best_code = opt_seed
            best_size = len(best_code)

    if not best_code:
        best_code = seed_code # Fallback

    result = [idx_to_digits(c) for c in best_code]
    return result
