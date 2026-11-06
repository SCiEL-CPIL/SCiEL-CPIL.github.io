"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    _start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    N = 32
    MAXV = 512
    
    def score_set(s):
        """Count distinct d>0 such that a, a+d, a+2d all in s."""
        bits = 0
        for x in s:
            bits |= (1 << x)
        # For each possible d, check if there exists a with a, a+d, a+2d in set
        # a ranges 1..MAXV-2d, so a+d up to MAXV-d, a+2d up to MAXV
        # bits has bit i set if i in s (1-indexed, so bit i means value i)
        count = 0
        for d in range(1, MAXV // 2 + 1):
            # Check: exists a in [1, MAXV-2d] such that bits has a, a+d, a+2d
            # (bits >> d) & (bits >> (2*d)) & bits should have a common bit
            # Actually: for each a, check bit a, bit a+d, bit a+2d
            # Equivalent: (bits & (bits >> d) & (bits >> (2*d))) != 0
            if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                count += 1
        return count
    
    def score_bitset(bits):
        """Fast score using precomputed bitset."""
        count = 0
        for d in range(1, MAXV // 2 + 1):
            if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                count += 1
        return count
    
    # Initial solution: try several random starts and pick best
    best_set = None
    best_score = -1
    
    # Try a structured initial: evenly spaced
    for trial in range(5):
        if trial == 0:
            # Evenly spaced
            s = list(range(1, MAXV, MAXV // N + 1))[:N]
            s = sorted(set(s))
            if len(s) < N:
                s = list(range(1, N + 1))
        else:
            s = random.sample(range(1, MAXV + 1), N)
            s = sorted(s)
        
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
    
    bits = 0
    for x in best_set:
        bits |= (1 << x)
    best_score = score_bitset(bits)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    current_set = best_set[:]
    current_bits = bits
    current_score = best_score
    
    elapsed = time.time() - _start
    while T > T_min and (time.time() - _start) < DEADLINE:
        # Random move: swap, insert, or remove+insert
        move_type = random.random()
        if move_type < 0.4:
            # Swap two elements
            i, j = random.sample(range(N), 2)
            old_i, old_j = current_set[i], current_set[j]
            new_i = random.randint(1, MAXV)
            new_j = random.randint(1, MAXV)
            # Ensure distinct
            if new_i == new_j or new_i in current_set or new_j in current_set:
                continue
            current_set[i] = new_i
            current_set[j] = new_j
            current_set.sort()
            new_bits = 0
            for x in current_set:
                new_bits |= (1 << x)
            new_score = score_bitset(new_bits)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                current_bits = new_bits
            else:
                current_set[i], current_set[j] = old_i, old_j
                current_set.sort()
                current_bits = 0
                for x in current_set:
                    current_bits |= (1 << x)
        elif move_type < 0.7:
            # Replace one element
            i = random.randint(0, N - 1)
            old = current_set[i]
            new = random.randint(1, MAXV)
            if new in current_set:
                continue
            current_set[i] = new
            current_set.sort()
            new_bits = 0
            for x in current_set:
                new_bits |= (1 << x)
            new_score = score_bitset(new_bits)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                current_bits = new_bits
            else:
                current_set[i] = old
                current_set.sort()
                current_bits = 0
                for x in current_set:
                    current_bits |= (1 << x)
        else:
            # Move one element to a new position
            i = random.randint(0, N - 1)
            old = current_set[i]
            new = random.randint(1, MAXV)
            if new in current_set:
                continue
            current_set[i] = new
            current_set.sort()
            new_bits = 0
            for x in current_set:
                new_bits |= (1 << x)
            new_score = score_bitset(new_bits)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                current_bits = new_bits
            else:
                current_set[i] = old
                current_set.sort()
                current_bits = 0
                for x in current_set:
                    current_bits |= (1 << x)
        
        T *= cooling
    
    # Greedy polish
    polished = True
    while polished and (time.time() - _start) < DEADLINE - 5:
        polished = False
        # Try swapping each pair
        for i in range(N):
            for j in range(i + 1, N):
                old_i, old_j = current_set[i], current_set[j]
                for new_i in range(1, MAXV + 1):
                    if new_i in current_set:
                        continue
                    for new_j in range(1, MAXV + 1):
                        if new_j == new_i or new_j in current_set:
                            continue
                        # This is too slow - O(N^2 * MAXV^2)
                        break
                break
            if (time.time() - _start) > DEADLINE - 5:
                break
    
    # Better greedy: try replacing each element with best alternative
    for _ in range(3):
        if (time.time() - _start) > DEADLINE - 5:
            break
        improved = True
        while improved and (time.time() - _start) < DEADLINE - 5:
            improved = False
            for i in range(N):
                old = current_set[i]
                best_alt = None
                best_alt_score = current_score
                for new in range(1, MAXV + 1):
                    if new in current_set:
                        continue
                    current_set[i] = new
                    current_set.sort()
                    new_bits = 0
                    for x in current_set:
                        new_bits |= (1 << x)
                    new_score = score_bitset(new_bits)
                    current_set[i] = old
                    if new_score > best_alt_score:
                        best_alt_score = new_score
                        best_alt = new
                if best_alt is not None:
                    current_set[i] = best_alt
                    current_set.sort()
                    current_bits = 0
                    for x in current_set:
                        current_bits |= (1 << x)
                    current_score = best_alt_score
                    improved = True
                    if (time.time() - _start) > DEADLINE - 5:
                        break
    
    # Track best
    if current_score > best_score:
        best_score = current_score
        best_set = current_set[:]
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
