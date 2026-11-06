"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

def _reset_length_bitmask(a, b):
    """
    Compute shortest reset word length using bitmasks for efficiency.
    a, b: lists of ints 0..N-1
    Returns length or -1 if not synchronizing.
    """
    # Precompute transitions for each letter as a list of bitmasks?
    # No, we need to map a subset (mask) to next subset.
    # Let's precompute for each state i, the bit set for a[i] and b[i].
    # Actually, it's easier to just compute on the fly but with bit operations.
    
    # Start with all states present: (1 << N) - 1
    start_mask = (1 << N) - 1
    
    # Check if a or b is already a reset (constant function)
    # If a maps all to same, then a is a reset word of length 1 (or 0 if we consider empty? No, length 1).
    # The problem asks for shortest reset word.
    # If the automaton is already synchronized? No, "shortest reset word" implies we start from full set.
    # If a is constant, applying 'a' to full set gives a singleton. Length 1.
    
    # BFS
    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Apply a
        # next_mask_a = union of a[i] for all i in mask
        next_a = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_a |= (1 << a[i])
            m -= lsb
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply b
        next_b = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_b |= (1 << b[i])
            m -= lsb
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    # Candidate 1: Standard Černý
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0] + [i - 1 for i in range(1, N)]
    
    # Candidate 2: Random Cyclic
    rng = random.Random(42)
    best_a, best_b = a1[:], b1[:]
    best_score = _reset_length_bitmask(best_a, best_b)
    
    # Try a few random cyclic structures to find a better starting point
    for _ in range(20):
        # a is a random permutation (cycle)
        perm = list(range(N))
        rng.shuffle(perm)
        # b is random
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        s = _reset_length_bitmask(perm, b_rand)
        if s > best_score:
            best_score = s
            best_a, best_b = perm[:], b_rand[:]
            
    # Simulated Annealing
    start_time = time.time()
    time_limit = 55.0
    current_a, current_b = best_a[:], best_b[:]
    current_score = best_score
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = 10.0 # Restart annealing if stuck, or just break? Let's restart with new random perturbation
            # Or just continue with low T
            pass
            
        # Generate neighbor
        new_a = current_a[:]
        new_b = current_b[:]
        
        # Mutation: change 1 or 2 transitions
        num_changes = rng.choice([1, 1, 1, 2])
        for _ in range(num_changes):
            if rng.random() < 0.5:
                idx = rng.randint(0, N-1)
                new_a[idx] = rng.randint(0, N-1)
            else:
                idx = rng.randint(0, N-1)
                new_b[idx] = rng.randint(0, N-1)
                
        new_score = _reset_length_bitmask(new_a, new_b)
        
        # Acceptance criterion
        if new_score > current_score:
            current_a, current_b = new_a, new_b
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_a, best_b = current_a[:], current_b[:]
        else:
            delta = new_score - current_score
            # delta <= 0
            prob = math.exp(delta / T) if T > 0 else 0
            if rng.random() < prob:
                current_a, current_b = new_a, new_b
                current_score = new_score
                
        T *= cooling_rate
        
    return best_a, best_b
# EVOLVE-BLOCK-END
