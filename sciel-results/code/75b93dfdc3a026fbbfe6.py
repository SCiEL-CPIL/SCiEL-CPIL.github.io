"""Optimized construction: grid-based init + simulated annealing for max 3-AP differences."""
import time
import random
import math

def _score_set(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S."""
    S = set(S)
    diffs = set()
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            a = lst[i]
            c = lst[j]
            if (a + c) % 2 == 0:
                b = (a + c) // 2
                if b in S and b > a:
                    diffs.add((c - a) // 2)
    return len(diffs)

def _score_list(lst):
    """Score from a list (converts to set internally)."""
    return _score_set(set(lst))

def _make_grid_init():
    """Create a 4x8 grid construction with good spacing."""
    # Use a grid where rows are spaced by M and columns are 1..8
    # This gives many small differences within rows and larger ones across rows
    M = 64
    S = set()
    for i in range(4):
        for j in range(8):
            val = i * M + j + 1
            if 1 <= val <= 512:
                S.add(val)
    return list(S)

def _random_individual():
    """Generate a random 32-element subset of 1..512."""
    return random.sample(range(1, 513), 32)

def construct():
    deadline = time.time() + 140  # Safety margin: 10s before 150s kill
    
    # Start with grid construction
    current = _make_grid_init()
    current_score = _score_set(set(current))
    best = list(current)
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor: swap one element with a random one from 1..512
        idx = rng.randrange(32)
        old_val = current[idx]
        new_val = rng.randint(1, 512)
        
        if new_val == old_val:
            T *= cooling
            continue
        
        # Check if new_val already in set
        if new_val in current:
            T *= cooling
            continue
        
        current[idx] = new_val
        new_score = _score_set(set(current))
        
        delta = new_score - current_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = list(current)
        else:
            current[idx] = old_val
        
        T *= cooling
    
    # Final local search: try all single-swap improvements
    changed = True
    while changed and time.time() < deadline:
        changed = False
        cur_set = set(best)
        for i in range(32):
            old_val = best[i]
            for new_val in range(1, 513):
                if new_val == old_val or new_val in cur_set:
                    continue
                best[i] = new_val
                new_score = _score_set(set(best))
                if new_score > best_score:
                    best_score = new_score
                    changed = True
                else:
                    best[i] = old_val
    
    return best
