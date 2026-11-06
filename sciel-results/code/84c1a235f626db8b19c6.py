import time
import random
import numpy as np

def _score_bitmask(mask):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d all in set represented by bitmask.
    mask is a Python int where bit i (0-indexed) represents value i+1 (so bit 0 = value 1, bit 511 = value 512).
    """
    diffs = set()
    n = 512
    # For each d from 1 to 255 (max d such that a+2d <= 512 for some a >= 1)
    # Actually d can be up to 255 since a >= 1, a+2d <= 512 => 2d <= 511 => d <= 255
    for d in range(1, 256):
        # Check if there exists a such that bits at a-1, a+d-1, a+2d-1 are all set
        # a ranges from 1 to 512-2d
        found = False
        for a in range(1, 513 - 2 * d):
            if (mask >> (a - 1)) & 1 and (mask >> (a + d - 1)) & 1 and (mask >> (a + 2 * d - 1)) & 1:
                found = True
                break
        if found:
            diffs.add(d)
    return len(diffs)

def _score_set(S):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d all in S."""
    S_set = set(S)
    diffs = set()
    S_list = sorted(S)
    for a in S_list:
        for b in S_list:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def _score_set_fast(S):
    """Faster scoring using sorted list and set membership."""
    S_set = set(S)
    S_list = sorted(S)
    diffs = set()
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            d = S_list[j] - a
            if d > 0 and (a + 2 * d) in S_set:
                diffs.add(d)
    return len(diffs)

def _greedy_init(rng):
    """Greedy construction: start with random elements, then add elements that maximize score improvement."""
    N = 32
    MAXV = 512
    S = set()
    # Start with a random base
    for _ in range(8):
        S.add(rng.randint(1, MAXV))
    
    current_score = _score_set_fast(S)
    
    # Greedily add elements
    while len(S) < N:
        best_elem = None
        best_score = current_score
        # Try all candidates
        for x in range(1, MAXV + 1):
            if x not in S:
                new_S = S | {x}
                new_score = _score_set_fast(new_S)
                if new_score > best_score:
                    best_score = new_score
                    best_elem = x
        if best_elem is not None:
            S.add(best_elem)
            current_score = best_score
        else:
            # No improvement, pick random
            candidates = [x for x in range(1, MAXV + 1) if x not in S]
            if candidates:
                S.add(rng.choice(candidates))
    
    return S, current_score

def _local_search(current, deadline, rng):
    """Perform local search / simulated annealing from current set."""
    N = 32
    MAXV = 512
    current_score = _score_set_fast(current)
    best = set(current)
    best_score = current_score
    
    T = 10.0
    T_min = 0.5
    cooling = 0.9995
    
    while time.time() < deadline:
        # Generate a move: swap one element
        to_remove = rng.choice(list(current))
        candidates = [x for x in range(1, MAXV + 1) if x not in current]
        to_add = rng.choice(candidates)
        
        if to_remove == to_add:
            continue
        
        new_set = set(current)
        new_set.remove(to_remove)
        new_set.add(to_add)
        new_score = _score_set_fast(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current = new_set
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def _two_opt(current, deadline, rng):
    """2-opt style: try swapping pairs of elements."""
    N = 32
    MAXV = 512
    current_score = _score_set_fast(current)
    best = set(current)
    best_score = current_score
    
    while time.time() < deadline:
        # Pick two elements to remove
        elems = list(current)
        r1 = rng.choice(elems)
        r2 = rng.choice(elems)
        if r1 == r2:
            continue
        
        # Pick two elements to add
        candidates = [x for x in range(1, MAXV + 1) if x not in current]
        a1 = rng.choice(candidates)
        a2 = rng.choice(candidates)
        if a1 == a2:
            continue
        
        new_set = set(current)
        new_set.discard(r1)
        new_set.discard(r2)
        new_set.add(a1)
        new_set.add(a2)
        
        if len(new_set) != N:
            continue
        
        new_score = _score_set_fast(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    # Start with greedy initialization
    best_overall, best_overall_score = _greedy_init(rng)
    
    # Multiple restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random initial set
        current = set(rng.sample(range(1, MAXV + 1), N))
        current_score = _score_set_fast(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search with a sub-deadline
        sub_deadline = min(deadline, start_time + 10.0 + num_restarts * 2.0)
        best, best_score = _local_search(current, sub_deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
        
        # If we have time, try 2-opt on the best
        if time.time() < deadline - 5.0:
            best2, best_score2 = _two_opt(best_overall, min(deadline - 2.0, start_time + 5.0), rng)
            if best_score2 > best_overall_score:
                best_overall = best2
                best_overall_score = best_score2
    
    return sorted(list(best_overall))
