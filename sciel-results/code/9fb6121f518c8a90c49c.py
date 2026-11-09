"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _score_set(S):
    """Return the number of distinct d > 0 such that a, a+d, a+2d all in S."""
    S_set = set(S)
    diffs = set()
    # For each pair (x, y) with x < y, check if x + 2*(y-x) is in S
    # Actually, for a 3-term AP a, a+d, a+2d: d = (y-x) where y = a+d, a+2d = x+2d
    # Better: for each a in S and each d, check a, a+d, a+2d in S
    # But that's O(n^2 * something). Let's do: for each pair (a, c) with a < c, 
    # if (c-a) is even, then d = (c-a)//2, and check if a+d in S.
    # Actually: a, a+d, a+2d means the three elements are a, a+d, a+2d.
    # So for any two elements x < z in S, if (z-x) is even, let d = (z-x)//2, 
    # and check if x+d in S.
    # But we need distinct d values.
    sorted_S = sorted(S)
    n = len(sorted_S)
    for i in range(n):
        for j in range(i+1, n):
            diff = sorted_S[j] - sorted_S[i]
            if diff % 2 == 0:
                d = diff // 2
                mid = sorted_S[i] + d
                if mid in S_set:
                    diffs.add(d)
    return len(diffs)


def _greedy_optimize(initial_set, max_iters=50000, time_budget=140.0):
    """Local search: try to improve the set by swapping elements."""
    S = list(initial_set)
    S_set = set(S)
    current_score = _score_set(S)
    
    start_time = time.time()
    best_S = list(S)
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Try random swaps: pick two elements, one in S and one not in S (in 1..512)
        # Replace one element from S with one not in S
        idx = random.randint(0, len(S) - 1)
        old_val = S[idx]
        
        # Pick a new value from 1..512 not in S
        candidates = [x for x in range(1, 513) if x not in S_set]
        if not candidates:
            break
        new_val = random.choice(candidates)
        
        S[idx] = new_val
        S_set.remove(old_val)
        S_set.add(new_val)
        
        new_score = _score_set(S)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            # Revert
            S[idx] = old_val
            S_set.remove(new_val)
            S_set.add(old_val)
    
    return best_S, best_score


def _construct_cantor_like(base, digits, size):
    """Construct a set using base representation with restricted digits."""
    S = set()
    for d1 in digits:
        for d2 in digits:
            for d3 in digits:
                for d4 in digits:
                    val = d1 + d2 * base + d3 * base**2 + d4 * base**3
                    if 1 <= val <= 512:
                        S.add(val)
    return sorted(S)


def _try_arithmetic_progressions():
    """Try unions of arithmetic progressions."""
    best_S = None
    best_score = 0
    start_time = time.time()
    
    # Try two APs: {a + i*d1} and {b + j*d2}
    for d1 in range(1, 20):
        for d2 in range(1, 20):
            if d1 == d2:
                continue
            # Try to find offsets a, b such that the union has 32 elements
            # and maximizes progression differences
            # This is complex, so let's just try some structured sets
            
    return best_S, best_score


def _random_search(time_budget=130.0):
    """Random restarts with local search."""
    start_time = time.time()
    best_S = list(range(1, 33))
    best_score = _score_set(best_S)
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget - 10:
            break
        
        # Generate a random initial set
        S = random.sample(range(1, 513), 32)
        S_score = _score_set(S)
        
        # Local search
        S_opt, score_opt = _greedy_optimize(S, max_iters=5000, time_budget=5.0)
        
        if score_opt > best_score:
            best_score = score_opt
            best_S = S_opt
    
    return best_S, best_score


def _structured_search(time_budget=120.0):
    """Try structured constructions: APs, Cantor-like, etc."""
    start_time = time.time()
    best_S = list(range(1, 33))
    best_score = _score_set(best_S)
    
    # Try various step sizes for single APs
    for step in range(1, 17):
        if time.time() - start_time > time_budget - 5:
            break
        S = [1 + i * step for i in range(32)]
        if max(S) <= 512:
            score = _score_set(S)
            if score > best_score:
                best_score = score
                best_S = list(S)
    
    # Try unions of two APs
    for d1 in range(1, 12):
        for d2 in range(1, 12):
            if time.time() - start_time > time_budget - 5:
                break
            if d1 == d2:
                continue
            # Try to split 32 into two groups
            for n1 in range(4, 29):
                n2 = 32 - n1
                # First AP: a1, a1+d1, ..., a1+(n1-1)*d1
                # Second AP: a2, a2+d2, ..., a2+(n2-1)*d2
                # Need all 32 distinct and in 1..512
                # Try a1 = 1, a2 = 1 + n1*d1 + 1 (or similar)
                a1 = 1
                S1 = [a1 + i * d1 for i in range(n1)]
                if max(S1) > 512:
                    continue
                a2 = max(S1) + 1
                S2 = [a2 + i * d2 for i in range(n2)]
                if max(S2) > 512:
                    continue
                S = S1 + S2
                if len(set(S)) != 32:
                    continue
                score = _score_set(S)
                if score > best_score:
                    best_score = score
                    best_S = list(S)
    
    return best_S, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_S = list(range(1, 33))
    best_score = _score_set(best_S)
    
    # Phase 1: Structured search
    if time.time() < deadline - 30:
        S_struct, score_struct = _structured_search(time_budget=25.0)
        if score_struct > best_score:
            best_score = score_struct
            best_S = S_struct
    
    # Phase 2: Random search with local optimization
    if time.time() < deadline - 15:
        S_rand, score_rand = _random_search(time_budget=130.0 - (time.time() - start_time))
        if score_rand > best_score:
            best_score = score_rand
            best_S = S_rand
    
    # Phase 3: Final local polish on the best set found
    remaining = deadline - time.time()
    if remaining > 2:
        S_polished, score_polished = _greedy_optimize(best_S, max_iters=20000, time_budget=remaining - 1)
        if score_polished > best_score:
            best_score = score_polished
            best_S = S_polished
    
    return list(best_S)

# EVOLVE-BLOCK-END
