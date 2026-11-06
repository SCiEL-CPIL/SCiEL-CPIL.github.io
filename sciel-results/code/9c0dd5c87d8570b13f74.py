import time
import random
import numpy as np

# EVOLVE-BLOCK-START

MAX_VAL = 512
N_ELEMS = 32
MAX_D = (MAX_VAL - 1) // 2  # 255

# Precompute bit masks for each element: mask[x] has bit i set if element (x - i) is a valid 1..512
# Actually we need: for a set S (bitmask), d is present iff exists a: a in S, a+d in S, a+2d in S
# Let S be a Python int bitmask where bit (x-1) is set if x in S.
# For difference d: (S & (S >> d) & (S >> (2*d))) != 0  ... but we need a, a+d, a+2d all present.
# a in S, a+d in S means: S has bit (a-1) and bit (a+d-1).
# Let's define: for each d, d is "covered" if there exists a such that bits (a-1), (a-1+d), (a-1+2d) are all set.
# That's: (S >> (a-1)) has bits 0, d, 2d set.
# Equivalently: S & (S >> d) & (S >> (2*d)) has a bit at position (a-1).
# So: covered_d = ((S & (S >> d) & (S >> (2*d))) != 0)

def _score_bitset(S_int):
    """S_int: Python int bitmask. bit (x-1) set iff x in set. Return count of distinct d covered."""
    count = 0
    # S_int >> d shifts so that bit (a-1+d) becomes bit (a-1).
    # We want: exists a: bit(a-1) in S, bit(a-1+d) in S, bit(a-1+2d) in S
    # = bit(a-1) in (S & (S>>d) & (S>>(2d)))
    for d in range(1, MAX_D + 1):
        if (S_int & (S_int >> d) & (S_int >> (2 * d))) != 0:
            count += 1
    return count

def _score_bitset_with_diffs(S_int):
    """Return (count, set of covered d's)."""
    diffs = set()
    for d in range(1, MAX_D + 1):
        if (S_int & (S_int >> d) & (S_int >> (2 * d))) != 0:
            diffs.add(d)
    return len(diffs), diffs

def _set_to_int(S):
    """Convert set of ints to bitmask int."""
    v = 0
    for x in S:
        v |= (1 << (x - 1))
    return v

def _int_to_set(v):
    """Convert bitmask int to set of ints."""
    s = set()
    while v:
        # get lowest set bit
        lsb = v & (-v)
        idx = lsb.bit_length() - 1
        s.add(idx + 1)
        v ^= lsb
    return s

def _popcount(v):
    return bin(v).count('1')

def _delta_score_bitset(S_int, remove_elem, add_elem):
    """
    Compute new score after removing remove_elem and adding add_elem from S_int.
    Returns new_score (int).
    """
    new_S = S_int
    new_S &= ~(1 << (remove_elem - 1))
    new_S |= (1 << (add_elem - 1))
    return _score_bitset(new_S)

def _score_list(S_list):
    """Score from a list (for initial setup)."""
    v = _set_to_int(set(S_list))
    return _score_bitset(v)

def _greedy_polish(S, time_budget=10.0):
    """
    Greedy 1-swap polish: repeatedly find the best single swap that improves score.
    S: list of 32 ints. Returns (improved_list, score).
    """
    start = time.time()
    S = list(S)
    s_set = set(S)
    S_int = _set_to_int(s_set)
    current_score = _score_bitset(S_int)
    
    # Precompute all candidate elements not in S
    all_elements = list(range(1, MAX_VAL + 1))
    
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_delta = 0
        best_swap = None
        best_new_S = None
        
        # For each element in S, try swapping with each element not in S
        for i, elem_out in enumerate(S):
            if (time.time() - start) > time_budget:
                break
            mask_out = 1 << (elem_out - 1)
            for elem_in in all_elements:
                if elem_in in s_set:
                    continue
                # Compute new score
                new_S_int = S_int
                new_S_int &= ~mask_out
                new_S_int |= (1 << (elem_in - 1))
                new_score = _score_bitset(new_S_int)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (i, elem_out, elem_in)
                    best_new_S = new_S_int
        
        if best_swap is not None and best_delta > 0:
            i, elem_out, elem_in = best_swap
            S[i] = elem_in
            s_set.remove(elem_out)
            s_set.add(elem_in)
            S_int = best_new_S
            current_score += best_delta
            improved = True
    
    return S, current_score

def _sa_search(S_init, time_budget=30.0, T_start=50.0, cooling=0.997, min_T=0.5):
    """
    Simulated annealing with bitset scoring.
    S_init: list of 32 ints.
    Returns (best_list, best_score).
    """
    start = time.time()
    S = list(S_init)
    s_set = set(S)
    S_int = _set_to_int(s_set)
    current_score = _score_bitset(S_int)
    best_score = current_score
    best_S = list(S)
    
    T = T_start
    stuck_count = 0
    max_stuck = 3000
    
    while (time.time() - start) < time_budget:
        # Generate neighbor: swap one element
        idx = random.randrange(N_ELEMS)
        elem_out = S[idx]
        elem_in = random.randint(1, MAX_VAL)
        
        if elem_in in s_set:
            continue
        
        mask_out = 1 << (elem_out - 1)
        new_S_int = S_int
        new_S_int &= ~mask_out
        new_S_int |= (1 << (elem_in - 1))
        
        new_score = _score_bitset(new_S_int)
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            S[idx] = elem_in
            s_set.remove(elem_out)
            s_set.add(elem_in)
            S_int = new_S_int
            current_score = new_score
            stuck_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_S = list(S)
        else:
            # Accept with prob exp(delta/T)
            if delta != 0 and random.random() < np.exp(delta / T):
                S[idx] = elem_in
                s_set.remove(elem_out)
                s_set.add(elem_in)
                S_int = new_S_int
                current_score = new_score
            else:
                stuck_count += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        if stuck_count > max_stuck:
            # Perturbation: random swap
            for _ in range(3):
                idx = random.randrange(N_ELEMS)
                elem_out = S[idx]
                elem_in = random.randint(1, MAX_VAL)
                if elem_in not in s_set:
                    S[idx] = elem_in
                    s_set.remove(elem_out)
                    s_set.add(elem_in)
                    S_int = _set_to_int(s_set)
                    current_score = _score_bitset(S_int)
            stuck_count = 0
            T = max(T, 30.0)
    
    return best_S, best_score

def _generate_initial_set():
    """Generate a random initial set of 32 elements."""
    return random.sample(range(1, MAX_VAL + 1), N_ELEMS)

def _generate_structured_set():
    """Generate a structured initial set (arithmetic progression or cluster)."""
    strategy = random.choice(['ap', 'cluster', 'random'])
    if strategy == 'ap':
        d = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        a = random.randint(1, MAX_VAL - 31 * d)
        s = set(a + i * d for i in range(32))
        if all(1 <= x <= MAX_VAL for x in s):
            return list(s)
    elif strategy == 'cluster':
        s = set()
        while len(s) < 32:
            base = random.randint(1, MAX_VAL - 31)
            for k in range(32):
                if base + k <= MAX_VAL:
                    s.add(base + k)
                if len(s) >= 32:
                    break
        return list(s)
    else:
        return random.sample(range(1, MAX_VAL + 1), N_ELEMS)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_set = None
    best_score = 0
    
    num_restarts = 40  # More restarts possible due to fast bitset scoring
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial set
        if restart % 3 == 0:
            S_init = _generate_structured_set()
        else:
            S_init = _generate_initial_set()
        
        # SA search with time budget proportional to remaining time
        remaining = deadline - time.time()
        sa_budget = min(20.0, remaining * 0.5)
        if sa_budget < 2.0:
            break
        
        S_sa, score_sa = _sa_search(S_init, time_budget=sa_budget)
        
        if score_sa > best_score:
            best_score = score_sa
            best_set = S_sa
        
        # Greedy polish on the SA result
        if time.time() < deadline - 5:
            polish_budget = min(5.0, deadline - time.time() - 5)
            S_polished, score_polished = _greedy_polish(S_sa, time_budget=polish_budget)
            if score_polished > best_score:
                best_score = score_polished
                best_set = S_polished
    
    # Final: if we have time, do one more polish on best_set
    if best_set and time.time() < deadline - 3:
        polish_budget = min(8.0, deadline - time.time() - 3)
        S_final, score_final = _greedy_polish(best_set, time_budget=polish_budget)
        if score_final > best_score:
            best_score = score_final
            best_set = S_final
    
    if best_set is None:
        best_set = list(range(1, N_ELEMS + 1))
    
    return best_set

# EVOLVE-BLOCK-END
