import time
import random
import numpy as np

# EVOLVE-BLOCK-START

MAX_VAL = 512
N_ELEMS = 32
MAX_D = (MAX_VAL - 1) // 2  # 255

def _score_bitset(S_int):
    """S_int: Python int bitmask. bit (x-1) set iff x in set. Return count of distinct d covered."""
    count = 0
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
        lsb = v & (-v)
        idx = lsb.bit_length() - 1
        s.add(idx + 1)
        v ^= lsb
    return s

def _popcount(v):
    return bin(v).count('1')

def _delta_score_swap(S_int, remove_elem, add_elem):
    """
    Compute the change in score after removing remove_elem and adding add_elem from S_int.
    Only checks d values where the removed or added element can participate in a 3-term AP.
    Returns (delta, new_S_int).
    """
    r = remove_elem - 1  # 0-indexed
    a = add_elem - 1     # 0-indexed
    
    # d values where removing r could affect coverage:
    # r as first: r, r+d, r+2d -> d in 1..(MAX_VAL-1-r)//2
    # r as second: r-d, r, r+d -> d in 1..min(r, MAX_VAL-1-r)
    # r as third: r-2d, r-d, r -> d in 1..r//2
    # d values where adding a could affect coverage:
    # a as first: a, a+d, a+2d -> d in 1..(MAX_VAL-1-a)//2
    # a as second: a-d, a, a+d -> d in 1..min(a, MAX_VAL-1-a)
    # a as third: a-2d, a-d, a -> d in 1..a//2
    
    candidate_ds = set()
    
    # From removal of r
    max_d_r_first = (MAX_VAL - 1 - r) // 2
    for d in range(1, max_d_r_first + 1):
        candidate_ds.add(d)
    max_d_r_second = min(r, MAX_VAL - 1 - r)
    for d in range(1, max_d_r_second + 1):
        candidate_ds.add(d)
    max_d_r_third = r // 2
    for d in range(1, max_d_r_third + 1):
        candidate_ds.add(d)
    
    # From addition of a
    max_d_a_first = (MAX_VAL - 1 - a) // 2
    for d in range(1, max_d_a_first + 1):
        candidate_ds.add(d)
    max_d_a_second = min(a, MAX_VAL - 1 - a)
    for d in range(1, max_d_a_second + 1):
        candidate_ds.add(d)
    max_d_a_third = a // 2
    for d in range(1, max_d_a_third + 1):
        candidate_ds.add(d)
    
    # Also need to check d values that might become uncovered due to removal
    # even if they don't directly involve r or a in the "new" sense
    # Actually the above should cover it: if a d was covered only by progressions
    # involving r, it will be checked. If a d becomes covered by progressions
    # involving a, it will be checked.
    
    delta = 0
    for d in candidate_ds:
        # Check if d is covered in original S
        orig_covered = (S_int & (S_int >> d) & (S_int >> (2 * d))) != 0
        # Check if d is covered in new S
        new_S = S_int
        new_S &= ~(1 << r)
        new_S |= (1 << a)
        new_covered = (new_S & (new_S >> d) & (new_S >> (2 * d))) != 0
        if new_covered and not orig_covered:
            delta += 1
        elif not new_covered and orig_covered:
            delta -= 1
    
    new_S_int = S_int
    new_S_int &= ~(1 << r)
    new_S_int |= (1 << a)
    return delta, new_S_int

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
    
    all_elements = list(range(1, MAX_VAL + 1))
    
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_delta = 0
        best_swap = None
        best_new_S = None
        
        for i, elem_out in enumerate(S):
            if (time.time() - start) > time_budget:
                break
            for elem_in in all_elements:
                if elem_in in s_set:
                    continue
                delta, new_S_int = _delta_score_swap(S_int, elem_out, elem_in)
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
        idx = random.randrange(N_ELEMS)
        elem_out = S[idx]
        elem_in = random.randint(1, MAX_VAL)
        
        if elem_in in s_set:
            continue
        
        delta, new_S_int = _delta_score_swap(S_int, elem_out, elem_in)
        
        if delta > 0:
            S[idx] = elem_in
            s_set.remove(elem_out)
            s_set.add(elem_in)
            S_int = new_S_int
            current_score += delta
            stuck_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            if delta != 0 and random.random() < np.exp(delta / T):
                S[idx] = elem_in
                s_set.remove(elem_out)
                s_set.add(elem_in)
                S_int = new_S_int
                current_score += delta
            else:
                stuck_count += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        if stuck_count > max_stuck:
            # Multi-swap perturbation: swap 2-3 elements randomly
            num_swaps = random.randint(2, 4)
            for _ in range(num_swaps):
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
            T = max(T, 40.0)
    
    return best_S, best_score

def _generate_initial_set():
    """Generate a random initial set of 32 elements."""
    return random.sample(range(1, MAX_VAL + 1), N_ELEMS)

def _generate_structured_set():
    """Generate a structured initial set."""
    strategy = random.choice(['ap', 'cluster', 'mixed', 'random'])
    if strategy == 'ap':
        d = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
        a = random.randint(1, MAX_VAL - 31 * d)
        if a < 1:
            return random.sample(range(1, MAX_VAL + 1), N_ELEMS)
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
    elif strategy == 'mixed':
        # Mix of small and large gaps
        s = set()
        while len(s) < 32:
            if random.random() < 0.5:
                base = random.randint(1, MAX_VAL - 10)
                d = random.choice([1, 2, 3])
                for k in range(random.randint(3, 8)):
                    x = base + k * d
                    if 1 <= x <= MAX_VAL:
                        s.add(x)
            else:
                s.add(random.randint(1, MAX_VAL))
        return list(s)
    else:
        return random.sample(range(1, MAX_VAL + 1), N_ELEMS)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_set = None
    best_score = 0
    
    num_restarts = 60  # More restarts possible due to faster delta scoring
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart % 3 == 0:
            S_init = _generate_structured_set()
        else:
            S_init = _generate_initial_set()
        
        remaining = deadline - time.time()
        sa_budget = min(25.0, remaining * 0.5)
        if sa_budget < 2.0:
            break
        
        S_sa, score_sa = _sa_search(S_init, time_budget=sa_budget)
        
        if score_sa > best_score:
            best_score = score_sa
            best_set = S_sa
        
        if time.time() < deadline - 5:
            polish_budget = min(8.0, deadline - time.time() - 5)
            S_polished, score_polished = _greedy_polish(S_sa, time_budget=polish_budget)
            if score_polished > best_score:
                best_score = score_polished
                best_set = S_polished
    
    if best_set and time.time() < deadline - 3:
        polish_budget = min(10.0, deadline - time.time() - 3)
        S_final, score_final = _greedy_polish(best_set, time_budget=polish_budget)
        if score_final > best_score:
            best_score = score_final
            best_set = S_final
    
    if best_set is None:
        best_set = list(range(1, N_ELEMS + 1))
    
    return best_set

# EVOLVE-BLOCK-END
