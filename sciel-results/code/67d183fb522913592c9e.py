"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_set(S):
    """Check if set S satisfies the constraint."""
    S_list = sorted(S)
    for i, a in enumerate(S_list):
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % a == 0:
                    return False
    return True

def _try_add(S, cand, n):
    """Check if cand can be added to S. Optimized: precompute sorted lists."""
    S_list = sorted(S)
    import bisect
    
    # Case 1: cand plays role of 'a' (smallest in triple)
    # Need: no b,c in S with b > cand, c >= b, cand | (b+c)
    idx = bisect.bisect_right(S_list, cand)
    larger = S_list[idx:]
    for i in range(len(larger)):
        b = larger[i]
        for j in range(i, len(larger)):
            c = larger[j]
            if (b + c) % cand == 0:
                return False
    
    # Case 2: cand plays role of 'b' (middle in triple)
    # Need: no a,c in S with a < cand, c >= cand, a | (cand + c)
    idx_smaller = bisect.bisect_left(S_list, cand)
    smaller = S_list[:idx_smaller]
    geq = S_list[idx_smaller:]
    for a in smaller:
        for c in geq:
            if (cand + c) % a == 0:
                return False

    # Case 3: cand plays role of 'c' (largest in triple)
    # Need: no a,b in S with a < b <= cand, a | (b + cand)
    idx_le = bisect.bisect_right(S_list, cand)
    leq = S_list[:idx_le]
    for i in range(len(leq)):
        a = leq[i]
        if a >= cand:
            break
        for j in range(i+1, len(leq)):
            b = leq[j]
            if (b + cand) % a == 0:
                return False
    return True

def _local_search_add(S, n, deadline):
    """Try to add skipped elements to S."""
    S = set(S)
    skipped = set(range(1, n+1)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding in descending order first (larger numbers are often safer)
        for cand in sorted(skipped, reverse=True):
            if time.time() > deadline:
                break
            if _try_add(S, cand, n):
                S.add(cand)
                skipped.discard(cand)
                improved = True
    return S

def _local_search_swap(S, n, deadline):
    """Try swaps: remove one element from S, add one from skipped, if net gain."""
    S = set(S)
    skipped = set(range(1, n+1)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(S)
        for rem in S_list:
            if time.time() > deadline:
                break
            S_temp = S - {rem}
            for add in sorted(skipped, reverse=True):
                if time.time() > deadline:
                    break
                if _try_add(S_temp, add, n):
                    S = S_temp | {add}
                    skipped.discard(add)
                    improved = True
                    break
    return S

def _construct_greedy(n, order, deadline):
    """Greedy construction based on a specific order of candidates."""
    S = set()
    for cand in order:
        if time.time() > deadline:
            break
        if _try_add(S, cand, n):
            S.add(cand)
    return S

def construct():
    start_time = time.time()
    total_deadline = start_time + 80  # Safety margin (evaluator kills at 90s)
    n = 256
    
    best_S = set()
    
    # Strategy 1: Descending order (Current behavior, usually good)
    S1 = _construct_greedy(n, range(n, 0, -1), total_deadline)
    S1 = _local_search_add(S1, n, total_deadline)
    S1 = _local_search_swap(S1, n, total_deadline)
    if len(S1) > len(best_S):
        best_S = S1
        
    # Strategy 2: Ascending order (Riskier, but might find different structures)
    if time.time() < total_deadline - 5:
        S2 = _construct_greedy(n, range(1, n+1), total_deadline)
        S2 = _local_search_add(S2, n, total_deadline)
        S2 = _local_search_swap(S2, n, total_deadline)
        if len(S2) > len(best_S):
            best_S = S2
            
    # Strategy 3: Randomized Greedy
    # We can run a few random restarts if time permits
    rng = random.Random(42)
    for trial in range(8):
        if time.time() > total_deadline - 5:  # Stop if close to deadline
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S_rand = _construct_greedy(n, order, total_deadline)
        S_rand = _local_search_add(S_rand, n, total_deadline)
        S_rand = _local_search_swap(S_rand, n, total_deadline)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    # Final Polish: Try swaps on best solution
    if time.time() < total_deadline - 2:
        S_polish = set(best_S)
        S_polish = _local_search_add(S_polish, n, total_deadline)
        S_polish = _local_search_swap(S_polish, n, total_deadline)
        if len(S_polish) > len(best_S):
            best_S = S_polish

    return sorted(best_S)
# EVOLVE-BLOCK-END
