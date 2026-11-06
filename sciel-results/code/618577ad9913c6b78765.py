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
    """Check if cand can be added to S."""
    # Case 1: cand plays role of 'a' (smallest in triple)
    # Need: no b,c in S with b > cand, c >= b, cand | (b+c)
    larger = [x for x in S if x > cand]
    for i in range(len(larger)):
        b = larger[i]
        for j in range(i, len(larger)):
            c = larger[j]
            if (b + c) % cand == 0:
                return False
    
    # Case 2: cand plays role of 'b' (middle in triple)
    # Need: no a,c in S with a < cand, c >= cand, a | (cand + c)
    smaller = [x for x in S if x < cand]
    geq = [x for x in S if x >= cand]
    for a in smaller:
        for c in geq:
            if (cand + c) % a == 0:
                return False

    # Case 3: cand plays role of 'c' (largest in triple)
    # Need: no a,b in S with a < b <= cand, a | (b + cand)
    for a in S:
        if a >= cand:
            continue
        for b in S:
            if a < b <= cand and (b + cand) % a == 0:
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
        # Try adding in ascending order? Might be slower, but catches cases where small numbers fit
        # But small numbers are risky. Let's stick to descending or random.
        # Let's try random order to escape local optima of the ordering
        # random.shuffle(list(skipped)) is not possible on set directly.
        # Let's just do descending.
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
    total_deadline = start_time + 85 # Safety margin
    n = 256
    
    best_S = set()
    
    # Strategy 1: Descending order (Current behavior, usually good)
    S1 = _construct_greedy(n, range(n, 0, -1), total_deadline)
    S1 = _local_search_add(S1, n, total_deadline)
    if len(S1) > len(best_S):
        best_S = S1
        
    # Strategy 2: Ascending order (Riskier, but might find different structures)
    if time.time() < total_deadline:
        S2 = _construct_greedy(n, range(1, n+1), total_deadline)
        S2 = _local_search_add(S2, n, total_deadline)
        if len(S2) > len(best_S):
            best_S = S2
            
    # Strategy 3: Randomized Greedy
    # We can run a few random restarts if time permits
    rng = random.Random(42)
    for _ in range(5):
        if time.time() > total_deadline - 5: # Stop if close to deadline
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S_rand = _construct_greedy(n, order, total_deadline)
        S_rand = _local_search_add(S_rand, n, total_deadline)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    # Final Polish: Try swaps
    # If we have time, try to swap elements to improve the set
    # This is computationally expensive, so we only do it if we have > 10s left
    if time.time() < total_deadline - 10:
        S_polish = set(best_S)
        skipped = set(range(1, n+1)) - S_polish
        swapped = True
        while swapped and time.time() < total_deadline - 2:
            swapped = False
            # Try to add any skipped element
            # If we can't, try swapping
            # For simplicity, let's just try to add again in random order
            for cand in list(skipped):
                if _try_add(S_polish, cand, n):
                    S_polish.add(cand)
                    skipped.discard(cand)
                    swapped = True
                    break
            # Try swaps: remove y, add x
            # This is O(|S| * |Skipped| * CostCheck). Might be too slow.
            # Let's stick to the add-only polish.
            
        if len(S_polish) > len(best_S):
            best_S = S_polish

    return sorted(best_S)
# EVOLVE-BLOCK-END
