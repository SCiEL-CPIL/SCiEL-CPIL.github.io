import time
import random
import math
import sys

# EVOLVE-BLOCK-START

N = 256

# Precompute all 3-APs and 4-APs in [1, N]
# A 3-AP is (a, b, c) such that a+c = 2b.
# A 4-AP is (a, b, c, d) such that a+d = b+c = 2(a+d-... no, a, a+d, a+2d, a+3d).

def _precompute_aps():
    """
    Precompute:
    - all_3aps: list of tuples (a, b, c)
    - all_4aps: list of tuples (a, b, c, d)
    - ap3_by_elem: dict mapping element -> list of 3-APs it is part of
    - ap4_by_elem: dict mapping element -> list of 4-APs it is part of
    """
    all_3aps = []
    all_4aps = []
    
    # Generate 3-APs
    # Middle b, difference d
    for b in range(1, N + 1):
        max_d = min(b - 1, N - b)
        for d in range(1, max_d + 1):
            a = b - d
            c = b + d
            all_3aps.append((a, b, c))
            
    # Generate 4-APs
    # Start a, difference d
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_4aps.append((a, b, c, e))
            
    return all_3aps, all_4aps

ALL_3APS, ALL_4APS = _precompute_aps()

# Build adjacency structures for fast delta calculation
# For each element x, store the 3-APs it belongs to, categorized by its role (0: first, 1: mid, 2: last)
# Role 0: x is 'a' in (a, b, c) -> need b, c in set
# Role 1: x is 'b' in (a, b, c) -> need a, c in set
# Role 2: x is 'c' in (a, b, c) -> need a, b in set

AP3_BY_ELEM = {i: [[], [], []] for i in range(1, N + 1)}
for (a, b, c) in ALL_3APS:
    AP3_BY_ELEM[a][0].append((b, c))
    AP3_BY_ELEM[b][1].append((a, c))
    AP3_BY_ELEM[c][2].append((a, b))

# For each element x, store the 4-APs it belongs to, categorized by its role (0: 1st, 1: 2nd, 2: 3rd, 3: 4th)
# Role 0: x is 'a' in (a, b, c, d) -> need b, c, d in set
# Role 1: x is 'b' in (a, b, c, d) -> need a, c, d in set
# Role 2: x is 'c' in (a, b, c, d) -> need a, b, d in set
# Role 3: x is 'd' in (a, b, c, d) -> need a, b, c in set

AP4_BY_ELEM = {i: [[], [], [], []] for i in range(1, N + 1)}
for (a, b, c, d) in ALL_4APS:
    AP4_BY_ELEM[a][0].append((b, c, d))
    AP4_BY_ELEM[b][1].append((a, c, d))
    AP4_BY_ELEM[c][2].append((a, b, d))
    AP4_BY_ELEM[d][3].append((a, b, c))

def _count_3aps(set_A):
    """Count 3-APs in a set."""
    count = 0
    for a, b, c in ALL_3APS:
        if a in set_A and b in set_A and c in set_A:
            count += 1
    return count

def _has_4ap(set_A):
    """Check if set contains a 4-AP."""
    for a, b, c, d in ALL_4APS:
        if a in set_A and b in set_A and c in set_A and d in set_A:
            return True
    return False

def _delta_add_3aps(set_A, x):
    """Calculate increase in 3-AP count if x is added to set_A."""
    delta = 0
    # x as first (a): need b, c in A
    for b, c in AP3_BY_ELEM[x][0]:
        if b in set_A and c in set_A:
            delta += 1
    # x as middle (b): need a, c in A
    for a, c in AP3_BY_ELEM[x][1]:
        if a in set_A and c in set_A:
            delta += 1
    # x as last (c): need a, b in A
    for a, b in AP3_BY_ELEM[x][2]:
        if a in set_A and b in set_A:
            delta += 1
    return delta

def _delta_remove_3aps(set_A, x):
    """Calculate decrease in 3-AP count if x is removed from set_A."""
    # Same logic as add, because the 3-APs involving x are exactly those that will be broken
    return _delta_add_3aps(set_A, x)

def _delta_add_4aps(set_A, x):
    """Calculate increase in 4-AP count if x is added to set_A."""
    delta = 0
    # x as 1st (a): need b, c, d in A
    for b, c, d in AP4_BY_ELEM[x][0]:
        if b in set_A and c in set_A and d in set_A:
            delta += 1
    # x as 2nd (b): need a, c, d in A
    for a, c, d in AP4_BY_ELEM[x][1]:
        if a in set_A and c in set_A and d in set_A:
            delta += 1
    # x as 3rd (c): need a, b, d in A
    for a, b, d in AP4_BY_ELEM[x][2]:
        if a in set_A and b in set_A and d in set_A:
            delta += 1
    # x as 4th (d): need a, b, c in A
    for a, b, c in AP4_BY_ELEM[x][3]:
        if a in set_A and b in set_A and c in set_A:
            delta += 1
    return delta

def _delta_remove_4aps(set_A, x):
    """Calculate decrease in 4-AP count if x is removed from set_A."""
    return _delta_add_4aps(set_A, x)

def _greedy_construct():
    """Greedy construction: add elements in order if they don't create a 4-AP."""
    A = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        # We can just check if any 4-AP containing x has other 3 elements in A
        creates_4ap = False
        # Optimization: use precomputed AP4_BY_ELEM
        for role in range(4):
            for others in AP4_BY_ELEM[x][role]:
                if all(o in A for o in others):
                    creates_4ap = True
                    break
            if creates_4ap:
                break
        if not creates_4ap:
            A.add(x)
    return A

def _random_greedy(seed_val):
    """Randomized greedy construction."""
    rng = random.Random(seed_val)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    A = set()
    for x in order:
        creates_4ap = False
        for role in range(4):
            for others in AP4_BY_ELEM[x][role]:
                if all(o in A for o in others):
                    creates_4ap = True
                    break
            if creates_4ap:
                break
        if not creates_4ap:
            A.add(x)
    return A

def _simulated_annealing(initial_set, time_limit=60.0, seed_val=42):
    """
    Simulated Annealing to maximize 3-APs while keeping 4-AP count 0.
    We allow temporary 4-APs with a penalty.
    Score = 3AP_count - PENALTY * 4AP_count
    """
    rng = random.Random(seed_val)
    start_time = time.time()
    
    A = set(initial_set)
    current_3aps = _count_3aps(A)
    current_4aps = 0 # Should be 0 for valid initial set
    
    # Verify initial set
    if _has_4ap(A):
         # Fallback to a known valid set if initial is bad
         A = _greedy_construct()
         current_3aps = _count_3aps(A)
         current_4aps = 0

    best_A = set(A)
    best_score = current_3aps
    best_3aps = current_3aps
    
    T = 1000.0
    cooling_rate = 0.999
    penalty = 10000.0 # High penalty to discourage 4-APs
    
    current_score = current_3aps - penalty * current_4aps
    
    while time.time() - start_time < time_limit:
        # Decay temperature
        T *= cooling_rate
        if T < 0.01:
            break
            
        # Propose move
        move_type = rng.random()
        new_A = A
        new_3aps = current_3aps
        new_4aps = current_4aps
        
        if move_type < 0.4:
            # Add a random element
            x = rng.randint(1, N)
            if x not in A:
                new_A = A | {x}
                new_3aps = current_3aps + _delta_add_3aps(A, x)
                new_4aps = current_4aps + _delta_add_4aps(A, x)
        elif move_type < 0.7:
            # Remove a random element
            if A:
                x = rng.choice(list(A))
                new_A = A - {x}
                new_3aps = current_3aps - _delta_remove_3aps(A, x)
                new_4aps = current_4aps - _delta_remove_4aps(A, x)
        else:
            # Swap: remove x, add y
            if A:
                x = rng.choice(list(A))
                y = rng.randint(1, N)
                if y not in A and x != y:
                    # Remove x, then add y
                    # Delta 3APs: - delta_remove(x) + delta_add(y, A \ {x})
                    # Delta 4APs: - delta_remove(x) + delta_add(y, A \ {x})
                    # Note: delta_add(y, A\{x}) needs to be calculated on A\{x}
                    # For simplicity and speed, we can just calculate deltas on A and adjust if x,y interact?
                    # Actually, x and y might be in the same AP.
                    # Let's just calculate on A\{x} for add, and A for remove.
                    
                    A_temp = A - {x}
                    d3_rem = _delta_remove_3aps(A, x)
                    d4_rem = _delta_remove_4aps(A, x)
                    
                    d3_add = _delta_add_3aps(A_temp, y)
                    d4_add = _delta_add_4aps(A_temp, y)
                    
                    new_3aps = current_3aps - d3_rem + d3_add
                    new_4aps = current_4aps - d4_rem + d4_add
                    new_A = A_temp | {y}

        new_score = new_3aps - penalty * new_4aps
        
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score > 0 or (T > 0 and rng.random() < math.exp(delta_score / T)):
            A = new_A
            current_3aps = new_3aps
            current_4aps = new_4aps
            current_score = new_score
            
            # Update best if valid
            if current_4aps == 0:
                if current_3aps > best_3aps:
                    best_3aps = current_3aps
                    best_A = set(A)
                    
    return list(best_A)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # 1. Generate initial solutions
    best_set = _greedy_construct()
    best_score = _count_3aps(best_set)
    
    # Try a few random greedy starts
    seeds = [1, 2, 3, 4, 5]
    for s in seeds:
        if time.time() > deadline - 30:
            break
        cand_set = _random_greedy(s)
        cand_score = _count_3aps(cand_set)
        if cand_score > best_score:
            best_score = cand_score
            best_set = cand_set
            
    # 2. Run Simulated Annealing from the best initial solution
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        # Split time for multiple SA runs if possible, or just one long one?
        # One long SA is usually better for convergence.
        # Let's do one SA run for most of the time.
        sa_start = time.time()
        sa_limit = remaining_time - 1.0
        improved_set = _simulated_annealing(best_set, time_limit=sa_limit, seed_val=12345)
        
        # Verify and update
        if improved_set:
            imp_score = _count_3aps(improved_set)
            if imp_score > best_score:
                best_score = imp_score
                best_set = improved_set
        
    # 3. Final Verification
    # Ensure no 4-APs
    if _has_4ap(best_set):
        # Fallback to greedy
        best_set = _greedy_construct()
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
