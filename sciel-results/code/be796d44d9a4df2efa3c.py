import time
import random
from itertools import combinations

def precompute_divisor_table():
    """
    For each a in 1..192, precompute which sums s (2..384) are divisible by a.
    Returns a list where div_table[a] is a set of s values divisible by a.
    """
    div_table = [set() for _ in range(193)]
    for a in range(2, 193):
        for s in range(a, 385, a):
            div_table[a].add(s)
    # a=1 divides everything
    div_table[1] = set(range(2, 385))
    return div_table

_DIV_TABLE = precompute_divisor_table()

def is_valid_incremental(S, x):
    """
    Check if S U {x} is valid, given S is valid.
    S is a list of distinct ints. x is not in S.
    Returns True if valid, False otherwise.
    """
    n = len(S)
    
    # 1. Check if x divides sum of any 3 distinct in S
    if n >= 3:
        div_set_x = _DIV_TABLE[x]
        # Use early termination
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                sj = S[j]
                s_ij = si + sj
                for k in range(j+1, n):
                    if (s_ij + S[k]) in div_set_x:
                        return False
    
    # 2. Check if any a in S divides x + y + z for y,z in S\{a}
    for i in range(n):
        a = S[i]
        if a == 1:
            # 1 divides everything, so we need NO 3 distinct other elements
            if n - 1 >= 3:
                return False
            continue
        
        div_set_a = _DIV_TABLE[a]
        # For each pair (y,z) in S\{a}, check if (x + y + z) in div_set_a
        for j in range(n):
            if j == i:
                continue
            y = S[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                z = S[k]
                if (x + y + z) in div_set_a:
                    return False
                    
    return True

def validate_set(S):
    """
    Fully validate a set S.
    """
    S_set = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        others = [x for x in S if x != a]
        if len(others) < 3:
            continue
        # Check if a divides sum of any 3 distinct in others
        div_set_a = _DIV_TABLE[a]
        for i2 in range(len(others)):
            o1 = others[i2]
            for i3 in range(i2+1, len(others)):
                o2 = others[i3]
                s_o12 = o1 + o2
                for i4 in range(i3+1, len(others)):
                    o3 = others[i4]
                    if (s_o12 + o3) in div_set_a:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Greedy Phase with multiple orderings
    orderings = [
        list(range(192, 0, -1)),  # Descending
        list(range(1, 193)),      # Ascending
    ]
    
    for ordering in orderings:
        if time.time() > deadline: break
        S = []
        for x in ordering:
            if time.time() > deadline: break
            if len(S) == 48: break
            if is_valid_incremental(S, x):
                S.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        if best_size == 48:
            return best_set

    # Randomized Greedy with fewer seeds to save time
    rng = random.Random(42)
    for trial in range(5):  # Reduced from 30 to 5
        if time.time() > deadline: break
        S = []
        candidates = list(range(1, 193))
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline: break
            if len(S) == 48: break
            if is_valid_incremental(S, x):
                S.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        if best_size == 48:
            return best_set

    # Local Search Phase
    current_set = best_set[:]
    current_size = best_size
    S_set = set(current_set)
    
    no_improve_count = 0
    max_no_improve = 20  # Reduced iterations
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Try to add
        missing = [x for x in range(1, 193) if x not in S_set]
        if missing:
            missing.sort(reverse=True)
            for x in missing:
                if time.time() > deadline: break
                if len(current_set) == 48: break
                if is_valid_incremental(current_set, x):
                    current_set.append(x)
                    S_set.add(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                    break
                    
        if improved:
            no_improve_count = 0
            if best_size == 48:
                return best_set
            continue
            
        # 2. Try swaps
        if current_set and missing:
            missing_set = set(missing)
            found_swap = False
            for _ in range(100):  # Reduced from 300 to 100
                if time.time() > deadline: break
                
                a = random.choice(current_set)
                b = random.choice(missing)
                
                S_minus_a = [v for v in current_set if v != a]
                
                if is_valid_incremental(S_minus_a, b):
                    current_set = S_minus_a + [b]
                    S_set.discard(a)
                    S_set.add(b)
                    found_swap = True
                    break
                    
            if found_swap:
                improved = True
                no_improve_count = 0
                continue
        
        if not improved:
            no_improve_count += 1
            if no_improve_count % 10 == 0:
                current_set = best_set[:]
                current_size = best_size
                S_set = set(current_set)
                
    # Final Validation
    if best_set and not validate_set(best_set):
        # If invalid, try to salvage by removing elements
        # Or just return a known small valid set
        # For now, return empty or smallest valid
        # We can try to build a small valid set greedily
        small_set = []
        for x in range(192, 0, -1):
            if time.time() > deadline: break
            if is_valid_incremental(small_set, x):
                small_set.append(x)
        best_set = small_set
        best_size = len(best_set)
        
    return best_set
