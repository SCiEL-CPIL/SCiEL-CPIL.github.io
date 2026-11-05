# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAX_CARD = 48

def _is_valid_add(c, S):
    """
    Check if c can be added to set S (list of ints).
    1. c must not divide sum of any 3 distinct elements in S.
    2. For each a in S, a must not divide (c + x + y) for any distinct x, y in S\{a}.
    """
    n = len(S)
    S_set = set(S)
    
    # Check 1: c divides x+y+z for x,y,z in S, x!=y!=z
    # Use set-based approach: for each pair (x,y), check if there exists z in S
    # such that (x+y+z) % c == 0
    # i.e., z ≡ -(x+y) (mod c), and z in S, z != x, z != y
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            # We need z in S such that (x + y + z) % c == 0
            # z ≡ -(x+y) mod c
            # But z must be in [1, 192] and in S
            # Since z <= 192, we can check: for each possible residue r where r ≡ -(x+y) mod c
            # and r in [1,192], check if r in S_set and r != x and r != y
            # But this is still potentially slow. Let's just iterate.
            # Actually, let's precompute for efficiency
            rem = (-x - y) % c
            # z must satisfy z ≡ rem (mod c) and z in S
            # z = rem, rem + c, rem + 2c, ... up to 192
            # But we can just check: is there a z in S with z % c == rem?
            # Precompute a dict: for each value in S, what's its mod c?
            # Actually, let's just do the triple loop but break early
            for k in range(j + 1, n):
                z = S[k]
                if (x + y + z) % c == 0:
                    return False
    
    # Check 2: For each a in S, check if a divides c + x + y for x, y in S\{a}
    # x, y distinct, both != a
    for idx_a, a in enumerate(S):
        # We need x, y in S, x != a, y != a, x != y
        # such that (c + x + y) % a == 0
        # i.e., x + y ≡ -c (mod a)
        # For each x in S\{a}, check if there exists y in S\{a}, y != x
        # such that (x + y) % a == (-c) % a
        target = (-c) % a
        # Build set of values in S excluding a
        S_minus_a = S_set - {a}
        for i in range(n):
            x = S[i]
            if x == a:
                continue
            # Need y in S_minus_a, y != x, such that (x + y) % a == target
            # y ≡ target - x (mod a)
            rem = (target - x) % a
            # y must be in S_minus_a and y % a == rem
            # Check if any y in S_minus_a satisfies this
            # We can iterate
            for j in range(n):
                y = S[j]
                if y == a or y == x:
                    continue
                if (x + y) % a == target:
                    return False
                    
    return True

def _is_valid_set(S):
    """Check if the entire set S is valid."""
    S_list = list(S)
    n = len(S_list)
    S_set = set(S_list)
    
    for i, c in enumerate(S_list):
        # Check 1: c divides sum of any 3 distinct OTHER elements
        for j in range(n):
            if j == i:
                continue
            x = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                y = S_list[k]
                for l in range(n):
                    if l == i or l == j or l == k:
                        continue
                    z = S_list[l]
                    if (x + y + z) % c == 0:
                        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Strategy 1: Descending Greedy with optimized check
    selected = []
    candidates = list(range(N, 0, -1))
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= MAX_CARD:
            break
            
        if _is_valid_add(c, selected):
            selected.append(c)
            
    best_set = list(selected)
    best_size = len(best_set)
    
    # Strategy 2: Try to improve via local search
    # Try swapping one element out and adding two others
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        
        # Try multiple random restarts with biased ordering
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Create a biased ordering: mix of descending and random
            cand = list(range(1, N + 1))
            # Bias towards larger numbers
            cand.sort(key=lambda x: (-x, rng.random()))
            
            S = []
            for c in cand:
                if len(S) >= MAX_CARD:
                    break
                if _is_valid_add(c, S):
                    S.append(c)
                
            if len(S) > best_size:
                best_set = list(S)
                best_size = len(S)
                
            # Local search: try to add more elements
            if time.time() > deadline - 5.0:
                break
                
            # Try adding any remaining element that fits
            remaining = [c for c in range(1, N + 1) if c not in S]
            rng.shuffle(remaining)
            for c in remaining:
                if len(S) >= MAX_CARD:
                    break
                if _is_valid_add(c, S):
                    S.append(c)
                    
            if len(S) > best_size:
                best_set = list(S)
                best_size = len(S)
    
    return best_set
# EVOLVE-BLOCK-END
