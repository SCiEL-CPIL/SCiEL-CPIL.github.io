import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _is_valid(S):
    """
    Check if set S is valid.
    S is a list of distinct integers.
    """
    n = len(S)
    # If n < 4, no 3 distinct others exist for any element?
    # If n=3, for element a, others are 2. Can't pick 3 distinct others.
    # So n < 4 is always valid.
    if n < 4:
        return True
        
    S_set = set(S)
    # Optimization: Precompute sums? 
    # For each a in S, check if any 3 distinct others sum to multiple of a.
    for a in S:
        others = [x for x in S if x != a]
        # We need to check if there exist u, v, w in others such that (u+v+w) % a == 0
        # If a is large, u+v+w is bounded.
        # Max sum of 3 from 1..192 is 190+191+192 = 573.
        # If a > 573, always valid. But a <= 192.
        # So we check multiples k*a <= 573.
        
        # To speed up, we can check residues.
        # But n is small (<=48). O(n^3) is 48^3 ~ 100k. 48 * 100k ~ 5M ops.
        # In Python, this might be slow if done repeatedly.
        
        # Better: For each a, iterate 3-combinations of others.
        # 48 choose 3 is ~ 17000.
        # 48 * 17000 = 816,000 checks.
        # Each check is sum % a.
        # This is feasible for a few iterations, but not for thousands.
        
        # We will use this for final validation.
        # For incremental checks in search, we can be smarter.
        
        found = False
        # Iterate combinations of 3 from others
        # Using itertools.combinations is implemented in C, so it's fast.
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                found = True
                break
        if found:
            return False
            
    return True

def _is_valid_addition(S, x):
    """
    Check if adding x to valid set S keeps it valid.
    S is a list. x is not in S.
    """
    n = len(S)
    
    # Check 1: x does not divide sum of any 3 in S
    if n >= 3:
        for t in combinations(S, 3):
            if sum(t) % x == 0:
                return False
            
    # Check 2: For each a in S, a does not divide x + u + v for distinct u, v in S \ {a}
    if n >= 3: # Need 2 others from S\{a} to combine with x. Total 3.
        # Wait, "3 DISTINCT OTHER".
        # If we add x, the set is S U {x}.
        # For a in S, others are (S \ {a}) U {x}.
        # We need 3 distinct from these.
        # Cases:
        # 1. 3 from S\{a}. (Already valid in S)
        # 2. 2 from S\{a}, 1 is x. Sum = x + u + v.
        # 3. 1 from S\{a}, 2 is x? No, x is unique.
        
        # So we check if a divides x + u + v for distinct u, v in S \ {a}.
        for a in S:
            others = [v for v in S if v != a]
            # We need distinct u, v in others such that (x + u + v) % a == 0
            # (u + v) % a == (-x) % a
            target = (-x) % a
            # Use a set of residues to speed up?
            # Or just double loop. n is small.
            found = False
            m = len(others)
            for i in range(m):
                u = others[i]
                need = (target - u) % a
                # Check if any v in others[j] (j!=i) has v % a == need
                # We can precompute residues of others?
                # For simplicity, loop.
                for j in range(m):
                    if i == j: continue
                    if others[j] % a == need:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
                
    return True

def construct():
    # Start with a greedy set from large to small
    S = []
    for x in range(_N, 0, -1):
        if _is_valid_addition(S, x):
            S.append(x)
            if len(S) == _MAX:
                break
        if time.time() > _DEADLINE - 5:
            break
            
    # If we reached max, return
    if len(S) == _MAX:
        return S
        
    # Local search to improve size
    best_S = list(S)
    start = time.time()
    
    all_vals = set(range(1, _N + 1))
    
    while time.time() - start < 30 and time.time() < _DEADLINE - 5:
        improved = False
        
        # Try insertion
        if len(S) < _MAX:
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            for x in candidates[:20]:
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > len(best_S):
                        best_S = list(S)
                    break
                    
        # Try swap
        if not improved and len(S) > 0:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            
            for a in to_remove[:5]:
                for x in candidates[:5]:
                    S_minus_a = [v for v in S if v != a]
                    if _is_valid_addition(S_minus_a, x):
                        S = S_minus_a + [x]
                        improved = True
                        break
                if improved:
                    break
                    
        if not improved:
            # Perturb
            if len(S) > 10:
                to_remove = random.sample(S, 3)
                S = [v for v in S if v not in to_remove]
            else:
                S = []
                for x in range(_N, 0, -1):
                    if _is_valid_addition(S, x):
                        S.append(x)
                        if len(S) == _MAX: break

    return best_S
