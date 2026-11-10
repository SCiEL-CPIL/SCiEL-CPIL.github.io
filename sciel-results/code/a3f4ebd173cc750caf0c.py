"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin before 90s wall-clock limit

def _within_deadline():
    return (time.time() - _START_TIME) < _DEADLINE

# Precompute GCD table
_GCD = [[0] * 129 for _ in range(129)]
for i in range(1, 129):
    for j in range(1, 129):
        _GCD[i][j] = math.gcd(i, j)

def _bad_triple(a, b, c):
    """Return True if the triple (a,b,c) has all three pairwise gcds equal."""
    return _GCD[a][b] == _GCD[a][c] == _GCD[b][c]

def _is_valid_set(S):
    """Check if set S has no bad triples."""
    S_list = list(S)
    for a, b, c in combinations(S_list, 3):
        if _bad_triple(a, b, c):
            return False
    return True

def _can_add(x, S_list):
    """Check if x can be added to S_list without creating a bad triple.
    Optimized: Group elements in S_list by their gcd with x.
    For a group with gcd d, check if any pair (a,b) in the group has gcd(a,b) == d.
    """
    if len(S_list) < 2:
        return True
    
    groups = {}
    for s in S_list:
        d = _GCD[s][x]
        if d not in groups:
            groups[d] = []
        groups[d].append(s)
    
    for d, group in groups.items():
        if len(group) < 2:
            continue
        # Check pairs in this group
        # We need gcd(a, b) == d
        for i in range(len(group)):
            a = group[i]
            for j in range(i + 1, len(group)):
                b = group[j]
                if _GCD[a][b] == d:
                    return False
    return True

def _greedy_pass(order, S_init=None):
    """Greedy construction using a specific order."""
    S = list(S_init) if S_init else []
    S_set = set(S)
    for x in order:
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _local_search(S, max_iters=200):
    """Local search: try to remove an element and add others to increase size."""
    S = list(S)
    full = list(range(1, 129))
    best_size = len(S)
    
    for _ in range(max_iters):
        if not _within_deadline():
            break
        if len(S) >= 128:
            break
            
        improved = False
        # Try removing each element and then greedily adding back as many as possible
        for i in range(len(S)):
            u = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            # Candidates to add: all not in S_temp
            candidates = [x for x in full if x not in S_temp_set]
            random.shuffle(candidates)
            
            added_any = False
            for v in candidates:
                if _can_add(v, S_temp):
                    S_temp.append(v)
                    S_temp_set.add(v)
                    added_any = True
            
            if len(S_temp) > len(S):
                S = S_temp
                best_size = len(S)
                improved = True
                break
        
        if not improved:
            break
            
    return S

def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal."""
    random.seed(42)
    full = list(range(1, 129))
    best_set = None
    best_size = 0
    
    # Strategy 1: Multiple Randomized Greedy Passes
    # We run many passes with different orderings
    num_passes = 500
    for i in range(num_passes):
        if not _within_deadline():
            break
        
        order = list(full)
        random.shuffle(order)
        
        S = _greedy_pass(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Local Search on Best Set
    if best_set is not None and _within_deadline():
        # Run local search multiple times
        for _ in range(5):
            if not _within_deadline():
                break
            S_improved = _local_search(best_set, max_iters=100)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = S_improved
            else:
                break

    # Ensure nonempty and valid
    if best_set is None or len(best_set) == 0:
        return [1]
    
    # Final Validation
    if not _is_valid_set(best_set):
        # Fallback: Remove elements until valid
        S_list = list(best_set)
        while not _is_valid_set(S_list):
            found = False
            for a, b, c in combinations(S_list, 3):
                if _bad_triple(a, b, c):
                    S_list.remove(a)
                    found = True
                    break
            if not found:
                break
        
        if not S_list:
            return [1]
        best_set = S_list
        best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END
