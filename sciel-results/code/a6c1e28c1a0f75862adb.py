# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 192
K = 3
MAX_CARD = 48

def _check_add(c, S):
    """Return True if adding c to set S (list) keeps validity.
    Check: (1) c does not divide sum of any 3 distinct elements of S.
           (2) for each a in S, no triple in (S\{a}) U {c} sums to multiple of a.
    """
    n = len(S)
    # (1) c divides sum of 3 distinct from S?
    # Optimization: if c > 3 * max(S), c can't divide any sum of 3 elements of S
    if c <= 3 * max(S) if S else True:
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                s_ij = si + S[j]
                for k in range(j+1, n):
                    if (s_ij + S[k]) % c == 0:
                        return False
    # (2) for each a in S, new triples containing c
    #     We need to check if (c + x + y) % a == 0 for any x, y in S, x!=a, y!=a, x!=y
    for a in S:
        for i in range(n):
            xi = S[i]
            if xi == a:
                continue
            for j in range(n):
                if j == i or S[j] == a:
                    continue
                if (c + xi + S[j]) % a == 0:
                    return False
    return True

def _is_valid_set(S):
    """Check if set S is fully valid."""
    for a in S:
        others = [x for x in S if x != a]
        for triple in itertools.combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _greedy_build(candidates, rng, deadline):
    """Build a set greedily from candidates in random order."""
    S = []
    for c in candidates:
        if time.time() > deadline:
            break
        if _check_add(c, S):
            S.append(c)
    return S

def _local_search(S, rng, deadline, max_iters=500):
    """Local search: try to add elements, or swap elements to improve."""
    S = list(S)
    S_set = set(S)
    available = [x for x in range(1, N+1) if x not in S_set]
    rng.shuffle(available)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        # Try to add an element
        for c in available:
            if c in S_set:
                continue
            if _check_add(c, S):
                S.append(c)
                S_set.add(c)
                available.remove(c)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one element, try to add a different one
        for i in range(len(S)):
            if time.time() > deadline:
                return S
            a = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            for c in available:
                if c in S_temp_set:
                    continue
                if _check_add(c, S_temp):
                    # Perform swap
                    S = S_temp + [c]
                    S_set = set(S)
                    if a not in available:
                        available.append(a)
                    if c in available:
                        available.remove(c)
                    improved = True
                    break
            
            if improved:
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: return well before 90s kill
    rng = random.Random(42)
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Multiple randomized greedy passes with different orderings
    # Try different biasings: prefer large numbers first, small first, random
    orderings = ['random', 'desc', 'asc']
    
    for pass_num in range(200):
        if time.time() > deadline:
            break
        
        # Choose ordering strategy
        if pass_num % 3 == 0:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif pass_num % 3 == 1:
            candidates = list(range(N, 0, -1))  # descending
        else:
            candidates = list(range(1, N + 1))  # ascending
        
        S = _greedy_build(candidates, rng, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Apply local search to improve
        if time.time() < deadline - 5.0:  # Need time for local search
            S_improved = _local_search(S, rng, deadline - 2.0, max_iters=200)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = list(S_improved)
        
        # Early termination if we hit the max possible cardinality
        if best_size >= MAX_CARD:
            break
    
    # Final local search on best set
    if time.time() < deadline - 3.0 and best_size < MAX_CARD:
        S_final = _local_search(best_set, rng, deadline - 1.0, max_iters=300)
        if len(S_final) > best_size:
            best_set = list(S_final)
            best_size = len(best_set)
    
    return best_set
# EVOLVE-BLOCK-END
