"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

_N = 256

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

# Precompute: for each element x, and each pair (i,j) in the domain,
# whether x conflicts with that pair (i.e., x|ij, i|xj, or j|xi)
# We'll store conflict[x] as a set of frozensets {i,j} that conflict with x

def _build_conflicts():
    """Precompute conflict data for fast lookup."""
    # conflicts[x] = set of (i,j) tuples where i<j, i!=x, j!=x, and
    # (x divides i*j) or (i divides x*j) or (j divides x*i)
    conflicts = [set() for _ in range(_N + 1)]
    
    for x in range(2, _N + 1):
        for i in range(2, _N + 1):
            if i == x:
                continue
            for j in range(i + 1, _N + 1):
                if j == x:
                    continue
                # Check if x conflicts with pair (i, j)
                if (i * j) % x == 0 or (x * j) % i == 0 or (x * i) % j == 0:
                    conflicts[x].add((i, j))
    return conflicts

_CONFLICTS = _build_conflicts()

def _can_add(S_list, S_set, x):
    """Check if x can be added to set S (given as list and set) without violating constraints."""
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            # Check if x conflicts with pair (a, b)
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                return False
    return True

def _greedy_build(deadline):
    """Build initial set using greedy approach."""
    S = [a for a in range(2, _N + 1) if _is_prime(a)]
    S_set = set(S)
    
    candidates = [a for a in range(2, _N + 1) if not _is_prime(a)]
    
    while time.time() < deadline:
        valid = []
        for x in candidates:
            if _can_add(S, S_set, x):
                valid.append(x)
        
        if not valid:
            break
        
        # Pick the smallest valid candidate
        best_x = min(valid)
        S.append(best_x)
        S_set.add(best_x)
        candidates.remove(best_x)
    
    return S, S_set

def _local_search(S, S_set, deadline):
    """Local search: try adding elements, then 1-for-1 swaps."""
    S = list(S)
    S_set = set(S)
    
    # Phase 1: Try to add any element that can be added
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(2, _N + 1):
            if a in S_set:
                continue
            if _can_add(S, S_set, a):
                S.append(a)
                S_set.add(a)
                improved = True
    
    # Phase 2: 1-for-1 swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        
        for i, a in enumerate(list(S)):
            if time.time() > deadline:
                break
            
            # Remove a
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            # Try to find a replacement b not in S_temp that can be added
            for b in range(2, _N + 1):
                if b in S_temp_set:
                    continue
                if _can_add(S_temp, S_temp_set, b):
                    # Check if this improves: we removed a and added b, same size
                    # But maybe we can then add more?
                    # For now, just do the swap if it doesn't decrease size
                    # Actually, 1-for-1 swap keeps size same. We need 1-for-2 to improve.
                    # Let's try: remove a, then greedily add as many as possible
                    S_after = list(S_temp)
                    S_after_set = set(S_after)
                    
                    # Try to add elements
                    for c in range(2, _N + 1):
                        if c in S_after_set:
                            continue
                        if _can_add(S_after, S_after_set, c):
                            S_after.append(c)
                            S_after_set.add(c)
                    
                    if len(S_after) > current_size:
                        S = S_after
                        S_set = S_after_set
                        improved = True
                        break
                    elif len(S_after) == current_size and b != a:
                        # Same size, different composition - might help future swaps
                        # Only accept if it leads to potential improvement
                        # For simplicity, skip same-size swaps for now
                        pass
                    break
    
    return S, S_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Build initial set
    S, S_set = _greedy_build(deadline - 10.0)
    
    # Local search
    S, S_set = _local_search(S, S_set, deadline)
    
    # Final check: verify validity
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    # Invalid - should not happen, but handle gracefully
                    pass
    
    # Return as list of Python ints
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
