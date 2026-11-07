import time
import random
import sys

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _has_bad_triple(S):
    """Return True if S contains distinct a,b,c with a+b,a+c,b+c all in S."""
    L = sorted(S)
    n = len(L)
    Sset = set(L)
    for i in range(n):
        a = L[i]
        for j in range(i+1, n):
            b = L[j]
            s_ab = a + b
            if s_ab not in Sset:
                continue
            for k in range(j+1, n):
                c = L[k]
                if (a+c) in Sset and (b+c) in Sset:
                    return True
    return False

def _validate(S):
    return not _has_bad_triple(set(S))

def _fast_check_add(S, x):
    """Check if adding x to S creates a bad triple. S is a set, x not in S."""
    # A new bad triple must include x.
    # For each pair (p,q) in S with p<q:
    #   If x < p: triple {x,p,q}. Need x+p, x+q, p+q all in S∪{x}.
    #     Since p,q > x, x+p, x+q, p+q > x, so they must be in S.
    #   If p < x < q: triple {p,x,q}. Need p+x, p+q, x+q all in S∪{x}.
    #     p+x, x+q > x, so must be in S. p+q could be anything.
    #   If p < q < x: triple {p,q,x}. Need p+q, p+x, q+x all in S∪{x}.
    #     p+x, q+x > x, so must be in S. p+q could be anything.
    
    L = sorted(S)
    Sset = S
    for i in range(len(L)):
        p = L[i]
        for j in range(i+1, len(L)):
            q = L[j]
            if x < p:
                # triple {x, p, q}
                if (x+p) in Sset and (x+q) in Sset and (p+q) in Sset:
                    return False
            elif p < x < q:
                # triple {p, x, q}
                if (p+x) in Sset and (x+q) in Sset and (p+q) in Sset:
                    return False
            elif p < q < x:
                # triple {p, q, x}
                if (p+q) in Sset and (p+x) in Sset and (q+x) in Sset:
                    return False
    return True

def _greedy_complete(S, deadline):
    """Try to add elements from 1..128 not in S in a greedy manner."""
    S = set(S)
    candidates = [i for i in range(1, N+1) if i not in S]
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        if _fast_check_add(S, x):
            S.add(x)
    return S

def _simulated_annealing(seed, deadline):
    """Simulated annealing to find a large valid set."""
    rng = random.Random(seed)
    
    # Start with a random valid set
    S = set()
    # Try to build an initial set greedily with random order
    candidates = list(range(1, N+1))
    rng.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        if _fast_check_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Annealing parameters
    T0 = 1.0
    T_end = 0.01
    cooling_rate = 0.95
    
    T = T0
    while time.time() < deadline and T > T_end:
        T *= cooling_rate
        if time.time() > deadline:
            break
        
        # Try to add a random element
        missing = [i for i in range(1, N+1) if i not in S]
        if not missing:
            break
        
        x = rng.choice(missing)
        
        if _fast_check_add(S, x):
            S.add(x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            # With probability exp(-1/T), accept the invalid state temporarily
            # But we can't actually have an invalid state, so we skip
            # Instead, we try to remove an element and add another
            if rng.random() < 0.5:
                # Try a swap: remove one element, add another
                if S:
                    to_remove = rng.choice(list(S))
                    S.remove(to_remove)
                    # Now try to add x
                    if _fast_check_add(S, x):
                        S.add(x)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                    else:
                        # Restore
                        S.add(to_remove)
    
    # Final greedy completion
    best_S = _greedy_complete(best_S, deadline)
    return best_S

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # Best known valid set: [65, 128] is sum-free, size 64
    best = set(range(65, 129))
    best_size = len(best)
    
    # Try simulated annealing from multiple random starts
    for seed in range(50):
        if time.time() > deadline - 2:
            break
        try:
            S = _simulated_annealing(seed, deadline)
            if _validate(S) and len(S) > best_size:
                best = S
                best_size = len(S)
        except:
            pass
    
    # Final greedy pass on best
    best = _greedy_complete(best, deadline)
    
    if not _validate(best):
        # Emergency fallback
        best = set(range(65, 129))
    
    return list(best)
