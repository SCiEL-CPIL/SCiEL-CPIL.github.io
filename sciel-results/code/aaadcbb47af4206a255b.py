"""B2[2] set maximization in 1..1000 via model-and-optimize: local search + restarts."""
import time
import random
import sys

DEADLINE = 145.0  # safety margin before 150s kill

def _counts_from_set(S):
    """Compute sum counts for set S (a<=b)."""
    counts = {}
    L = sorted(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i, n):
            b = L[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return counts

def _try_add(S, counts, x):
    """Return new counts if adding x keeps all <=2, else None. x not in S."""
    if x in S:
        return None
    L = sorted(S)
    new_counts = dict(counts)
    for y in L:
        s = x + y
        if new_counts.get(s, 0) + 1 > 2:
            return None
        new_counts[s] = new_counts.get(s, 0) + 1
    s2 = 2 * x
    if new_counts.get(s2, 0) + 1 > 2:
        return None
    new_counts[s2] = new_counts.get(s2, 0) + 1
    return new_counts

def _remove(S, counts, x):
    """Remove x from S, return new counts."""
    L = sorted(S)
    new_counts = dict(counts)
    for y in L:
        if y == x:
            continue
        s = x + y
        new_counts[s] -= 1
        if new_counts[s] == 0:
            del new_counts[s]
    s2 = 2 * x
    new_counts[s2] -= 1
    if new_counts[s2] == 0:
        del new_counts[s2]
    return new_counts

def _greedy_seed():
    chosen = []
    counts = {}
    for x in range(1, 1001):
        # quick check
        ok = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts.get(s2, 0) + 1 > 2:
                ok = False
        if ok:
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s2 = 2 * x
            counts[s2] = counts.get(s2, 0) + 1
            chosen.append(x)
    return chosen, counts

def _local_search(S, counts, time_limit):
    """Improve feasible set via Add/Swap moves (Hill Climbing)."""
    S = list(S)
    counts = dict(counts)
    best_S = list(S)
    best_counts = dict(counts)
    best_size = len(S)
    end = time.time() + time_limit
    S_set = set(S)
    
    while time.time() < end:
        improved = False
        
        # Phase 1: Try to add any element (smallest first)
        for x in range(1, 1001):
            if x in S_set:
                continue
            new_counts = _try_add(S, counts, x)
            if new_counts is not None:
                S.append(x)
                S_set.add(x)
                counts = new_counts
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                    best_counts = dict(counts)
        
        if not improved:
            # Phase 2: Try Swaps
            swap_found = False
            for x in range(1, 1001):
                if x in S_set:
                    continue
                
                # Determine which y in S blocks x
                conflicts = []
                for y in S:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        conflicts.append(y)
                
                if not conflicts:
                    continue
                
                # Try removing one of the conflicting y's
                y = conflicts[0]
                
                # Perform Swap: Remove y, Add x
                counts_after_rem = _remove(S, counts, y)
                S_tmp = [z for z in S if z != y]
                S_set_tmp = S_set - {y}
                
                new_counts = _try_add(S_tmp, counts_after_rem, x)
                if new_counts is not None:
                    # Swap successful
                    S = S_tmp + [x]
                    S_set = S_set_tmp | {x}
                    counts = new_counts
                    swap_found = True
                    improved = True
                    break # Restart Phase 1 with new state
            
            if not swap_found and not improved:
                # Stuck in local minimum. Try a random perturbation
                if S:
                    y = random.choice(S)
                    counts = _remove(S, counts, y)
                    S.remove(y)
                    S_set.discard(y)

    return best_S, best_counts

def construct():
    start = time.time()
    rng = random.Random(42)
    
    # Greedy seed
    S, counts = _greedy_seed()
    best_S = list(S)
    best_counts = dict(counts)
    best_size = len(S)
    
    # Local search on greedy
    S, counts = _local_search(S, counts, time_limit=10.0)
    if len(S) > best_size:
        best_S, best_counts, best_size = list(S), dict(counts), len(S)
    
    # Random restarts: start from random feasible-ish sets via randomized greedy
    while time.time() < DEADLINE - 20.0:
        # Randomized greedy
        S_r = []
        counts_r = {}
        order = list(range(1, 1001))
        rng.shuffle(order)
        for x in order:
            ok = True
            for y in S_r:
                s = x + y
                if counts_r.get(s, 0) + 1 > 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * x
                if counts_r.get(s2, 0) + 1 > 2:
                    ok = False
            if ok:
                for y in S_r:
                    s = x + y
                    counts_r[s] = counts_r.get(s, 0) + 1
                s2 = 2 * x
                counts_r[s2] = counts_r.get(s2, 0) + 1
                S_r.append(x)
        
        # Local search
        S_l, counts_l = _local_search(S_r, counts_r, time_limit=15.0)
        if len(S_l) > best_size:
            best_S, best_counts, best_size = list(S_l), dict(counts_l), len(S_l)
        
        # Check deadline
        if time.time() > DEADLINE - 5.0:
            break
    
    # Final local search on best
    S_f, counts_f = _local_search(best_S, best_counts, time_limit=DEADLINE - 2.0 - time.time())
    if len(S_f) > best_size:
        best_S, best_counts, best_size = list(S_f), dict(counts_f), len(S_f)
    
    return best_S
