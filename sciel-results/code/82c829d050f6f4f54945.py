"""Rewritten program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _precompute_gcd_matrix():
    """Precompute gcd for all pairs in [1,128]."""
    n = 129
    g = [[0]*n for _ in range(n)]
    for i in range(1, n):
        for j in range(i, n):
            g[i][j] = g[j][i] = gcd(i, j)
    return g

def _is_bad_triple(a, b, c, g):
    """Check if (a,b,c) is a bad triple."""
    return g[a][b] == g[a][c] == g[b][c]

def _greedy_build(g):
    """Greedy: add elements in order if they don't create a bad triple."""
    selected = []
    for x in range(1, 129):
        ok = True
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if _is_bad_triple(selected[i], selected[j], x, g):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def _check_validity(s, g):
    """Full validity check for set s."""
    s_list = sorted(s)
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            for k in range(j+1, len(s_list)):
                if _is_bad_triple(s_list[i], s_list[j], s_list[k], g):
                    return False
    return True

def _try_add(s, x, g):
    """Check if x can be added to s without creating a bad triple."""
    s_list = sorted(s)
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            if _is_bad_triple(s_list[i], s_list[j], x, g):
                return False
    return True

def _local_search(s, g, max_iters=3000, deadline=None):
    """Local search with 1-swap and 1-add moves."""
    s = set(s)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try 1-add: add one element
        added = False
        for x in range(1, 129):
            if x not in s and _try_add(s, x, g):
                s.add(x)
                added = True
                break
        
        if added:
            continue
        
        # Try 1-swap: remove one, add one
        improved = False
        s_list = sorted(s)
        for rem in s_list:
            for add in range(1, 129):
                if add in s:
                    continue
                new_s = s - {rem} | {add}
                # Check validity: only need to check triples involving add
                # (rem is gone, so triples with rem are fine)
                # Triples: (add, a, b) where a,b in new_s - {add}
                others = sorted(new_s - {add})
                valid = True
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        if _is_bad_triple(add, others[i], others[j], g):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    s = new_s
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    g = _precompute_gcd_matrix()
    
    # Greedy initial solution
    best_set = set(_greedy_build(g))
    best_size = len(best_set)
    
    # Local search from greedy
    s = _local_search(best_set, g, max_iters=5000, deadline=deadline)
    if len(s) > best_size:
        best_set = s
        best_size = len(s)
    
    # Multiple restarts with random perturbations
    while time.time() < deadline:
        k = random.randint(1, 4)
        s = set(best_set)
        to_remove = random.sample(sorted(s), min(k, len(s)))
        s -= set(to_remove)
        candidates = [x for x in range(1, 129) if x not in s]
        if candidates:
            to_add = random.sample(candidates, min(k, len(candidates)))
            s |= set(to_add)
        
        s = _local_search(s, g, max_iters=2000, deadline=deadline)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        
        if time.time() > deadline:
            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
