"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def _is_sidon(s):
    """Check if set s is a Sidon set (all pairwise sums a+b with a<=b are distinct)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _get_sums_set(s):
    """Compute the set of all pairwise sums a+b for a<=b in s."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sums.add(s[i] + s[j])
    return sums

def _is_maximal(s, N, sums_set):
    """
    Check if Sidon set s is inclusion-maximal in [1,N].
    A missing element x is blocked if adding x would create a sum collision.
    """
    s_set = set(s)
    for x in range(1, N + 1):
        if x in s_set:
            continue
        # If 2x is already in sums, x is blocked
        if 2 * x in sums_set:
            continue
        # If x+a is in sums for some a in s, x is blocked
        blocked = False
        for a in s:
            if (x + a) in sums_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _build_conflict_graph(N):
    """
    Build a conflict graph where an edge (u,v) exists if u and v cannot both
    be in a Sidon set. Two numbers u,v conflict if 2u == 2v (impossible for distinct)
    or if there exist a,b in some Sidon context... Actually, the simplest conflict:
    u and v conflict if 2u = u+v (i.e., u=v) or if u+v creates issues.
    
    Better approach: u and v directly conflict if {u,v} is not Sidon, which happens
    only if 2u = 2v or u+v = u+u or u+v = v+v, i.e., u=v. So no direct pairwise
    conflicts for distinct elements. The conflicts are higher-order.
    
    Instead, let's use a different model: think of it as finding a small set S such that
    every x not in S is "blocked" by S. We'll use a greedy dominating set approach
    on the blocking structure.
    """
    pass

def _greedy_construct_with_blocking(N, random_order):
    """
    Greedy construction that prioritizes elements that block many others.
    """
    selected = []
    sums = set()
    s_set = set()
    
    # Precompute for each x, what sums it would create
    # and which other elements it blocks
    
    for x in random_order:
        if x in s_set:
            continue
        
        # Check if x can be added (Sidon property)
        new_sums = [2 * x]
        for a in selected:
            new_sums.append(x + a)
        
        # Check internal collisions
        if len(set(new_sums)) != len(new_sums):
            continue
        # Check collision with existing sums
        if sums.intersection(new_sums):
            continue
        
        # Compute how many unblocked elements this x would block
        # An element y (not in s_set, y != x) is blocked by x if:
        # - 2y is in (sums ∪ new_sums) OR
        # - y+a is in (sums ∪ new_sums) for some a in (selected ∪ {x})
        
        all_sums_after = sums | set(new_sums)
        all_elements_after = s_set | {x}
        
        block_count = 0
        for y in range(1, N + 1):
            if y in all_elements_after:
                continue
            # Check if y would be blocked
            if 2 * y in all_sums_after:
                block_count += 1
                continue
            for a in all_elements_after:
                if (y + a) in all_sums_after:
                    block_count += 1
                    break
        
        # Heuristic: prefer elements that block more
        # But also consider: we want SMALL set, so prefer high blocking power
        selected.append(x)
        sums.update(new_sums)
        s_set.add(x)
    
    return selected

def _minimize_by_removal(s, N):
    """
    Try to remove elements while maintaining maximality.
    """
    s = list(s)
    changed = True
    while changed:
        changed = False
        for i in range(len(s) - 1, -1, -1):
            candidate = s[:i] + s[i+1:]
            if not candidate:
                break
            cand_sums = _get_sums_set(candidate)
            if _is_maximal(candidate, N, cand_sums):
                s = candidate
                changed = True
                break
    return s

def _local_search_swap(s, N, iterations=50):
    """
    Try to swap elements to find a smaller maximal set.
    Actually, swaps keep size same. We need removals.
    Let's try: remove one, add none, check if still maximal.
    That's what minimize does. For local search, try different removal orders.
    """
    s = list(s)
    best = s[:]
    best_size = len(s)
    
    for _ in range(iterations):
        # Try random removal order
        trial = s[:]
        indices = list(range(len(trial)))
        random.shuffle(indices)
        
        for idx in indices:
            if not trial:
                break
            # Find current position of the element at original index idx
            # This is tricky because list shrinks. Let's just try removing each element
            # in the current list in random order.
            pos = random.randint(0, len(trial) - 1)
            candidate = trial[:pos] + trial[pos+1:]
            if not candidate:
                continue
            cand_sums = _get_sums_set(candidate)
            if _is_maximal(candidate, N, cand_sums):
                trial = candidate
        
        if len(trial) < best_size:
            best = trial[:]
            best_size = len(trial)
    
    return best

def _construct_via_graph_dominating(N):
    """
    Model as: find small set S such that every x not in S is blocked by S.
    Use greedy set cover on the blocking relation.
    
    For each candidate element x, define B(x) = set of y that x helps block.
    We want to cover all elements not in S with blocking from S.
    But S also must be Sidon.
    """
    # First, build a valid Sidon set using greedy
    order = list(range(1, N + 1))
    random.shuffle(order)
    s = _greedy_construct_with_blocking(N, order)
    
    # Now minimize
    s = _minimize_by_removal(s, N)
    
    # Local search with different random orders
    best = s
    best_size = len(s)
    
    for _ in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        s2 = _greedy_construct_with_blocking(N, order)
        s2 = _minimize_by_removal(s2, N)
        if len(s2) < best_size:
            best = s2
            best_size = len(s2)
    
    return best

def construct():
    best_size = N + 1
    best_set = None
    start_time = time.time()
    time_limit = 28.0
    
    random.seed(42)
    
    # Strategy: multiple random greedy constructions with blocking heuristic,
    # then aggressive minimization and local search.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        s = _greedy_construct_with_blocking(N, order)
        
        if not s:
            continue
        
        # Quick check: if already smaller than best, keep it
        if len(s) < best_size:
            # Minimize it
            s_min = _minimize_by_removal(s, N)
            if len(s_min) < best_size:
                best_size = len(s_min)
                best_set = s_min[:]
        
        # Also try local search on current best
        if best_set and time.time() - start_time < time_limit - 3.0:
            improved = _local_search_swap(best_set, N, iterations=10)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    if best_set is None:
        # Fallback: simple greedy
        order = list(range(1, N + 1))
        s = _greedy_construct_with_blocking(N, order)
        return s if s else [1]
    
    return best_set
# EVOLVE-BLOCK-END
