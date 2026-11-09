import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _precompute_bad_triples():
    """Precompute all bad triples in [1,128]."""
    elements = list(range(1, 129))
    bad_triples = []
    for i in range(len(elements)):
        a = elements[i]
        for j in range(i + 1, len(elements)):
            b = elements[j]
            ab_lcm = _lcm(a, b)
            for k in range(j + 1, len(elements)):
                c = elements[k]
                if _lcm(a, c) == ab_lcm and _lcm(b, c) == ab_lcm:
                    bad_triples.append((a, b, c))
    return bad_triples

def _greedy_hitting_set(bad_triples, deadline=None, random_seed=None):
    """
    Greedy hitting set: repeatedly pick the element in the most remaining bad triples,
    remove it, and eliminate all triples containing it.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Build incidence: element -> list of triple indices
    triple_set = set()
    for t in bad_triples:
        triple_set.add(t)
    
    active_triples = set(triple_set)
    removed = set()
    
    while active_triples:
        if deadline and time.time() > deadline:
            break
        
        # Count how many active triples each element appears in
        counts = {}
        for t in active_triples:
            for x in t:
                counts[x] = counts.get(x, 0) + 1
        
        if not counts:
            break
        
        # Pick element with max count; break ties randomly
        max_count = max(counts.values())
        candidates = [x for x, c in counts.items() if c == max_count]
        
        if len(candidates) > 1:
            chosen = random.choice(candidates)
        else:
            chosen = candidates[0]
        
        removed.add(chosen)
        
        # Remove all active triples containing chosen
        new_active = set()
        for t in active_triples:
            if chosen not in t:
                new_active.add(t)
        active_triples = new_active
    
    # The solution is [1,128] minus removed
    solution = set(range(1, 129)) - removed
    return solution

def _local_search_add(S, deadline=None):
    """Try to add elements back to S that don't create bad triples."""
    S = set(S)
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for x in range(1, 129):
            if x in S:
                continue
            # Check if adding x creates any bad triple
            S_list = sorted(S)
            conflict = False
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    if _is_bad_triple(x, a, b):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                S.add(x)
                improved = True
                break
    return S

def _local_search_swap(S, max_iters=200, deadline=None):
    """Try swaps to improve S."""
    S = set(S)
    best_size = len(S)
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        for a in sorted(S):
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                S_list = sorted(S)
                conflict = False
                for i in range(len(S_list)):
                    aa = S_list[i]
                    for j in range(i + 1, len(S_list)):
                        bb = S_list[j]
                        if _is_bad_triple(x, aa, bb):
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        improved = True
                    break
            if not improved:
                S.add(a)
        if not improved:
            break
    return S

def _validate_and_fix(S):
    """Ensure no bad triples remain; remove elements to fix."""
    S = set(S)
    S_list = sorted(S)
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            S.discard(c)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Precompute bad triples
    bad_triples = _precompute_bad_triples()
    
    best_set = set()
    best_size = 0
    
    # Multiple greedy hitting set runs with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        S = _greedy_hitting_set(bad_triples, deadline=deadline, random_seed=seed)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search: try to add back elements
    if time.time() < deadline:
        improved = _local_search_add(best_set, deadline=deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Local search: try swaps
    if time.time() < deadline:
        improved = _local_search_swap(best_set, max_iters=100, deadline=deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Final validation
    best_set = _validate_and_fix(best_set)
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
