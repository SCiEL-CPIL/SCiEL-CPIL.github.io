"""Improved program: precompute bad triples, use multiple greedy strategies + local search."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _precompute_bad_triples():
    """Precompute all triples (a,b,c) with 1<=a<b<c<=128 where lcm(a,b)=lcm(a,c)=lcm(b,c)."""
    bad = set()
    for a in range(1, 129):
        for b in range(a+1, 129):
            lab = _lcm(a, b)
            for c in range(b+1, 129):
                if _lcm(a, c) == lab and _lcm(b, c) == lab:
                    bad.add((a, b, c))
    return bad

def _greedy(order, bad_triples_set):
    """Greedy: add numbers in given order if no bad triple is formed."""
    selected = set()
    for x in order:
        ok = True
        for a, b in combinations(selected, 2):
            if (a, b, x) in bad_triples_set or (a, x, b) in bad_triples_set or (x, a, b) in bad_triples_set:
                # Check if any permutation forms a bad triple
                triple = tuple(sorted((a, b, x)))
                if triple in bad_triples_set:
                    ok = False
                    break
        if ok:
            selected.add(x)
    return selected

def _local_search(selected, bad_triples_set, deadline):
    """Try to improve by swapping elements out/in."""
    selected = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(selected):
            for y in range(1, 129):
                if y in selected:
                    continue
                # Try replacing x with y
                new_set = (selected - {x}) | {y}
                # Check validity
                valid = True
                for a, b, c in combinations(new_set, 3):
                    triple = tuple(sorted((a, b, c)))
                    if triple in bad_triples_set:
                        valid = False
                        break
                if valid and len(new_set) > len(selected):
                    selected = new_set
                    improved = True
                    break
            if improved:
                break
    return selected

def construct():
    deadline = time.time() + 70  # safety margin
    
    # Precompute bad triples
    bad_triples_set = _precompute_bad_triples()
    
    if time.time() > deadline:
        return [1]
    
    # Try multiple greedy orderings
    best = set()
    
    # Natural order
    s1 = _greedy(range(1, 129), bad_triples_set)
    best = max(best, s1, key=len)
    
    # Reverse order
    s2 = _greedy(range(128, 0, -1), bad_triples_set)
    best = max(best, s2, key=len)
    
    # Random orderings
    import random
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 129))
        random.shuffle(order)
        s = _greedy(order, bad_triples_set)
        best = max(best, s, key=len)
    
    # Local search on best
    if time.time() < deadline:
        best = _local_search(best, bad_triples_set, deadline)
    
    return list(best)
# EVOLVE-BLOCK-END
