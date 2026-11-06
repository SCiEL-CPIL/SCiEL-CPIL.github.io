import time
from itertools import combinations
from fractions import Fraction

def construct():
    deadline = time.time() + 80
    
    elems = list(range(1, 25))
    recip = {x: Fraction(1, x) for x in elems}
    
    # Precompute: for each x, all subsets of [1,24]\{x} that sum to 1/x
    x_to_subsets = {x: set() for x in elems}
    for x in elems:
        target = recip[x]
        others = [y for y in elems if y != x]
        n = len(others)
        max_size = min(n, 6)
        for size in range(1, max_size + 1):
            for subset in combinations(others, size):
                s = sum(recip[y] for y in subset)
                if s == target:
                    x_to_subsets[x].add(frozenset(subset))
    
    def is_valid(S):
        S_set = set(S)
        for x in S_set:
            for subset in x_to_subsets[x]:
                if subset <= S_set:
                    return False
        return True
    
    # Start with seed solution
    best = list(range(13, 25))
    best_size = len(best)
    
    # Greedy: try to add elements
    current = set(best)
    for x in sorted(elems, reverse=True):
        if x not in current:
            if is_valid(current | {x}):
                current.add(x)
                best_size = len(current)
                best = list(current)
    
    # Random restarts
    import random
    random.seed(42)
    for _ in range(20000):
        if time.time() > deadline:
            break
        order = elems[:]
        random.shuffle(order)
        S = set()
        for x in order:
            if is_valid(S | {x}):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)
    
    return best
