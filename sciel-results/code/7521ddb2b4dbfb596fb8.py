"""Improved program for large integer sets avoiding equal-lcm triples."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import lcm
from functools import lru_cache

@lru_cache(maxsize=None)
def _lcm(a, b):
    return lcm(a, b)

def _has_bad_triple(selected, x):
    """Check if adding x creates a bad triple with any pair in selected."""
    for i in range(len(selected)):
        a = selected[i]
        la = _lcm(a, x)
        for j in range(i + 1, len(selected)):
            b = selected[j]
            if _lcm(a, b) == la and _lcm(b, x) == la:
                return True
    return False

def _greedy(order, selected=None):
    """Greedy: add elements in given order if compatible."""
    if selected is None:
        selected = []
    for x in order:
        if x in selected:
            continue
        if not _has_bad_triple(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to add more elements, and try swaps to improve."""
    selected = list(selected)
    remaining = [x for x in range(1, 129) if x not in selected]
    
    # Try to add any remaining element
    changed = True
    while changed:
        changed = False
        for x in remaining:
            if time.time() > deadline:
                return selected
            if not _has_bad_triple(selected, x):
                selected.append(x)
                remaining.remove(x)
                changed = True
    return selected

def _try_swap(selected, remaining, deadline):
    """Try removing one element and adding two others."""
    best = list(selected)
    for i, rem in enumerate(remaining):
        if time.time() > deadline:
            break
        for j in range(len(selected)):
            if time.time() > deadline:
                break
            candidate = selected[:j] + selected[j+1:] + [rem]
            # Try to fill with another element
            for add in remaining:
                if add == rem:
                    continue
                if not _has_bad_triple(candidate, add):
                    new_set = candidate + [add]
                    if len(new_set) > len(best):
                        best = new_set
    return best

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    # Multiple orderings
    orders = [
        list(range(1, 129)),           # ascending
        list(range(128, 0, -1)),       # descending
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],  # odd first
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],  # even first
        sorted(range(1, 129), key=lambda x: x.bit_count(), reverse=True),  # more bits first
        sorted(range(1, 129), key=lambda x: x.bit_count()),                 # fewer bits first
        # Powers of 2 first
        [2**k for k in range(7) if 2**k <= 128] + [x for x in range(1, 129) if x & (x-1) != 0],
        # Primes first
        [x for x in range(2, 129) if all(x % p != 0 for p in range(2, int(x**0.5)+1) if p*p <= x)] + 
        [x for x in range(1, 129) if x == 1 or not all(x % p != 0 for p in range(2, int(x**0.5)+1) if p*p <= x)],
    ]
    
    best = []
    for order in orders:
        if time.time() > deadline:
            break
        candidate = _greedy(order)
        candidate = _local_search(candidate, deadline)
        if len(candidate) > len(best):
            best = candidate
    
    # Try to improve with swaps
    remaining = [x for x in range(1, 129) if x not in best]
    best = _try_swap(best, remaining, deadline)
    
    # Final pass
    best = _local_search(best, deadline)
    
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
