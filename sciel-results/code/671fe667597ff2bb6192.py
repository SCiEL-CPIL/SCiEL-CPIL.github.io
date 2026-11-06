"""Rewritten program: greedy + local-search for max equal-gcd-triple-free subset of [1,128]."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = 80.0  # safety margin under 90s kill

def _has_bad_triple(a, b, c):
    return gcd(a, b) == gcd(a, c) == gcd(b, c)

def _is_valid_addition(selected_set, selected_list, x):
    # Check all pairs in selected_list against x
    for i in range(len(selected_list)):
        a = selected_list[i]
        for j in range(i + 1, len(selected_list)):
            b = selected_list[j]
            if _has_bad_triple(a, b, x):
                return False
    return True

def _greedy(order):
    selected = []
    selected_set = set()
    for x in order:
        if _is_valid_addition(selected_set, selected, x):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected):
    selected_set = set(selected)
    excluded = [x for x in range(1, 129) if x not in selected_set]
    improved = True
    while improved:
        if time.time() > _DEADLINE:
            break
        improved = False
        # Try adding each excluded element
        for x in excluded:
            if time.time() > _DEADLINE:
                break
            if _is_valid_addition(selected_set, selected, x):
                selected.append(x)
                selected_set.add(x)
                excluded.remove(x)
                improved = True
                break
        if improved:
            continue
        # Try swaps: remove one, add two? Just try remove-one-add-one first for simplicity
        # Actually try: for each in selected, remove it, then see if we can add more
        best_swap = None
        for idx, a in enumerate(selected):
            if time.time() > _DEADLINE:
                break
            # temporarily remove a
            new_sel = selected[:idx] + selected[idx+1:]
            new_set = selected_set - {a}
            # try to add excluded elements greedily
            added = []
            for x in excluded:
                if x == a:
                    continue
                if _is_valid_addition(new_set, new_sel, x):
                    new_sel.append(x)
                    new_set.add(x)
                    added.append(x)
            if len(added) >= 1 and len(new_sel) > len(selected):
                best_swap = (new_sel, new_set)
                break
        if best_swap:
            selected, selected_set = best_swap
            excluded = [x for x in range(1, 129) if x not in selected_set]
            improved = True
    return selected

def construct():
    if time.time() > _DEADLINE:
        return [1]
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: (x.bit_length(), x)),
        sorted(range(1, 129), key=lambda x: (-x.bit_length(), x)),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
    ]
    
    best = []
    for order in orders:
        if time.time() > _DEADLINE - 10:
            break
        cand = _greedy(order)
        if len(cand) > len(best):
            best = cand
    
    # Local search on best
    best = _local_search(best)
    
    # Try local search on a few other greedy results
    for order in orders[:3]:
        if time.time() > _DEADLINE - 5:
            break
        cand = _greedy(order)
        cand = _local_search(cand)
        if len(cand) > len(best):
            best = cand
    
    if not best:
        best = [1]
    return best
# EVOLVE-BLOCK-END
