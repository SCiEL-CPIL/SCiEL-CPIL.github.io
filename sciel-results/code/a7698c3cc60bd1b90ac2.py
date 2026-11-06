"""Improved program: vectorized conflict checking + effective local search with swaps."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

N = 192
K = 3
MAX_CARD = 48

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i in range(n):
        a = sel[i]
        others = [sel[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add_fast(selected, x):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    # Check x against existing triples
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            return False
    # Check each existing element a against pairs + x
    for i in range(n):
        a = selected[i]
        for j in range(n):
            if j == i:
                continue
            s1 = selected[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                s2 = selected[k]
                if (s1 + s2 + x) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Greedy construction with given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x):
            selected.append(x)
    return selected

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts and structured orderings."""
    best = []
    start_time = time.time()
    
    # Try several random orderings
    for attempt in range(30):
        if time.time() > deadline:
            break
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        result = _greedy(rand_order, deadline)
        if len(result) > len(best):
            best = result
    
    # Structured orderings
    structured_orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(96, 0, -1)) + list(range(97, N + 1)),
        [x for x in range(N, 0, -1) if x > 64],
        [x for x in range(N, 0, -1) if x % 2 == 1],
        [x for x in range(N, 0, -1) if x > 48],
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best

def _local_improve(selected, deadline):
    """Local search: try to add more elements, then try 1-for-2 swaps."""
    selected = list(selected)
    sel_set = set(selected)
    candidates = [x for x in range(1, N + 1) if x not in sel_set]
    candidates.sort(reverse=True)
    
    # Phase A: Greedy add
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x not in sel_set:
                if _can_add_fast(selected, x):
                    selected.append(x)
                    sel_set.add(x)
                    candidates.remove(x)
                    improved = True
    
    # Phase B: Try 1-for-2 swaps (remove one, add two)
    # This can increase cardinality by 1
    swap_improved = True
    while swap_improved and time.time() < deadline:
        swap_improved = False
        sel_set = set(selected)
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            remaining = selected[:i] + selected[i+1:]
            rem_set = set(remaining)
            
            # Find all candidates that can be added to remaining
            addable = []
            for x in range(1, N + 1):
                if x in rem_set or x == removed:
                    continue
                if _can_add_fast(remaining, x):
                    addable.append(x)
            
            # Try to find two elements from addable that can both be added
            for idx_a in range(len(addable)):
                if time.time() > deadline:
                    break
                a = addable[idx_a]
                for idx_b in range(idx_a + 1, len(addable)):
                    b = addable[idx_b]
                    if _can_add_fast(remaining + [a], b):
                        # Success: remove one, add two
                        new_selected = remaining + [a, b]
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            sel_set = set(selected)
                            swap_improved = True
                            break
                if swap_improved:
                    break
            if swap_improved:
                break
    
    return selected

def _weighted_greedy(deadline):
    """Greedy that prioritizes elements with fewer conflicts (prefer large numbers)."""
    order = list(range(N, 0, -1))
    return _greedy(order, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin before 90s limit
    
    # Phase 1: Multiple greedy strategies
    best = []
    
    # Weighted greedy (prefer large numbers)
    result = _weighted_greedy(deadline)
    if len(result) > len(best):
        best = result
    
    # Randomized greedy
    result = _random_greedy(deadline)
    if len(result) > len(best):
        best = result
    
    # Phase 2: Local improvement with swaps
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: More local improvement rounds
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 4: Try additional random restarts if time allows
    if time.time() < deadline - 10:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            rand_order = list(range(1, N + 1))
            random.shuffle(rand_order)
            result = _greedy(rand_order, deadline - 5)
            if len(result) > len(best):
                best = result
            if time.time() < deadline - 5:
                improved = _local_improve(best, deadline - 5)
                if len(improved) > len(best):
                    best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy(list(range(N, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END
