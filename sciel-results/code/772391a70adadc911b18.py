"""Rewritten program: hypergraph independent set via randomized greedy + vectorized local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 192
K = 3
MAX_CARDS = 48
TIME_LIMIT = 55.0  # safety margin below 60s

def _check_valid(selected_arr):
    """Check if selected_arr (numpy array of ints) is valid.
    No element in selected_arr divides the sum of any 3 distinct OTHER elements."""
    n = len(selected_arr)
    if n < 4:
        return True
    # For each element a, check all triples of other elements
    # a does not divide (b+c+d) for any distinct b,c,d in selected\{a}
    for i in range(n):
        a = selected_arr[i]
        others = np.delete(selected_arr, i)
        if len(others) < 3:
            continue
        # Generate all triples from others
        # Vectorized: compute all triple sums and check divisibility
        # For efficiency with small n, use combinations but vectorize the sum check
        # Actually for n up to ~48, combinations gives C(47,3) ~ 16215 per element, total ~ 777k
        # Let's use a smarter vectorized approach:
        # For a fixed a, we need: no triple (b,c,d) from others with (b+c+d) % a == 0
        # This is equivalent to: for all distinct b,c,d in others, (b+c+d) % a != 0
        # 
        # Vectorized approach: compute all pair sums, then for each third element check
        # Actually, let's just do it with numpy for the inner loop:
        # Generate all index triples
        m = len(others)
        if m < 3:
            continue
        # Use meshgrid-style for triples - but this is O(m^3) memory which is bad for m=47
        # m^3 = 103823, that's fine
        i_idx = np.repeat(np.arange(m), m * m)
        j_idx = np.tile(np.repeat(np.arange(m), m), m)
        k_idx = np.tile(np.arange(m), m * m)
        # Filter distinct indices
        mask = (i_idx != j_idx) & (j_idx != k_idx) & (i_idx != k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        triple_sums = others[i_idx] + others[j_idx] + others[k_idx]
        if np.any(triple_sums % a == 0):
            return False
    return True

def _check_valid_fast(selected_list):
    """Faster check using precomputed approach for a list of selected integers."""
    n = len(selected_list)
    if n < 4:
        return True
    sel = np.array(selected_list, dtype=np.int64)
    for i in range(n):
        a = sel[i]
        others = np.delete(sel, i)
        m = len(others)
        if m < 3:
            continue
        # Compute all triple sums vectorized
        # Use broadcasting: others[:, None, None] + others[None, :, None] + others[None, None, :]
        # But filter distinct indices
        # For m up to 47, this creates a 47x47x47 array = 103823 elements, fine
        sums = others[:, None, None] + others[None, :, None] + others[None, None, :]
        # Mask out non-distinct indices
        idx = np.arange(m)
        distinct_mask = (idx[:, None, None] != idx[None, :, None]) & \
                        (idx[None, :, None] != idx[None, None, :]) & \
                        (idx[:, None, None] != idx[None, None, :])
        triple_sums = sums[distinct_mask]
        if np.any(triple_sums % a == 0):
            return False
    return True

def _greedy_construct(order, deadline):
    """Greedy: add elements in given order if valid."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        candidate = selected + [x]
        if _check_valid_fast(candidate):
            selected.append(x)
    return selected

def _local_search(selected, deadline, rng):
    """Local search: try swaps to increase cardinality or maintain."""
    selected = list(selected)
    n = N
    remaining = set(range(1, n + 1)) - set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try 1-in swaps: remove one, add one
        for i in range(len(selected)):
            if time.time() > deadline:
                return selected
            for x in list(remaining):
                candidate = selected[:i] + selected[i+1:] + [x]
                if _check_valid_fast(candidate):
                    # Only accept if it doesn't reduce size (it maintains size)
                    # We want to increase size, so this is neutral
                    # But it might enable future additions
                    selected = candidate
                    remaining = set(range(1, n + 1)) - set(selected)
                    improved = True
                    break
            if improved:
                break
        # Try adding an element
        for x in sorted(remaining, reverse=True):
            if time.time() > deadline:
                return selected
            candidate = selected + [x]
            if _check_valid_fast(candidate):
                selected = candidate
                remaining = set(range(1, n + 1)) - set(selected)
                improved = True
                break
    return selected

def _try_random_greedy(seed, deadline):
    """Randomized greedy with random order."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = _greedy_construct(order, deadline)
    selected = _local_search(selected, deadline, rng)
    return selected

def _try_degree_greedy(seed, deadline):
    """Greedy based on degree (number of conflicts) heuristic."""
    rng = random.Random(seed)
    # Precompute for each number x, how many triples (a,b,c) from 1..N have a|b+c+d where x is in the triple
    # Simpler: just use random order with some bias
    order = list(range(1, N + 1))
    # Sort by value descending with some randomization
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    selected = _greedy_construct(order, deadline)
    selected = _local_search(selected, deadline, rng)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best = []
    
    # Strategy 1: Greedy from large to small (baseline)
    order_large = list(range(N, 0, -1))
    selected = _greedy_construct(order_large, deadline)
    if len(selected) > len(best):
        best = list(selected)
    
    # Strategy 2: Multiple random greedy attempts
    for attempt in range(50):
        if time.time() > deadline:
            break
        seed = attempt * 7919 + 12345
        selected = _try_random_greedy(seed, deadline)
        if len(selected) > len(best):
            best = list(selected)
    
    # Strategy 3: Degree-based greedy
    for attempt in range(20):
        if time.time() > deadline:
            break
        seed = attempt * 104729 + 98765
        selected = _try_degree_greedy(seed, deadline)
        if len(selected) > len(best):
            best = list(selected)
    
    # Final local search on best
    if time.time() < deadline - 2:
        best = _local_search(best, deadline, random.Random(42))
    
    # Ensure we return at most MAX_CARDS elements
    if len(best) > MAX_CARDS:
        best = best[:MAX_CARDS]
    
    return best
# EVOLVE-BLOCK-END
