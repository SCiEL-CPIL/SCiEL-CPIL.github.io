"""
Maximum Independent Set in a 4-AP conflict graph for [1, 512].

We model the problem as finding the largest subset S of {1,...,512} with no
4-term arithmetic progression.  This is a Maximum Independent Set (MIS) problem
on a graph where vertices are 1..512 and edges connect pairs that would
complete a 4-AP.  In fact, the constraint is on quadruples, so we use a
conflict hypergraph, but for branch-and-bound we can use the following approach:

Actually, the 4-AP constraint is: there should be no a, d such that
a, a+d, a+2d, a+3d are all in S (with d != 0).

A practical approach: use randomized greedy + local search, but also try a
branch-and-bound with bitset-based pruning.

Given the 90-second budget, I'll implement:
1. A fast greedy construction (randomized, multiple restarts)
2. Local search (add/remove) to improve
3. A branch-and-bound with strong upper bounds using the structure of 4-APs

For the branch-and-bound, the key insight is that 4-APs are structured. We can
use the fact that the maximum size of a 4-AP-free set in [1,N] is known to be
O(N / sqrt(log N)) for large N, giving us an upper bound.

Actually, for N=512, known results suggest the maximum is around 100-150.

Let me implement a robust randomized approach with local search:
"""

import time
import random
import numpy as np

N = 512
K = 4

def has_4ap(selected_set, x):
    """Check if adding x to selected_set creates a 4-AP."""
    # Check all 4-APs containing x
    # x can be at position 0, 1, 2, or 3 in the 4-AP
    for d in range(1, N + 1):
        # x at position 0: x, x+d, x+2d, x+3d
        if x + d <= N and x + 2 * d <= N and x + 3 * d <= N:
            if x + d in selected_set and x + 2 * d in selected_set and x + 3 * d in selected_set:
                return True
        # x at position 1: x-d, x, x+d, x+2d
        if x - d >= 1 and x + d <= N and x + 2 * d <= N:
            if x - d in selected_set and x + d in selected_set and x + 2 * d in selected_set:
                return True
        # x at position 2: x-2d, x-d, x, x+d
        if x - 2 * d >= 1 and x - d >= 1 and x + d <= N:
            if x - 2 * d in selected_set and x - d in selected_set and x + d in selected_set:
                return True
        # x at position 3: x-3d, x-2d, x-d, x
        if x - 3 * d >= 1 and x - 2 * d >= 1 and x - d >= 1:
            if x - 3 * d in selected_set and x - 2 * d in selected_set and x - d in selected_set:
                return True
    return False

def has_4ap_bitset(selected_bits, x):
    """Check if adding x to selected (bitset) creates a 4-AP. Uses bit operations."""
    # x is 1-indexed, so bit position x-1
    # Check all d
    for d in range(1, N + 1):
        # x at pos 0: need bits at x+d-1, x+2d-1, x+3d-1
        if x + 3 * d <= N:
            i1 = x + d - 1
            i2 = x + 2 * d - 1
            i3 = x + 3 * d - 1
            if selected_bits[i1] and selected_bits[i2] and selected_bits[i3]:
                return True
        # x at pos 1: need bits at x-d-1, x+d-1, x+2d-1
        if x - d >= 1 and x + 2 * d <= N:
            i0 = x - d - 1
            i2 = x + d - 1
            i3 = x + 2 * d - 1
            if selected_bits[i0] and selected_bits[i2] and selected_bits[i3]:
                return True
        # x at pos 2: need bits at x-2d-1, x-d-1, x+d-1
        if x - 2 * d >= 1 and x + d <= N:
            i0 = x - 2 * d - 1
            i1 = x - d - 1
            i3 = x + d - 1
            if selected_bits[i0] and selected_bits[i1] and selected_bits[i3]:
                return True
        # x at pos 3: need bits at x-3d-1, x-2d-1, x-d-1
        if x - 3 * d >= 1:
            i0 = x - 3 * d - 1
            i1 = x - 2 * d - 1
            i2 = x - d - 1
            if selected_bits[i0] and selected_bits[i1] and selected_bits[i2]:
                return True
    return False

def greedy_randomized(deadline):
    """Randomized greedy: pick elements in random order, add if no 4-AP is formed."""
    selected = [False] * (N + 1)
    remaining = list(range(1, N + 1))
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        if not has_4ap_bitset(selected, x):
            selected[x] = True
    
    return selected

def local_search_improve(selected, deadline, max_iterations=5000):
    """Try to add elements or swap to improve the set."""
    selected = list(selected)
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing element
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not selected[x]:
                if not has_4ap_bitset(selected, x):
                    selected[x] = True
                    improved = True
        
        # Try swaps: remove one, add two
        if time.time() < deadline:
            for x_remove in range(1, N + 1):
                if time.time() > deadline:
                    break
                if selected[x_remove]:
                    # Try removing x_remove and adding two others
                    selected[x_remove] = False
                    original = list(selected)
                    added = []
                    for x_add in range(1, N + 1):
                        if time.time() > deadline:
                            break
                        if not original[x_add]:
                            if not has_4ap_bitset(selected, x_add):
                                selected[x_add] = True
                                added.append(x_add)
                    if len(added) >= 2:
                        improved = True
                    else:
                        # Revert
                        for x in range(1, N + 1):
                            selected[x] = original[x]
    
    return selected

def base3_construction():
    """Classic base-3 construction for 4-AP-free sets."""
    # Numbers with no digit 2 in base 3 up to N
    result = set()
    for x in range(1, N + 1):
        y = x
        ok = True
        while y > 0:
            if y % 3 == 2:
                ok = False
                break
            y //= 3
        if ok:
            result.add(x)
    return sorted(result)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # Start with base-3 construction as initial solution
    best = base3_construction()
    best_size = len(best)
    
    # Try randomized greedy with multiple restarts
    for trial in range(50):
        if time.time() > deadline:
            break
        
        seed = trial
        rng = random.Random(seed)
        # Temporarily set global random state
        old_state = random.getstate()
        random.seed(seed)
        
        selected = greedy_randomized(deadline)
        size = sum(selected)
        
        # Local search improvement
        selected = local_search_improve(selected, deadline, max_iterations=2000)
        size = sum(selected)
        
        if size > best_size:
            best_size = size
            best = [i + 1 for i in range(1, N + 1) if selected[i]]
        
        random.setstate(old_state)
    
    # Final verification
    best_set = set(best)
    for a in best_set:
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                if (a + d) in best_set and (a + 2 * d) in best_set and (a + 3 * d) in best_set:
                    # This should not happen; if it does, remove one element
                    best_set.discard(a + 3 * d)
    
    return sorted(best_set)
