"""Rewritten program: exact Maximum Weight Independent Set via branch-and-bound
with bitset conflict encoding, exploiting the structure of the lcm>256 constraint."""

import time
from math import gcd
import numpy as np
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)  # 255

    # Precompute conflict matrix as numpy bitsets
    # conflict_mask[i] is a numpy uint64 array where bit j is set if lcm(numbers[i], numbers[j]) <= N
    # n=255, so we need ceil(255/64) = 4 uint64 words
    n_words = (n + 63) // 64  # 4

    # Build conflict bitsets
    conflict = [np.zeros(n_words, dtype=np.uint64) for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                # Set bit j in conflict[i] and bit i in conflict[j]
                conflict[i][j // 64] |= (1 << (j % 64))
                conflict[j][i // 64] |= (1 << (i % 64))

    # Weight: 1/x for each number
    weights = np.array([1.0 / x for x in numbers], dtype=np.float64)

    # Helper: check if a set represented by bitset is independent (no internal conflicts)
    def is_independent(bitset):
        # For each vertex in bitset, check it doesn't conflict with any other vertex in bitset
        for i in range(n):
            if bitset[i // 64] & (1 << (i % 64)):
                # Check if any neighbor of i is also in bitset
                c = conflict[i]
                # AND with bitset and check if any bit is set
                for w in range(n_words):
                    if c[w] & bitset[w]:
                        return False
        return True

    # Helper: get candidate vertices not in current_set (bitset) that don't conflict with it
    def get_candidates(current_bitset):
        candidates = []
        for i in range(n):
            if current_bitset[i // 64] & (1 << (i % 64)):
                continue  # already selected
            # Check if i conflicts with any vertex in current_bitset
            c = conflict[i]
            conflicts = False
            for w in range(n_words):
                if c[w] & current_bitset[w]:
                    conflicts = True
                    break
            if not conflicts:
                candidates.append(i)
        return candidates

    # Branch and bound for MWIS
    # State: current selection (bitset), list of candidate vertices that can still be added
    # Upper bound: current_weight + sum of weights of all candidates (optimistic)

    best_weight = 0.0
    best_selection = np.zeros(n_words, dtype=np.uint64)

    def update_best(selection, weight):
        nonlocal best_weight, best_selection
        if weight > best_weight:
            best_weight = weight
            best_selection = selection.copy()

    # Use recursive branch and bound with pruning
    # Sort candidates by weight descending for better pruning
    def branch_and_bound(current_bitset, current_weight, candidates):
        if elapsed() > DEADLINE:
            return

        # Update best
        update_best(current_bitset, current_weight)

        if not candidates:
            return

        # Upper bound: current_weight + sum of weights of remaining candidates
        upper_bound = current_weight + sum(weights[c] for c in candidates)
        if upper_bound <= best_weight + 1e-15:
            return

        # Sort candidates by weight descending
        candidates.sort(key=lambda x: -weights[x])

        # Try adding each candidate
        for idx, cand in enumerate(candidates):
            if elapsed() > DEADLINE:
                return

            # New bitset with cand added
            new_bitset = current_bitset.copy()
            new_bitset[cand // 64] |= (1 << (cand % 64))
            new_weight = current_weight + weights[cand]

            # Find new candidates: those in candidates[idx+1:] that don't conflict with cand
            new_candidates = []
            for c in candidates[idx + 1:]:
                # Check if c conflicts with cand
                if conflict[cand][c // 64] & (1 << (c % 64)):
                    continue
                new_candidates.append(c)

            # Upper bound for this branch
            ub = new_weight + sum(weights[c] for c in new_candidates)
            if ub > best_weight + 1e-15:
                branch_and_bound(new_bitset, new_weight, new_candidates)

    # Initial greedy solution to get a good lower bound
    # Sort all vertices by weight descending
    initial_order = sorted(range(n), key=lambda i: -weights[i])
    greedy_bitset = np.zeros(n_words, dtype=np.uint64)
    greedy_weight = 0.0
    for i in initial_order:
        # Check if i conflicts with any vertex in greedy_bitset
        c = conflict[i]
        conflicts = False
        for w in range(n_words):
            if c[w] & greedy_bitset[w]:
                conflicts = True
                break
        if not conflicts:
            greedy_bitset[i // 64] |= (1 << (i % 64))
            greedy_weight += weights[i]

    update_best(greedy_bitset, greedy_weight)

    # Initial candidates for branch and bound: all vertices
    initial_candidates = list(range(n))
    branch_and_bound(np.zeros(n_words, dtype=np.uint64), 0.0, initial_candidates)

    # Convert best selection back to numbers
    result = []
    for i in range(n):
        if best_selection[i // 64] & (1 << (i % 64)):
            result.append(numbers[i])

    return result
