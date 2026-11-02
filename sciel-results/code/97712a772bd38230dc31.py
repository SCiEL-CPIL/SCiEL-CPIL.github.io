"""Construct kissing configuration in R^11 using 4-sparse vectors with support intersection ≤ 2."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
import random


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    # Generate all 4-subsets of {0, ..., 10}
    all_supports = list(combinations(range(D), 4))
    n_supports = len(all_supports)  # C(11,4) = 330

    # Precompute intersection sizes between all pairs of supports
    # For each pair, check if intersection size ≤ 2
    # We'll use a bit representation for fast intersection
    support_bits = []
    for s in all_supports:
        bits = 0
        for pos in s:
            bits |= (1 << pos)
        support_bits.append(bits)

    # Precompute which pairs are compatible (intersection ≤ 2)
    # Two supports are compatible if their intersection has size ≤ 2
    compatible = [[True] * n_supports for _ in range(n_supports)]
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            inter = support_bits[i] & support_bits[j]
            # Count bits
            inter_size = bin(inter).count('1')
            if inter_size > 2:
                compatible[i][j] = False
                compatible[j][i] = False

    # Greedy construction with multiple random orderings
    best_selected = []
    best_size = 0

    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        random.seed(restart * 12345 + 678)
        order = list(range(n_supports))
        random.shuffle(order)

        selected = []
        selected_set = set()

        for idx in order:
            # Check if this support is compatible with all already selected
            is_compatible = True
            for sel_idx in selected:
                if not compatible[idx][sel_idx]:
                    is_compatible = False
                    break

            if is_compatible:
                selected.append(idx)
                selected_set.add(idx)

        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]

    # Local search: try to add more by removing one and adding two
    if time.time() < deadline - 5:
        current = best_selected[:]
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing one element and adding two
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                removed = current[i]
                new_set = current_set - {removed}
                new_list = current[:i] + current[i+1:]

                # Find candidates that can be added
                addable = []
                for j in range(n_supports):
                    if j in new_set:
                        continue
                    # Check compatibility with all in new_set
                    can_add = True
                    for k in new_list:
                        if not compatible[j][k]:
                            can_add = False
                            break
                    if can_add:
                        addable.append(j)

                if len(addable) >= 2:
                    # Try all pairs of addable
                    for a in range(len(addable)):
                        if time.time() > deadline - 5:
                            break
                        for b in range(a + 1, len(addable)):
                            ja, jb = addable[a], addable[b]
                            if not compatible[ja][jb]:
                                continue
                            # This swap increases size by 1
                            new_current = new_list + [ja, jb]
                            current = new_current
                            current_set = set(current)
                            improved = True
                            break
                        if improved:
                            break
                if improved:
                    break

    # Build the actual vectors
    # For each selected support, assign signs. Use a deterministic sign pattern.
    # We need to ensure that for any two vectors sharing ≤ 2 positions, the inner product ≤ 2.
    # Since they share at most 2 positions, the inner product is at most 2 in absolute value.
    # So any sign assignment works!

    vectors = []
    for idx in best_selected:
        v = np.zeros(D, dtype=np.int64)
        for pos in all_supports[idx]:
            v[pos] = 1
        vectors.append(v)

    if not vectors:
        # Fallback: return a single vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v], dtype=np.int64)

    result = np.array(vectors, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
