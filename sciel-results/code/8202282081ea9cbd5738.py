"""Covering design C(19,7,4) - fast greedy + removal + simulated annealing."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0


def _build_index(v, t):
    """Map each t-subset to an integer index."""
    idx = {}
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        i += 1
    return idx, i


def _greedy_construct(v, k, t, idx, total, deadline):
    """Fast greedy using numpy uncovered mask."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    start = time.time()

    # Precompute for each point x and each (t-1)-subset of other points,
    # which t-subset index it covers. This is expensive but done once.
    # Actually, let's just compute gain on the fly but efficiently.

    while uncovered.any():
        if time.time() > deadline:
            break

        first_idx = int(np.argmax(uncovered))
        # Find the t-subset corresponding to first_idx
        # We need a reverse map: index -> tuple
        # Build it lazily or precompute
        # For speed, precompute list of tuples
        # Actually, let's build a reverse list at start
        # We'll do this outside for clarity, but for now, iterate
        # Better: build reverse map once

        # For this run, build reverse map
        # We'll pass it in or build here. Let's build in construct and pass.
        # For now, this is a placeholder - we'll fix in construct

        # Find first uncovered t-subset by scanning
        # This is slow. Better to have reverse map.
        # Let's restructure: build reverse map in construct.

        # For now, break and restructure
        break

    return blocks


def _greedy_construct_fast(v, k, t, idx, total, reverse_map, deadline):
    """Fast greedy with precomputed reverse map."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []

    # Precompute: for each point x, and for each (t-1)-subset S of points not containing x,
    # the index of the t-subset S ∪ {x}. Store as a dict: (x, frozenset(S)) -> index
    # This is too large. Instead, compute gain on the fly.

    while uncovered.any():
        if time.time() > deadline:
            break

        first_idx = int(np.argmax(uncovered))
        first_tsub = reverse_map[first_idx]
        block = list(first_tsub)

        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    ci = idx.get(tsub, -1)
                    if ci >= 0 and uncovered[ci]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            if time.time() > deadline:
                break

        block.sort()
        blocks.append(block)
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                uncovered[ci] = False

    return blocks


def _improve_blocks(blocks, v, k, t, idx, deadline):
    """Deterministic removal pass."""
    n = len(idx)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    blocks = list(blocks)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            removable = True
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0 and coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        coverage[ci] -= 1
                blocks.pop(i)
                improved = True
                break

    return blocks


def _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=10000):
    """Randomized removal attempts."""
    n = len(idx)
    blocks = list(blocks)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    rng = random.Random(42)
    for _ in range(n_trials):
        if time.time() > deadline:
            break
        if not blocks:
            break
        i = rng.randrange(len(blocks))
        block = blocks[i]
        removable = True
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0 and coverage[ci] <= 1:
                removable = False
                break
        if removable:
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
                    coverage[ci] -= 1
            blocks.pop(i)

    return blocks


def _simulated_annealing(blocks, v, k, t, idx, deadline):
    """Simulated annealing to escape local optima in block count."""
    n = len(idx)
    blocks = list(blocks)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    rng = random.Random(123)
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return blocks

    current_count = len(blocks)
    best_count = current_count
    best_blocks = list(blocks)
    best_coverage = coverage.copy()

    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.01
    n_steps = int(min(50000, total_time * 200))  # adaptive step count

    for step in range(n_steps):
        if time.time() > deadline:
            break
        if not blocks:
            break

        # Temperature schedule
        progress = step / max(1, n_steps)
        T = T_start * (T_end / T_start) ** progress

        # Try to remove a random block
        i = rng.randrange(len(blocks))
        block = blocks[i]

        # Check if removable
        uncovered_count = 0
        affected = []
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                if coverage[ci] <= 1:
                    uncovered_count += 1
                affected.append(ci)

        if uncovered_count == 0:
            # Always accept removal
            for ci in affected:
                coverage[ci] -= 1
            blocks.pop(i)
            current_count = len(blocks)
            if current_count < best_count:
                best_count = current_count
                best_blocks = list(blocks)
                best_coverage = coverage.copy()
        else:
            # Accept with probability exp(-uncovered_count / T)
            if rng.random() < np.exp(-uncovered_count / max(T, 1e-10)):
                # Accept: remove block, coverage becomes invalid for those t-subsets
                # But we track coverage; when we remove, coverage[ci] decreases
                # If coverage[ci] was 1, it becomes 0 -> uncovered
                for ci in affected:
                    coverage[ci] -= 1
                blocks.pop(i)
                current_count = len(blocks)
                # Don't update best here since coverage is broken
            else:
                # Reject: do nothing
                pass

    # Return best found
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index and reverse map
    idx, total = _build_index(v, t)
    reverse_map = [None] * total
    for comb, i in idx.items():
        reverse_map[i] = comb

    # Fast greedy construction
    blocks = _greedy_construct_fast(v, k, t, idx, total, reverse_map, deadline)

    # Deterministic improvement pass
    if time.time() < deadline - 5:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)

    # Randomized removal pass
    if time.time() < deadline - 3:
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline)

    # Simulated annealing
    if time.time() < deadline - 2:
        blocks = _simulated_annealing(blocks, v, k, t, idx, deadline)

    return blocks
