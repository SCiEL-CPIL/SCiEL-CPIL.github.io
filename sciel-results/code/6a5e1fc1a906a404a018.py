"""Search-based constructor for the difference-triangle-set task.

Strategy: generate many varied greedy candidates (different row orderings and
randomized tie-breaking) and keep the best one found before an explicit deadline
guard. Each candidate is built by greedy set-cover: extend each row by the
smallest next mark whose pairwise differences are still unused.
"""

import time
import random
import sys


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_rows = None
    best_scope = float('inf')

    # First, run the simple greedy to get a baseline
    rows = _greedy_construct(n_rows, k)
    if rows:
        best_rows = rows
        best_scope = max(max(r) for r in rows)

    # Now try randomized variants to improve
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        rows = _greedy_construct_randomized(n_rows, k, rng, best_scope)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope

    if best_rows is None:
        # Fallback: should not happen, but just in case
        return _greedy_construct(n_rows, k)
    return best_rows


def _greedy_construct(n_rows, k):
    """Deterministic greedy: smallest next mark with unused differences."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _greedy_construct_randomized(n_rows, k, rng, target_scope):
    """Randomized greedy with row-order shuffling and occasional larger jumps."""
    used = set()
    rows = []
    row_order = list(range(n_rows))
    rng.shuffle(row_order)

    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Sometimes try a random jump instead of just +1
            if rng.random() < 0.15:
                jump = rng.randint(1, 5)
                x += jump
            else:
                x += 1
            
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
                # Prune: if current max already exceeds target, this is worse
                if x > target_scope:
                    return None
        rows.append(row)

    # Check if this solution is valid (should be by construction, but verify)
    all_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None  # Invalid
                all_diffs.add(d)

    return rows
# EVOLVE-BLOCK-END
