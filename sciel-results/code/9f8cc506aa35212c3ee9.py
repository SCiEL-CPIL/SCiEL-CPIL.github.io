"""Fresh design: time-budgeted local search minimizing distinct colours
used in a K16 edge-colouring where every K4 spans >=5 distinct colours."""

import time
import random
import numpy as np

_N = 16
_DEADLINE = 125.0  # safety margin under 150s wall-clock


def _edge_index(i, j):
    # (i, j) -> 0..119 in lexicographic order
    # number of pairs before row i: sum_{r<i} (N-1-r)
    # row i has edges (i, i+1..N-1)
    before = i * (_N - 1) - i * (i - 1) // 2
    return before + (j - i - 1)


def _build_adj():
    # adjacency: for each vertex, list of (other_vertex, edge_index)
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i + 1, _N):
            e = _edge_index(i, j)
            adj[i].append((j, e))
            adj[j].append((i, e))
    return adj


_ADJ = _build_adj()


def _all_k4s():
    ks = []
    for a in range(_N):
        for b in range(a + 1, _N):
            for c in range(b + 1, _N):
                for d in range(c + 1, _N):
                    ks.append((a, b, c, d))
    return ks


_K4S = _all_k4s()
# Precompute the 6 edge indices for each K4
_K4_EDGES = []
for (a, b, c, d) in _K4S:
    _K4_EDGES.append((
        _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
        _edge_index(b, c), _edge_index(b, d), _edge_index(c, d),
    ))


def _base_solution():
    # (i+j) mod 17, valid per seed
    cols = np.empty(120, dtype=np.int32)
    for i in range(_N):
        for j in range(i + 1, _N):
            cols[_edge_index(i, j)] = (i + j) % 17
    return cols


def _distinct_in_k4(cols, e6):
    return len(set(cols[e6]))


def _is_valid(cols):
    # every K4 must span >=5 distinct colours
    for e6 in _K4_EDGES:
        if len(set(cols[e6])) < 5:
            return False
    return True


def _bad_k4s(cols):
    """Return list of K4s (by e6) that violate the >=5 distinct rule."""
    bad = []
    for e6 in _K4_EDGES:
        if len(set(cols[e6])) < 5:
            bad.append(e6)
    return bad


def _count_distinct(cols):
    return len(set(cols.tolist()))


def _random_recolor_target(cols, max_c):
    # pick a random existing colour to reuse (bias toward shrinking palette)
    present = [c for c in range(max_c) if np.any(cols == c)]
    if present:
        return int(random.choice(present))
    return 0


def _neighbors_of_edge(e):
    # K4s containing edge e: we need all K4s whose edge-set includes e.
    # Precompute edge -> list of k4 edge6s for fast local validity.
    return _EDGE_TO_K4[e]


_EDGE_TO_K4 = {}
for idx, e6 in enumerate(_K4_EDGES):
    for e in e6:
        _EDGE_TO_K4.setdefault(e, []).append(idx)


def _local_valid(cols, e):
    """Check only K4s that contain edge e. Returns True if all are >=5 distinct."""
    for kidx in _EDGE_TO_K4[e]:
        e6 = _K4_EDGES[kidx]
        if len(set(cols[e6])) < 5:
            return False
    return True


def _local_bad_count(cols, e):
    cnt = 0
    for kidx in _EDGE_TO_K4[e]:
        e6 = _K4_EDGES[kidx]
        if len(set(cols[e6])) < 5:
            cnt += 1
    return cnt


def _full_distinct_and_valid(cols):
    d = _count_distinct(cols)
    v = _is_valid(cols)
    return d, v


def construct():
    rng = random.Random(12345)
    t0 = time.time()

    cols = _base_solution()
    best_cols = cols.copy()
    best_d = _count_distinct(cols)  # 17
    assert _is_valid(cols)

    # Verify base is valid quickly; if not, fall back to a guaranteed-valid.
    if not _is_valid(best_cols):
        # Should not happen; but guard.
        best_cols = _base_solution()
        best_d = _count_distinct(best_cols)

    n_edges = 120

    # Simulated-annealing-ish local search.
    # Objective: minimize distinct colours, hard constraint validity.
    # Move: recolor one edge to a random existing colour (or a new lower colour).
    # Accept if validity preserved and distinct count does not increase
    # (and sometimes accept equal to escape plateaus).

    # Track the set of colours currently used to bias reuse.
    max_c = int(np.max(cols)) + 1

    step = 0
    stall = 0
    while time.time() < _DEADLINE:
        step += 1
        e = rng.randrange(n_edges)
        old = int(cols[e])

        # Candidate target colour: prefer an existing colour to shrink palette.
        present = [c for c in range(max_c) if c != old]
        if present:
            new = int(rng.choice(present))
        else:
            new = old

        if new == old:
            continue

        # Local validity check (only K4s containing e).
        cols[e] = new
        ok = _local_valid(cols, e)
        if ok:
            # Count distinct after change.
            d = _count_distinct(cols)
            if d < best_d:
                best_cols = cols.copy()
                best_d = d
                max_c = max(max_c, new + 1)
                stall = 0
            else:
                # Accept equal-distinct with some probability to explore.
                if rng.random() < 0.5:
                    stall += 1
                else:
                    cols[e] = old
                    stall += 1
        else:
            cols[e] = old
            stall += 1

        # Occasionally try to introduce a brand-new colour only if it helps
        # (never helps reduce distinct), so skip.

        # Periodic full re-validation to guard against bugs.
        if step % 20000 == 0:
            if not _is_valid(cols):
                cols = best_cols.copy()

        # If we've hit a long stall and best_d is small, tighten search.
        if stall > 400000:
            # Restart from best with fresh randomness.
            cols = best_cols.copy()
            stall = 0

    # Final: ensure returned solution is valid.
    if not _is_valid(best_cols):
        best_cols = _base_solution()
        best_d = 17

    # Return in lexicographic order as a list of ints in 0..119.
    # best_cols is already indexed in lexicographic order.
    return [int(c) for c in best_cols]
