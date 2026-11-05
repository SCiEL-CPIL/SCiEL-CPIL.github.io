"""Multi-start steepest-ascent local search for Maximum Cut.

Cut value for a ±1 labeling s:  cut = E/2 - (1/4) * s^T A s.
Flipping vertex v changes the cut by  gain = s[v] * (A @ s)[v].
We run steepest-ascent local search (flip the vertex with the largest
positive gain) from many random + structured initializations, keeping the
best labeling found, with an explicit wall-clock deadline guard.
"""
import time
import numpy as np


def _cut_value(s, A, E):
    # s in {+1,-1};  cut = E/2 - (1/4) * s^T A s
    return 0.5 * E - 0.25 * float(s @ (A @ s))


def _local_search(s, A, E, deadline):
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    cur = _cut_value(s, A, E)
    flips = 0
    while True:
        if time.time() > deadline:
            break
        Ax = A @ s
        gains = s * Ax
        i = int(np.argmax(gains))
        if gains[i] <= 1e-9:
            break
        s[i] = -s[i]
        cur += gains[i]
        flips += 1
        if flips % 64 == 0:
            cur = _cut_value(s, A, E)
    return s, cur


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    E = 0.5 * float(A.sum())

    start = time.time()
    deadline = start + 115.0  # safety margin under the 120 s limit
    rng = np.random.default_rng(12345)

    # Structured starts plus many random starts.
    inits = []
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1.0
    inits.append(alt)
    inits.append(np.ones(n, dtype=np.float64))
    for _ in range(60):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=n))

    best_s = None
    best_val = -1.0
    for s0 in inits:
        if time.time() > deadline:
            break
        s, val = _local_search(s0, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s

    # Final polish pass on the current best (tighter convergence).
    if best_s is not None and time.time() < deadline:
        s, val = _local_search(best_s, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s

    return (best_s > 0).astype(int)
