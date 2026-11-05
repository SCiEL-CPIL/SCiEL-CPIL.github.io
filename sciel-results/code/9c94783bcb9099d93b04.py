"""Multi-start local search for Maximum Cut with simulated annealing polish.

Combines fast steepest-ascent local search from many diverse starts with a
simulated annealing pass to escape local optima, all under a strict wall-clock
deadline.
"""
import time
import numpy as np


def _cut_value(s, A, E):
    # s in {+1,-1};  cut = E/2 - (0.25) * s^T A s
    return 0.5 * E - 0.25 * float(s @ (A @ s))


def _local_search(s, A, E, deadline):
    """Steepest-ascent local search: flip vertex with largest positive gain."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    cur = _cut_value(s, A, E)
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
    return s, cur


def _simulated_annealing(s, A, E, deadline, rng, t_start=5.0, t_end=0.1, cooling=0.999):
    """Simulated annealing to escape local optima."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    cur = _cut_value(s, A, E)
    best_s = s.copy()
    best_val = cur
    T = t_start
    
    while T > t_end:
        if time.time() > deadline:
            break
        # Pick a random vertex to flip
        i = rng.integers(0, n)
        # Compute gain of flipping vertex i
        Ax = A @ s
        gain = s[i] * Ax[i]
        
        # Accept if gain > 0, or with probability exp(gain/T) if gain < 0
        if gain > 0 or rng.random() < np.exp(gain / T):
            s[i] = -s[i]
            cur += gain
            if cur > best_val:
                best_val = cur
                best_s = s.copy()
        
        T *= cooling
    
    return best_s, best_val


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    E = 0.5 * float(A.sum())

    start = time.time()
    deadline = start + 115.0  # safety margin under the 120 s limit
    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = []
    
    # Structured starts
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1.0
    inits.append(alt)
    inits.append(np.ones(n, dtype=np.float64))
    inits.append(-np.ones(n, dtype=np.float64))
    
    # Random starts
    for _ in range(40):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=n))
    
    best_s = None
    best_val = -1.0

    # Phase 1: Local search from all starts
    for s0 in inits:
        if time.time() > deadline:
            break
        s, val = _local_search(s0, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s.copy()

    # Phase 2: Simulated annealing polish on the best solution
    if best_s is not None and time.time() < deadline - 10.0:
        # Run SA with multiple temperature schedules
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            s, val = _simulated_annealing(
                best_s, A, E, deadline - 5.0, rng,
                t_start=5.0, t_end=0.1, cooling=0.999
            )
            if val > best_val:
                best_val = val
                best_s = s.copy()
            # Local search after SA
            s, val = _local_search(best_s, A, E, deadline - 5.0)
            if val > best_val:
                best_val = val
                best_s = s.copy()

    # Final local search polish
    if best_s is not None and time.time() < deadline:
        s, val = _local_search(best_s, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s

    return (best_s > 0).astype(int)
