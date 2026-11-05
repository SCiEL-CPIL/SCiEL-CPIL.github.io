"""Multi-start local search for Maximum Cut with ILS and SA polish.

Combines fast steepest-ascent local search from many diverse starts with
Iterated Local Search (ILS) perturbations and simulated annealing to escape
local optima, all under a strict wall-clock deadline.
"""
import time
import numpy as np


def _cut_value(s, A, E):
    # s in {+1,-1};  cut = E/2 - (0.25) * s^T A s
    return 0.5 * E - 0.25 * float(s @ (A @ s))


def _local_search(s, A, E, deadline):
    """Steepest-ascent local search with incremental Ax update."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    Ax = A @ s
    while True:
        if time.time() > deadline:
            break
        gains = s * Ax
        i = int(np.argmax(gains))
        if gains[i] <= 1e-9:
            break
        # Flip vertex i: s[i] -> -s[i]
        # Ax[j] changes by -2 * s[i] * A[j,i] for all j
        old_s_i = s[i]
        s[i] = -s[i]
        Ax -= 2.0 * old_s_i * A[:, i]
    return s, _cut_value(s, A, E)


def _perturb(s, rng, k_min=5, k_max=15):
    """Randomly flip k vertices with adaptive range."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    k = int(rng.integers(k_min, k_max + 1))
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _perturb_adaptive(s, rng, no_improve, n):
    """Adaptive perturbation: larger when stuck longer."""
    s = s.astype(np.float64).copy()
    # Scale perturbation size with stagnation
    base_k = 5 + min(no_improve, 20)
    k = int(rng.integers(max(3, base_k - 3), base_k + 10))
    k = min(k, n // 2)
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _simulated_annealing(s, A, E, deadline, rng, t_start=3.0, t_end=0.05, cooling=0.998):
    """Simulated annealing with incremental Ax update."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    Ax = A @ s
    best_s = s.copy()
    best_val = _cut_value(s, A, E)
    cur = best_val
    T = t_start

    while T > t_end:
        if time.time() > deadline:
            break
        # Pick a random vertex to flip
        i = int(rng.integers(0, n))
        gain = s[i] * Ax[i]

        # Accept if gain > 0, or with probability exp(gain/T) if gain < 0
        if gain > 0 or (gain < 0 and rng.random() < np.exp(gain / T)):
            old_s_i = s[i]
            s[i] = -s[i]
            Ax -= 2.0 * old_s_i * A[:, i]
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
    deadline = start + 110.0  # safety margin under the 120 s limit
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
    for _ in range(30):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=n))

    best_s = None
    best_val = -1.0

    # Phase 1: Local search from all starts
    for s0 in inits:
        if time.time() > deadline - 25.0:
            break
        s, val = _local_search(s0, A, E, deadline - 25.0)
        if val > best_val:
            best_val = val
            best_s = s.copy()

    # Phase 2: ILS on the best solution with adaptive perturbation
    if best_s is not None and time.time() < deadline - 20.0:
        current_s = best_s.copy()
        current_val = best_val
        no_improve_count = 0
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline - 20.0:
                break
            # Adaptive perturbation based on stagnation
            if no_improve_count > 5:
                pert_s = _perturb_adaptive(current_s, rng, no_improve_count, n)
            else:
                pert_s = _perturb(current_s, rng)
            # Local search from perturbed
            ls_s, ls_val = _local_search(pert_s, A, E, deadline - 20.0)
            if ls_val > current_val:
                current_s = ls_s
                current_val = ls_val
                no_improve_count = 0
                if current_val > best_val:
                    best_val = current_val
                    best_s = current_s.copy()
            else:
                no_improve_count += 1

    # Phase 3: Simulated annealing polish
    if best_s is not None and time.time() < deadline - 10.0:
        sa_deadline = deadline - 10.0
        for _ in range(3):
            if time.time() > sa_deadline:
                break
            s, val = _simulated_annealing(
                best_s, A, E, sa_deadline, rng,
                t_start=3.0, t_end=0.05, cooling=0.998
            )
            # Local search after SA
            s, val = _local_search(s, A, E, sa_deadline)
            if val > best_val:
                best_val = val
                best_s = s.copy()

    # Final high-precision local search polish
    if best_s is not None and time.time() < deadline:
        s, val = _local_search(best_s, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s

    return (best_s > 0).astype(int)
