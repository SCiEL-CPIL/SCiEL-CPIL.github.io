"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def _greedy_mis(conflicts, N, rng, deadline):
    """Greedy min-degree maximum independent set."""
    alive = np.ones(N, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    # Generate candidates with norm^2 = 4
    candidates = []
    
    # Weight 1, entries ±2
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    
    # Weight 2, entries ±1
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    # Weight 4, entries ±1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    dots = candidates @ candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Multiple random restarts with greedy, keep best
    best_selected = []
    n_restarts = 50
    time_per_restart = (deadline - start_time - 10) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        rng = np.random.default_rng(42 + restart)
        selected = _greedy_mis(conflicts, N, rng, deadline - 10)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    result = candidates[best_selected]
    return result

# EVOLVE-BLOCK-END
