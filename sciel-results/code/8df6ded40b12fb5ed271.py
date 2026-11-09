# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def ring_routing() -> np.ndarray:
    N = 10
    D = 15

    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW edge mask for each demand as a (D, N) binary matrix
    # cw_matrix[i, e] = 1 if edge e is used when demand i goes clockwise
    cw_matrix = np.zeros((D, N), dtype=np.int32)
    ccw_matrix = np.zeros((D, N), dtype=np.int32)
    weights = np.zeros(D, dtype=np.float64)

    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        e = s
        while e != t:
            cw_matrix[i, e] = 1
            e = (e + 1) % N
        ccw_matrix[i] = 1 - cw_matrix[i]

    # Precompute weight-multiplied matrices for fast load computation
    # When demand i is CW: load contribution = weights[i] * cw_matrix[i]
    # When demand i is CCW: load contribution = weights[i] * ccw_matrix[i]

    def compute_max_load(routing):
        """routing is a 1D array of 0/1 (0=CW, 1=CCW)"""
        # For each demand, select CW or CCW edges
        # load[e] = sum over i of weights[i] * (routing[i] * ccw_matrix[i,e] + (1-routing[i]) * cw_matrix[i,e])
        # = sum_i weights[i] * cw_matrix[i,e] + sum_i weights[i] * routing[i] * (ccw_matrix[i,e] - cw_matrix[i,e])
        # Let base = weights^T @ cw_matrix (shape N)
        # Let delta = (weights * routing)^T @ (ccw_matrix - cw_matrix)
        # load = base + delta
        base = weights @ cw_matrix  # (N,)
        diff = ccw_matrix - cw_matrix  # (D, N)
        weighted_routing = weights * routing  # (D,)
        delta = weighted_routing @ diff  # (N,)
        loads = base + delta
        return loads.max()

    def compute_loads(routing):
        base = weights @ cw_matrix
        diff = ccw_matrix - cw_matrix
        weighted_routing = weights * routing
        delta = weighted_routing @ diff
        return base + delta

    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin

    best_routing = None
    best_max_load = float('inf')

    # Try all-CW and all-CCW
    for init in [np.zeros(D, dtype=np.float64), np.ones(D, dtype=np.float64)]:
        ml = compute_max_load(init)
        if ml < best_max_load:
            best_max_load = ml
            best_routing = init.copy()

    random.seed(42)
    np.random.seed(42)

    # Local search with multiple restarts
    def local_search(current, current_ml, max_inner_iters=5000):
        improved = True
        inner_iter = 0
        while improved and inner_iter < max_inner_iters and time.time() < deadline - 5:
            improved = False
            inner_iter += 1
            # Try flipping each single bit
            for i in range(D):
                trial = current.copy()
                trial[i] = 1.0 - trial[i]
                trial_ml = compute_max_load(trial)
                if trial_ml < current_ml:
                    current = trial
                    current_ml = trial_ml
                    improved = True
        return current, current_ml

    # Phase 1: Multiple random restarts with hill climbing
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() >= deadline - 30:
            break
        current = np.random.randint(0, 2, D).astype(np.float64)
        current_ml = compute_max_load(current)
        current, current_ml = local_search(current, current_ml)
        if current_ml < best_max_load:
            best_max_load = current_ml
            best_routing = current.copy()

    # Phase 2: Perturbation + local search from best
    for r in range(200):
        if time.time() >= deadline - 30:
            break
        current = best_routing.copy()
        # Randomly flip 1-3 bits
        n_flip = random.randint(1, 3)
        for _ in range(n_flip):
            idx = random.randint(0, D - 1)
            current[idx] = 1.0 - current[idx]
        current_ml = compute_max_load(current)
        current, current_ml = local_search(current, current_ml)
        if current_ml < best_max_load:
            best_max_load = current_ml
            best_routing = current.copy()

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        current = best_routing.copy()
        current_ml = compute_max_load(current)
        T = 3.0
        T_min = 0.001
        cool_rate = 0.999

        while T > T_min and time.time() < deadline - 5:
            for _ in range(200):
                if time.time() >= deadline - 5:
                    break
                idx = random.randint(0, D - 1)
                trial = current.copy()
                trial[idx] = 1.0 - trial[idx]
                trial_ml = compute_max_load(trial)
                delta = trial_ml - current_ml
                if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = trial
                    current_ml = trial_ml
                    if current_ml < best_max_load:
                        best_max_load = current_ml
                        best_routing = current.copy()
            T *= cool_rate

    # Phase 4: Final fine-tuning - try all single flips and double flips from best
    if time.time() < deadline - 5:
        current = best_routing.copy()
        current_ml = compute_max_load(current)
        # Single flips
        for i in range(D):
            trial = current.copy()
            trial[i] = 1.0 - trial[i]
            trial_ml = compute_max_load(trial)
            if trial_ml < current_ml:
                current = trial
                current_ml = trial_ml
        # Double flips
        for i in range(D):
            for j in range(i + 1, D):
                trial = current.copy()
                trial[i] = 1.0 - trial[i]
                trial[j] = 1.0 - trial[j]
                trial_ml = compute_max_load(trial)
                if trial_ml < current_ml:
                    current = trial
                    current_ml = trial_ml
        if current_ml < best_max_load:
            best_max_load = current_ml
            best_routing = current.copy()

    # Return as float array: 0 for CW, 1 for CCW (>= 0.5 means CCW)
    result = np.zeros(D, dtype=float)
    for i in range(D):
        if best_routing[i] >= 0.5:
            result[i] = 1.0
        else:
            result[i] = 0.0

    return result

# EVOLVE-BLOCK-END
