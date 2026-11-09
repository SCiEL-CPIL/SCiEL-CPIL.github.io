# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import differential_evolution

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute clockwise edge masks for each demand
    cw_masks = np.zeros((D, n), dtype=np.int8)
    for i, (s, t, w) in enumerate(demands):
        if s != t:
            cur = s
            while cur != t:
                cw_masks[i, cur] = 1
                cur = (cur + 1) % n

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Efficient load computation
    def compute_loads(routing_bits):
        """
        routing_bits: array of 0/1, shape (15,)
        Returns edge loads, shape (10,)
        """
        route_masks = cw_masks * (1 - routing_bits[:, None]) + (1 - cw_masks) * routing_bits[:, None]
        loads = (weights[:, None] * route_masks).sum(axis=0)
        return loads

    def max_load(routing_bits):
        loads = compute_loads(routing_bits)
        return loads.max()

    def smooth_max_load(routing_bits, beta=1000.0):
        """
        Smooth approximation of max load using log-sum-exp.
        This provides a differentiable surrogate for optimization.
        """
        loads = compute_loads(routing_bits)
        # log-sum-exp for numerical stability
        m = loads.max()
        if m == 0:
            return 0.0
        # Soft max: (1/beta) * log(sum(exp(beta * loads)))
        # But we want to minimize the max, so we use a large beta
        # Actually, for minimization of max, we can just use the max directly
        # The smooth version is useful for gradient-based methods
        return m + (1.0/beta) * np.log(np.sum(np.exp(beta * (loads - m))))

    start_time = time.time()
    deadline = start_time + 600  # 30 second safety margin

    # Start with all clockwise
    best_bits = np.zeros(D, dtype=np.int8)
    best_ml = max_load(best_bits)

    # Also try all counter-clockwise
    all_ccw = np.ones(D, dtype=np.int8)
    ml_ccw = max_load(all_ccw)
    if ml_ccw < best_ml:
        best_bits = all_ccw
        best_ml = ml_ccw

    # Phase 1: Local search (single and pair flips)
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i in range(D):
            if time.time() > deadline - 30:
                break
            new_bits = best_bits.copy()
            new_bits[i] = 1 - new_bits[i]
            ml = max_load(new_bits)
            if ml < best_ml - 1e-9:
                best_bits = new_bits
                best_ml = ml
                improved = True

    # Pair flips
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(D):
                if time.time() > deadline - 30:
                    break
                for j in range(i+1, D):
                    if time.time() > deadline - 30:
                        break
                    new_bits = best_bits.copy()
                    new_bits[i] = 1 - new_bits[i]
                    new_bits[j] = 1 - new_bits[j]
                    ml = max_load(new_bits)
                    if ml < best_ml - 1e-9:
                        best_bits = new_bits
                        best_ml = ml
                        improved = True
                        break
                if time.time() > deadline - 30:
                    break

    # Phase 2: Differential Evolution on continuous domain [0,1]^15
    # with penalty for load imbalance and rounding to binary
    if time.time() < deadline - 60:
        def objective(x):
            """
            Objective function for differential evolution.
            x is a continuous vector in [0, 1]^15.
            We round to binary and compute the max load.
            Add a penalty for being far from binary to encourage convergence.
            """
            # Round to binary
            bits = (x >= 0.5).astype(np.int8)
            ml = max_load(bits)

            # Penalty for non-binary values (encourages convergence to 0 or 1)
            # This helps the optimizer find discrete solutions
            penalty = 0.1 * np.sum((x - 0.5)**2)

            return ml + penalty

        try:
            # Run differential evolution
            de_result = differential_evolution(
                objective,
                bounds=[(0, 1)] * D,
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=True,
                init='latinhypercube'
            )

            if de_result.success or de_result.fun < best_ml:
                de_bits = (de_result.x >= 0.5).astype(np.int8)
                de_ml = max_load(de_bits)
                if de_ml < best_ml:
                    best_bits = de_bits
                    best_ml = de_ml

        except Exception:
            pass

    # Phase 3: Simulated annealing from best found
    if time.time() < deadline - 20:
        current = best_bits.copy()
        current_ml = best_ml
        T = 2.0
        T_min = 0.01
        cool_rate = 0.999

        while T > T_min and time.time() < deadline - 10:
            for _ in range(50):
                if time.time() >= deadline - 10:
                    break
                idx = random.randint(0, D-1)
                trial = current.copy()
                trial[idx] = 1 - trial[idx]
                trial_ml = max_load(trial)
                delta = trial_ml - current_ml
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current = trial
                    current_ml = trial_ml
                    if current_ml < best_ml:
                        best_ml = current_ml
                        best_bits = current.copy()
            T *= cool_rate

    # Phase 4: Final local search polish
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(D):
            if time.time() > deadline - 5:
                break
            new_bits = best_bits.copy()
            new_bits[i] = 1 - new_bits[i]
            ml = max_load(new_bits)
            if ml < best_ml - 1e-9:
                best_bits = new_bits
                best_ml = ml
                improved = True

    # Convert to float array: 0 for cw, 1 for ccw
    result = best_bits.astype(np.float64)
    return result

# EVOLVE-BLOCK-END
