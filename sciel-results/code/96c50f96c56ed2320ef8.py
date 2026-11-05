"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.time() > _DEADLINE


def _soft_min(durs, temperature=0.5):
    """Smooth surrogate for min using log-sum-exp: -log(sum(exp(-d/T)))*T.
    As T->0 this approaches hard min; for small T it's a smooth, differentiable
    approximation that handles near-ties gracefully."""
    d = np.asarray(durs, dtype=np.float64)
    if len(d) == 1:
        return d[0]
    # Numerically stable: subtract min before exponentiating
    d_min = d.min()
    # Use scaled version: soft_min ≈ d_min - T * log(sum(exp(-(d - d_min)/T)))
    scaled = -(d - d_min) / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return d_min - temperature * log_sum


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path for single op
    if n == 1:
        return 0

    # Extract key metrics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft-min duration and avg duration across machine options
    best_durs = np.empty(n, dtype=np.float64)
    avg_durs = np.empty(n, dtype=np.float64)

    # Count how many ready ops compete for each machine (load-balancing signal)
    machine_competition = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_competition[m] = machine_competition.get(m, 0) + 1

    for i, op in enumerate(ready_operations):
        durs = np.array(list(op["machine_options"].values()), dtype=np.float64)
        best_durs[i] = _soft_min(durs, temperature=0.3)
        avg_durs[i] = np.mean(durs)

    # Normalization to [0, 1] (higher = more urgent / more efficient)
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Urgency: more remaining work and more remaining ops => more urgent
    urgency_work = _normalize(remaining_works)
    urgency_ops = _normalize(remaining_ops)
    urgency = 0.7 * urgency_work + 0.3 * urgency_ops

    # Efficiency: shorter operations reduce idle time; invert so shorter = higher
    eff_best = _normalize(-best_durs)  # shorter soft-min duration -> higher
    eff_avg = _normalize(-avg_durs)
    efficiency = 0.6 * eff_best + 0.4 * eff_avg

    # Load balancing: prefer ops whose best machine has fewer competing ready ops
    # Fewer competitors on the best machine => less congestion => higher score
    comp_scores = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        # Find the machine with the soft-min duration (i.e., the "best" machine)
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        comp_scores[i] = machine_competition.get(best_m, 1)
    # Invert: fewer competitors = higher score
    load_balance = _normalize(-comp_scores)

    # Combined score: urgency dominant, efficiency secondary, load balance tertiary
    scores = 1.0 * urgency + 0.4 * efficiency + 0.2 * load_balance

    # Pick the operation with the highest combined score
    best_idx = int(np.argmax(scores))

    return best_idx
# EVOLVE-BLOCK-END
