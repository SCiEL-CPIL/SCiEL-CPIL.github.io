"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def _soft_min(values, temperature=1.0):
    """Smooth approximation of min using log-sum-exp trick."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    v = np.array(values, dtype=np.float64)
    vmin = np.min(v)
    # soft min = -1/T * log(sum(exp(-T*v)))
    # For stability: vmin - 1/T * log(sum(exp(-T*(v - vmin))))
    scaled = -v / temperature
    max_s = np.max(scaled)
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return -log_sum * temperature


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        # Single operation: assign to fastest machine
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Avoid division by zero
    remaining_ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    avg_work = remaining_works / remaining_ops_safe

    # For each operation, compute min and max duration across machines
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Spread: how much variance in processing times across machines
    spread = max_durations - min_durations

    # --- Regret-based scoring (game-theoretic opportunity cost) ---
    # For each operation, compute the "regret" of NOT scheduling it now:
    # Regret = (cost of delaying on least-loaded machine) weighted by criticality
    #
    # The key insight: an operation that would be expensive to place later
    # (because its best machines will be busy) has high regret if delayed.
    # We approximate this by:
    #   1. The min duration (if we delay, we might lose access to fast machines)
    #   2. The remaining work (more work = more critical path impact)
    #   3. The spread (high spread = high risk of being forced onto slow machine)
    #   4. Flexibility (more machine options = less regret, since alternatives exist)

    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    # Normalized features
    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    naw = _norm(avg_work)
    md_inv = 1.0 - _norm(min_durations)  # shorter min duration = more urgent
    nm_inv = 1.0 - _norm(num_machines_available)  # fewer options = more regret
    ns = _norm(spread)  # high spread = more regret

    # Soft-min duration: smooth surrogate for the best machine duration
    # Lower soft-min means the operation has a very good machine option
    soft_min_durs = np.array([
        _soft_min(list(op["machine_options"].values()), temperature=1.0)
        if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    smd_inv = 1.0 - _norm(soft_min_durs)

    # Regret score: combination of opportunity cost factors
    # Weight heavily on:
    #   - Remaining work (critical path)
    #   - Loss of fast machine access (min duration inverse)
    #   - Spread (risk of bad assignment later)
    #   - Fewer machine options (less flexibility = more regret)
    #   - Soft-min duration (smooth version of best machine availability)
    score = (
        2.5 * nw          # total remaining work (LRPT core)
        + 1.5 * naw       # density (work per remaining op)
        + 1.2 * md_inv    # urgency: short min duration means fast machine might be taken
        + 0.8 * nm_inv    # low flexibility: few machine options = high regret if delayed
        + 0.6 * ns        # high spread: risk of being forced onto slow machine
        + 0.4 * smd_inv   # soft-min: smooth proxy for best machine access
        + 0.3 * no        # number of remaining ops
    )

    best_idx = int(np.argmax(score))

    # Assign to the fastest available machine (SPT)
    op = ready_operations[best_idx]
    mo = op.get("machine_options", {})
    if mo:
        best_machine = min(mo, key=mo.get)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
