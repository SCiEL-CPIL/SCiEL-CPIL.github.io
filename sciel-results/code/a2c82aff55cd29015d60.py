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
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Avoid division by zero
    remaining_ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    avg_work = remaining_works / remaining_ops_safe

    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Compute slack-based urgency: operations with less slack relative to their work are more urgent
    # Slack proxy: (remaining_work - min_duration * remaining_ops) / remaining_ops
    # This captures how much "extra" work remains beyond the minimum possible duration
    slack = (remaining_works - min_durations * remaining_ops_safe) / remaining_ops_safe

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
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)
    ns = _norm(slack)

    # Regret-based scoring: model as dynamic bipartite matching where we prioritize
    # operations that would cause the most future congestion if delayed.
    # Regret = opportunity cost of not assigning this op to its least-loaded machine now.
    
    # For each ready operation, compute the "regret" of not scheduling it:
    # 1. How much work it represents relative to what could be done in parallel
    # 2. How constrained its machine options are (fewer options = higher regret if delayed)
    # 3. How urgent it is relative to other operations competing for the same machines
    
    # Machine contention proxy: for each op, how many other ready ops share its machines
    # and what fraction of their remaining work is on shared machines
    machine_contention = np.zeros(n)
    for i in range(n):
        my_machines = set(ready_operations[i]["machine_options"].keys())
        for j in range(n):
            if i == j:
                continue
            shared = my_machines.intersection(ready_operations[j]["machine_options"].keys())
            if shared:
                # Weight by how much of op j's work is on shared machines
                total_work_j = sum(ready_operations[j]["machine_options"].values())
                shared_work_j = sum(ready_operations[j]["machine_options"][m] for m in shared)
                if total_work_j > 0:
                    machine_contention[i] += shared_work_j / total_work_j
    
    # Normalize contention
    nc = _norm(machine_contention)
    
    # Regret score: high remaining work + high contention + low slack = high regret if delayed
    # The contention term captures game-theoretic opportunity cost: if many ops compete
    # for the same machines, delaying any one of them increases overall congestion
    score = 2.5 * nw + 1.2 * naw + 0.8 * nc + 0.3 * no + 0.3 * md_inv + 0.1 * nm + 0.8 * (1.0 - ns)

    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
