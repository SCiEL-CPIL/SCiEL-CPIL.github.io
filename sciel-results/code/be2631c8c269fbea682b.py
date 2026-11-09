"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.optimize import linear_sum_assignment

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit

# Track machine loads across calls for load balancing
_machine_loads = {}


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def _soft_min(vals, temp=1.0):
    """Smooth min via log-sum-exp."""
    if len(vals) == 0:
        return 0.0
    shifted = -vals / temp
    max_shifted = np.max(shifted)
    if max_shifted > -1e10:
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        return -temp * log_sum_exp
    return float(np.min(vals))


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses Hungarian assignment on a cost matrix combining machine-specific processing
    times and current cumulative machine loads to minimize max completion time.
    """
    if not ready_operations:
        return 0

    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        return 0

    # Collect all machines that appear in any ready operation's options
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    machines = sorted(all_machines)
    m = len(machines)

    if m == 0:
        return 0

    # Build cost matrix: cost[i][j] = processing time of op i on machine j + load of machine j
    # For machines where op i cannot run, use a large penalty
    INF = 1e9
    cost_matrix = np.full((n, m), INF, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        for mach, dur in op["machine_options"].items():
            if mach in machines:
                j = machines.index(mach)
                load = _machine_loads.get(mach, 0.0)
                # Cost = processing time + current load (proxy for completion time on that machine)
                cost_matrix[i, j] = dur + load

    # Also incorporate urgency: operations with more remaining work should be prioritized
    # We blend the assignment cost with an urgency bias
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    urgency = remaining_works / max_rw  # higher = more urgent

    # For each operation, add an urgency-based offset to all feasible machine costs
    # This makes the Hungarian algorithm prefer assigning urgent ops to good machines
    urgency_weight = 0.3  # weight for urgency in cost
    for i in range(n):
        feasible_mask = cost_matrix[i] < INF
        if np.any(feasible_mask):
            # Subtract urgency bonus from feasible costs (lower cost = preferred)
            cost_matrix[i, feasible_mask] -= urgency_weight * urgency[i] * np.max(cost_matrix[i, feasible_mask])

    # Ensure no negative costs break the algorithm (shift to be non-negative)
    min_cost = np.min(cost_matrix[cost_matrix < INF])
    if min_cost < 0:
        cost_matrix[cost_matrix < INF] -= min_cost

    # Run Hungarian algorithm
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        # Fallback to greedy: pick op with highest remaining work
        return int(np.argmax(remaining_works))

    # Find the first row assignment (operation 0's assigned machine) — but we need
    # to return the index of the operation to start, not a machine assignment.
    # Strategy: the operation assigned to the machine with the lowest resulting
    # completion time should be started first. Among all assigned pairs, pick the
    # operation whose (load + duration) is smallest — that's the one that finishes
    # soonest and frees a machine.
    # Actually, we want to pick the operation that, when started, most reduces
    # the makespan. Use the assignment to determine priority.
    # 
    # Better approach: use the Hungarian result to determine a priority ordering.
    # The operation assigned to the currently least-loaded feasible machine
    # should be started first (it completes soonest, freeing capacity).
    
    # Compute completion time for each assigned pair
    completion_times = np.zeros(n, dtype=np.float64)
    assigned = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < INF:
            mach = machines[c]
            load = _machine_loads.get(mach, 0.0)
            dur = ready_operations[r]["machine_options"].get(mach, 0)
            completion_times[r] = load + dur
            assigned[r] = c
        else:
            # Fallback: use soft-min duration
            vals = list(ready_operations[r]["machine_options"].values())
            completion_times[r] = _soft_min(np.array(vals, dtype=np.float64)) if vals else 0.0
            assigned[r] = 0

    # Pick the operation with the minimum completion time (finishes soonest)
    # This frees a machine earliest, which is good for makespan
    best_idx = int(np.argmin(completion_times))

    # Update machine load tracking for the chosen operation
    if best_idx in assigned:
        chosen_mach = machines[assigned[best_idx]]
    else:
        # Pick the machine with minimum duration for this op
        opts = ready_operations[best_idx]["machine_options"]
        chosen_mach = min(opts, key=lambda k: opts[k]) if opts else None

    if chosen_mach is not None:
        dur = ready_operations[best_idx]["machine_options"].get(chosen_mach, 0)
        _machine_loads[chosen_mach] = _machine_loads.get(chosen_mach, 0.0) + dur

    return best_idx
# EVOLVE-BLOCK-END
