"""Flexible job-shop dispatch: regret-based look-ahead heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a regret heuristic: prioritize the operation whose best vs second-best
    machine assignment has the largest cost difference, where cost combines
    the operation duration with an estimate of future machine load.
    
    Returns:
        int: index of the ready operation to start
        OR (int, machine): index and preferred machine
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we've been running too long, fall back to a simple rule
    _t0 = _time.time()
    DEADLINE = 110.0  # seconds; leaves margin before 120s limit

    # Gather basic info
    machine_options_list = [op["machine_options"] for op in ready_operations]
    remaining_work_list = [op["remaining_work"] for op in ready_operations]
    remaining_ops_list = [op["remaining_ops"] for op in ready_operations]

    # Build a quick estimate of machine load: for each machine, sum of durations
    # of all ready operations that could use it (proxy for congestion)
    # Also consider remaining_work as a proxy for how much more load a machine will get
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    all_machines = list(all_machines)

    # Estimate machine load: for each machine, sum durations of all ready ops that
    # list it as an option, weighted by how "urgent" those ops are
    machine_load = {}
    for m in all_machines:
        load = 0.0
        for i, mo in enumerate(machine_options_list):
            if m in mo:
                # Weight by remaining work: ops with more remaining work will
                # keep the machine busy longer
                load += mo[m] * (1.0 + 0.1 * remaining_work_list[i])
        machine_load[m] = load

    # For each ready operation, compute the regret:
    # cost(i, m) = duration(i, m) + alpha * machine_load[m] + beta * remaining_work[i]
    # regret(i) = min_m cost(i,m) - second_min_m cost(i,m)
    # We want to pick the operation with the HIGHEST regret (most sensitive to machine choice)
    # Tie-break: among high-regret ops, pick the one with lowest min cost (most urgent)

    alpha = 0.3   # weight on machine load
    beta = 0.5    # weight on remaining work (urgency)

    regrets = []
    min_costs = []
    best_machines = []

    for i, mo in enumerate(machine_options_list):
        if len(mo) == 0:
            regrets.append(0.0)
            min_costs.append(1e9)
            best_machines.append(None)
            continue

        # Compute cost for each machine option
        costs = []
        for m, dur in mo.items():
            ml = machine_load.get(m, 0.0)
            c = dur + alpha * ml + beta * remaining_work_list[i]
            costs.append((c, m))

        # Sort by cost
        costs.sort(key=lambda x: x[0])
        best_cost = costs[0][0]
        best_m = costs[0][1]

        if len(costs) >= 2:
            second_cost = costs[1][0]
            regret = second_cost - best_cost
        else:
            regret = 0.0

        regrets.append(regret)
        min_costs.append(best_cost)
        best_machines.append(best_m)

    # Score: primary = regret (higher is more critical to decide now)
    # Secondary (tie-break) = -min_cost (lower cost is better)
    # Tertiary: -remaining_work (more remaining work = more urgent)

    # Combine into a single score: regret dominates, then urgency
    # Use a weighted combination where regret is the main driver
    scores = []
    for i in range(n):
        s = regrets[i]
        # Add a small urgency component to break ties
        s += 0.1 * (remaining_work_list[i] - np.mean(remaining_work_list))
        scores.append(s)

    # Check deadline
    if _time.time() - _t0 > DEADLINE:
        # Fallback: pick max remaining work
        return int(np.argmax(remaining_work_list))

    # Pick the operation with the highest score (most regret)
    best_idx = int(np.argmax(scores))

    # Return (index, machine) to let the simulator know our preferred machine
    # This is optional per the interface; returning just index is also valid
    # We'll return the tuple to provide machine preference
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
