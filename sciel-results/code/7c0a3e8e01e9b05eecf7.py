"""Flexible job-shop dispatch: Adaptive Regret with Job Pressure Heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses an adaptive regret heuristic where machine congestion is estimated 
    by the 'pressure' of ready jobs (their remaining work) distributed across 
    their eligible machines. This better approximates future machine load 
    than just summing immediate durations.
    
    Returns:
        (int, machine): index of the ready operation to start and preferred machine
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the fastest machine for the single operation
        best_m = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, best_m)

    _t0 = _time.time()
    DEADLINE = 115.0  # Safety margin

    # Extract data
    machine_options_list = [op["machine_options"] for op in ready_operations]
    remaining_work_list = np.array([op["remaining_work"] for op in ready_operations])
    
    # Identify all machines
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    all_machines = list(all_machines)

    # 1. Estimate Machine Pressure
    # Pressure is a proxy for future load. We assume that a job's remaining work
    # will be distributed among its eligible machines.
    # We distribute the job's remaining_work evenly across its options to estimate
    # the "share" of load each machine will receive from that job.
    machine_pressure = {m: 0.0 for m in all_machines}
    
    for i, mo in enumerate(machine_options_list):
        rw = remaining_work_list[i]
        num_opts = len(mo)
        if num_opts == 0:
            continue
        # Distribute the job's remaining work pressure across its eligible machines
        pressure_per_machine = rw / num_opts
        for m in mo.keys():
            machine_pressure[m] += pressure_per_machine

    # 2. Compute Costs and Regrets
    # Cost(i, m) = duration(i, m) + alpha * machine_pressure[m]
    # We use a dynamic alpha based on the variance of pressure to balance duration vs load.
    pressures = np.array([machine_pressure[m] for m in all_machines])
    if len(pressures) > 0 and np.std(pressures) > 1e-6:
        # Scale alpha so that the pressure term is comparable to durations
        # Average duration is roughly average remaining_work / num_ops, but we don't have num_ops.
        # We'll use a fixed scaling factor that has worked well, or normalize.
        # Let's normalize pressure to be in the same order of magnitude as durations.
        # Typical duration is ~1-10. Typical remaining work is ~10-100.
        # So pressure is larger. We divide by 10 to bring it down.
        alpha = 0.5 
    else:
        alpha = 0.0

    regrets = np.zeros(n)
    min_costs = np.full(n, 1e9)
    best_machines = [None] * n

    for i in range(n):
        mo = machine_options_list[i]
        if not mo:
            continue
        
        costs = []
        for m, dur in mo.items():
            p = machine_pressure.get(m, 0.0)
            c = dur + alpha * p
            costs.append((c, m))
        
        if not costs:
            continue
            
        costs.sort(key=lambda x: x[0])
        best_cost, best_m = costs[0]
        min_costs[i] = best_cost
        best_machines[i] = best_m
        
        if len(costs) >= 2:
            second_cost = costs[1][0]
            regrets[i] = second_cost - best_cost
        else:
            regrets[i] = 0.0

    # 3. Combine Regret and Urgency
    # We want to prioritize operations that are "critical" (high regret) AND "urgent" (high remaining work).
    # Normalize both to [0, 1] to make them comparable.
    
    max_regret = np.max(regrets) if np.max(regrets) > 1e-9 else 1.0
    max_rw = np.max(remaining_work_list) if np.max(remaining_work_list) > 1e-9 else 1.0
    
    norm_regrets = regrets / max_regret
    norm_rw = remaining_work_list / max_rw

    # Check deadline
    if _time.time() - _t0 > DEADLINE:
        # Fallback: pick max remaining work
        return (int(np.argmax(remaining_work_list)), best_machines[int(np.argmax(remaining_work_list))])

    # Weighted Score: Regret is the primary driver for "which decision is hardest to make",
    # but Urgency ensures we don't starve long jobs.
    # A good balance is often 0.6 * regret + 0.4 * urgency, but let's make it adaptive.
    # If all regrets are low, rely more on urgency.
    regret_spread = np.std(regrets)
    if regret_spread < 0.1:
        w_regret = 0.2
        w_urgency = 0.8
    else:
        w_regret = 0.7
        w_urgency = 0.3

    scores = w_regret * norm_regrets + w_urgency * norm_rw
    
    best_idx = int(np.argmax(scores))
    best_m = best_machines[best_idx]
    
    if best_m is None:
        # Fallback if something went wrong
        best_m = min(machine_options_list[best_idx].items(), key=lambda x: x[1])[0]

    return (best_idx, best_m)
# EVOLVE-BLOCK-END
