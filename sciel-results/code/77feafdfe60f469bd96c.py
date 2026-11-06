"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_DEADLINE = 110.0  # safety margin below the 120s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.

    Strategy:
    1. Job Selection: Prioritize jobs based on a combined score of Urgency 
       (remaining work/ops) and Regret (difference between best and second-best 
       machine costs). This ensures that jobs with few machine options and 
       high remaining work are scheduled first to avoid bottlenecks.
    2. Machine Selection: For the selected job, choose the machine that 
       minimizes the effective cost: duration + W * (other load on that machine).
       W is tuned to balance immediate processing time against future congestion.

    Parameters:
    - W (Congestion Weight): 0.8. Reduced from 1.0 to allow slightly more 
      utilization of efficient machines even if shared, relying on Regret to 
      manage criticality.
    - REGRET_W: 0.5. Increased to strongly prioritize jobs with high regret.
    """
    if _elapsed() > _DEADLINE:
        # Safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    # Precompute data for each operation
    op_data = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1.0))
        opts = op.get("machine_options", {})
        # Filter out zero durations if any, though unlikely
        valid_opts = {m: d for m, d in opts.items() if d > 0}
        op_data.append((rw, ro, valid_opts))

    # Calculate machine loads based on *all* ready operations
    # This approximates the contention if all these ops were to be scheduled immediately
    machine_load = {}
    for rw, ro, opts in op_data:
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Tuned Parameters
    W = 0.8          # Congestion weight
    REGRET_W = 0.5   # Weight for regret in job scoring
    URGENCY_RW = 1.0 # Weight for remaining work in urgency
    URGENCY_RO = 0.3 # Weight for remaining ops in urgency

    best_score = -float('inf')
    best_index = -1
    best_machine = None

    for i in range(n):
        rw, ro, opts = op_data[i]
        if not opts:
            continue

        # Calculate cost for each eligible machine
        costs = []
        for m, d in opts.items():
            d = float(d)
            load = machine_load.get(m, 0.0)
            other_load = load - d
            # Effective cost: current duration + weighted penalty for other jobs wanting this machine
            c = d + W * other_load
            costs.append((c, m))

        if not costs:
            continue

        # Find best and second best costs for regret calculation
        costs.sort(key=lambda x: x[0])
        best_cost = costs[0][0]
        best_m = costs[0][1]
        
        if len(costs) > 1:
            second_cost = costs[1][0]
            regret = second_cost - best_cost
        else:
            regret = 0.0

        # Calculate Urgency
        urgency = URGENCY_RW * rw + URGENCY_RO * ro

        # Combined Score:
        # We want to maximize this score.
        # High urgency is good.
        # High regret is good (forces us to schedule this job now to get its preferred machine).
        # The urgency is scaled by the inverse of the best cost? 
        # If we use score = urgency * (1 + REGRET_W * regret), we favor urgent jobs.
        # If a job has very low cost (fast), it doesn't necessarily make it "more urgent" in terms of makespan,
        # but it might be better to do it now. However, usually, we prioritize "long" jobs to avoid them waiting?
        # Actually, SPT (Shortest Processing Time) is good for mean flow time, but for Makespan, LPT (Longest) or critical path is often better.
        # remaining_work is a proxy for critical path length. High remaining_work => High priority.
        
        # Let's use: score = urgency * (1 + REGRET_W * regret)
        # This ensures that a job with 100 remaining work and high regret is prioritized over a job with 10 remaining work.
        
        score = urgency * (1.0 + REGRET_W * regret)

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

    # Fallback
    if best_index == -1:
        best_index = 0
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None

    if best_machine is not None:
        return (best_index, best_machine)
    return best_index

# EVOLVE-BLOCK-END
