"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0

def _check_deadline():
    return _time.monotonic() >= _DEADLINE

def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # 1. Extract features
    # We need:
    # - machine_id: The machine chosen for this op (we assume we pick the one with min duration to minimize blocking)
    # - duration: The duration on that chosen machine
    # - job_id: To identify which job is unblocked
    # - remaining_work: To estimate the "weight" of the job's future impact (proxy for critical path length)
    
    machines = [0] * n
    durations = [0.0] * n
    jobs = [0] * n
    rws = [0.0] * n

    for i, op in enumerate(ready_operations):
        jobs[i] = op['job']
        rws[i] = op['remaining_work']
        opts = op['machine_options']
        if not opts:
            continue
        # Choose the machine that minimizes duration (SPT-like choice for the machine assignment)
        # This is a local decision: use the fastest available machine for this op.
        best_m, best_d = min(opts.items(), key=lambda x: x[1])
        machines[i] = best_m
        durations[i] = best_d

    # 2. Construct Contention Graph Metrics
    # The "Total Edge Weight" of the graph represents the total potential for conflict.
    # An edge exists between op i and op j if machines[i] == machines[j].
    # The weight of the conflict is estimated by the sum of their durations (or max, depending on overlap).
    # For a greedy dispatch, we assume that if we pick op k:
    # 1. Op k is removed from the ready pool.
    # 2. All edges connected to k are removed.
    # 3. The next op of job[k] becomes ready (but we don't have its data, so we approximate its conflict potential
    #    by assuming it will have a similar machine preference profile, or simply ignore it for the *immediate*
    #    graph reduction calculation, focusing on the removal of current conflicts).
    #
    # To be more robust, we can estimate the "Cost" of the current state as the sum over all machines
    # of the sum of durations of all ready ops assigned to that machine.
    # Let C(m) = sum(d_i for i in ready if machines[i] == m)
    # Total Conflict Potential T = sum_m C(m)
    #
    # If we pick op k (on machine m_k):
    # - C(m_k) decreases by durations[k].
    # - The next op of job[k] will eventually be added. Let's estimate its duration as 
    #   avg_remaining_work_per_op for that job? Or just 0 for simplicity?
    #   Actually, a better proxy for "reduction in blocking" is simply the duration of the op we are removing.
    #   However, if we pick a short op, we remove a small amount of load. If we pick a long op, we remove a large amount.
    #   BUT, picking a long op blocks the machine for a long time.
    #
    # Let's refine the "Graph Edge Weight" concept.
    # In a contention graph, the "badness" is often the number of pairs that *could* conflict.
    # But here, all ops on the same machine *do* conflict sequentially.
    # The "pressure" on a machine is the total work.
    #
    # Alternative Structural View: **Regret-Based Conflict Resolution**
    # For each candidate k, calculate the "Conflict Score" of the *remaining* set if k is removed.
    # Conflict Score = Sum over all machines m of (Sum of durations of remaining ops on m)^2 ?
    # Why squared? Because the makespan is often dominated by the busiest machine. 
    # Minimizing the sum of squares of machine loads is a standard proxy for load balancing.
    #
    # Let L_m = sum(d_i for i in ready, machines[i]==m)
    # Objective: Minimize J = sum_m L_m^2
    #
    # If we pick op k (machine m_k, duration d_k):
    # New L_{m_k} = L_{m_k} - d_k
    # New J = J_old - L_{m_k}^2 + (L_{m_k} - d_k)^2
    # Delta J = (L_{m_k} - d_k)^2 - L_{m_k}^2 = L_{m_k}^2 - 2*L_{m_k}*d_k + d_k^2 - L_{m_k}^2 = d_k^2 - 2*L_{m_k}*d_k
    # We want to *minimize* the new J, which means we want the *most negative* Delta J (largest reduction).
    # Minimize (d_k^2 - 2*L_{m_k}*d_k)  <=>  Maximize (2*L_{m_k}*d_k - d_k^2)
    #
    # This metric says: "Pick the op that reduces the load on the most heavily loaded machine the most, 
    # but penalize very long durations (d_k^2) because they might not be as effective at reducing the 
    # *squared* load if the machine isn't that loaded, or because they block the machine."
    #
    # Let's compute L_m for all machines present.
    
    # Identify unique machines
    unique_machines = set(machines)
    
    # Calculate current load per machine
    load = {}
    for m in unique_machines:
        load[m] = 0.0
    for i in range(n):
        load[machines[i]] += durations[i]
    
    # For each candidate, calculate the "Benefit" of removing it
    # Benefit = 2 * L_{m_k} * d_k - d_k^2
    # We pick the candidate with the Max Benefit.
    
    best_idx = 0
    best_benefit = -float('inf')
    
    for i in range(n):
        m = machines[i]
        d = durations[i]
        L = load[m]
        
        # If the machine is only used by this op, L=d, Benefit = 2d^2 - d^2 = d^2.
        # If the machine is heavily loaded, L >> d, Benefit ~ 2*L*d.
        # This effectively prioritizes ops on busy machines, and among them, those with longer durations
        # (up to a point) because they reduce the squared load more.
        
        # However, there is a nuance: "Remaining Work" urgency.
        # If a job has very little remaining work, it's not critical.
        # If a job has a lot, it's critical.
        # Let's blend the Load Reduction Benefit with Job Urgency.
        
        # Urgency Factor: rws[i] / (rws[i] + 1)  (saturates at 1)
        # Or just use rws[i] directly if scales are similar.
        
        # Let's stick to the pure structural "Load Balancing via Conflict Graph Reduction" first.
        # This is fundamentally different from "Shortest Processing Time" or "Most Work Remaining".
        # It is a **Global Load Balancing** heuristic.
        
        benefit = 2.0 * L * d - d * d
        
        # Tie-breaker: If benefits are close, prefer the job with more remaining work (urgency)
        # to ensure critical jobs progress.
        # We can add a small epsilon * rws[i] to the benefit.
        # But rws and d might have different scales.
        # Let's normalize rws roughly.
        
        if benefit > best_benefit:
            best_benefit = benefit
            best_idx = i
        elif abs(benefit - best_benefit) < 1e-6:
            # Tie-break: Higher remaining work is better
            if rws[i] > rws[best_idx]:
                best_idx = i

    return best_idx

# EVOLVE-BLOCK-END
