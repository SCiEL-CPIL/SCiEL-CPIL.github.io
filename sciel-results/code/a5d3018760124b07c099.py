"""Flexible job-shop dispatch: load-balancing bottleneck minimization."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Dispatch rule: real-time load-balancing via bottleneck minimization.
    
    For each candidate operation, estimate the resulting machine load profile
    and select the operation whose assignment minimizes the maximum machine
    load (bottleneck), preventing any single machine from becoming a
    critical-path constraint.
    
    Returns the index of the operation to start next.
    """
    if len(ready_operations) == 0:
        return 0
    if len(ready_operations) == 1:
        return 0
    
    # Safety: if we're close to deadline, just pick argmax remaining_work
    if _elapsed() > _DEADLINE - 2.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    
    # Collect all unique machines across all ready operations
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    all_machines = sorted(all_machines)
    n_machines = len(all_machines)
    
    if n_machines == 0:
        return 0
    
    # Build machine index mapping
    machine_idx = {m: i for i, m in enumerate(all_machines)}
    
    # For each machine, estimate its current "load" from the sum of processing
    # times of all ready operations that can use it. This is a proxy for
    # how congested each machine will be.
    # We'll compute for each operation, for each of its machine options,
    # what the resulting bottleneck would be if we assigned that operation
    # to that machine.
    
    # Pre-compute: for each machine, the total estimated load if we assign
    # all ready operations optimally. But since we're deciding one at a time,
    # we estimate the "remaining load" on each machine from the OTHER ready ops.
    
    # Approach: For each candidate operation i, for each machine m in op i's options:
    #   - The load on machine m becomes: duration_i_on_m + sum of estimated loads from other ops
    #   - For other ops, estimate their load contribution to machine m as:
    #     if op j can use m, then its duration on m contributes to m's load
    #     but we don't know which machine op j will use, so we use a weighted estimate
    #
    # Simpler structural approach:
    #   1. For each machine, compute the "potential load" = sum over all ready ops
    #      that can use that machine of (min duration on that machine across options)
    #      This gives an upper bound on how much work could hit each machine.
    #   2. For each candidate op, the bottleneck after assigning it to its best machine
    #      is: max over machines of (machine_load_after_assignment)
    #   3. Pick the op whose best assignment leads to the smallest bottleneck.
    
    # Step 1: For each machine, compute potential load from all ready ops
    # potential_load[m] = sum over ops that can use m of min(duration on m, other options)
    # Actually, let's use: for each op, its "preferred" duration is the min across its machines.
    # But for load on a specific machine, use the actual duration on that machine if the op uses it.
    
    # Let's do: for each machine m, est_load[m] = sum of durations of all ready ops on m
    # (assuming each op is assigned to m if m is in its options -- this overestimates)
    # Better: est_load[m] = sum over ops j that can use m of (duration_j_on_m / num_options_j)
    # This distributes each op's work evenly across its available machines.
    
    est_load = np.zeros(n_machines, dtype=np.float64)
    for op in ready_operations:
        mo = op["machine_options"]
        n_opts = len(mo)
        for m, dur in mo.items():
            if m in machine_idx:
                est_load[machine_idx[m]] += dur / n_opts
    
    # Current bottleneck (before assigning any new op)
    current_bottleneck = np.max(est_load) if n_machines > 0 else 0.0
    
    # For each candidate operation, compute the best bottleneck reduction
    # or the minimum resulting bottleneck when we assign it to one of its machines.
    
    best_score = np.full(n_ops, np.inf, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            best_score[i] = 0.0
            continue
        
        # For each machine option, compute the resulting bottleneck
        min_bottleneck = np.inf
        
        for m, dur in mo.items():
            if m not in machine_idx:
                continue
            mi = machine_idx[m]
            
            # New load on machine m if we assign op i to m
            # The op's contribution to m is dur (not dur/n_opts, since we're actually assigning it)
            # Other ops still contribute dur/n_opts to m
            # So new_load_m = est_load[mi] + dur - dur/n_opts (remove the fractional part, add full)
            n_opts = len(mo)
            new_load_m = est_load[mi] + dur - dur / n_opts
            
            # New bottleneck = max of all machine loads with m updated
            # We can compute this efficiently
            # The new bottleneck is max(current_bottleneck, new_load_m) 
            # unless current_bottleneck was on machine m and new_load_m < old load on m
            old_load_m = est_load[mi]
            if new_load_m >= current_bottleneck:
                bottleneck_after = new_load_m
            else:
                # Bottleneck might be on another machine
                # Find max load excluding machine m
                if n_machines > 1:
                    # Create a view excluding m
                    # For efficiency with small n_machines, just compute
                    max_other = 0.0
                    for k in range(n_machines):
                        if k != mi:
                            if est_load[k] > max_other:
                                max_other = est_load[k]
                    bottleneck_after = max(new_load_m, max_other)
                else:
                    bottleneck_after = new_load_m
            
            if bottleneck_after < min_bottleneck:
                min_bottleneck = bottleneck_after
        
        if min_bottleneck == np.inf:
            min_bottleneck = current_bottleneck
        
        # We want to MINIMIZE the bottleneck, so lower is better
        # But we also want to consider urgency: operations with more remaining work
        # on the critical path should be prioritized
        # Combine: score = bottleneck_after - alpha * remaining_work
        # Lower score = better (we pick the min)
        
        # Actually, let's think about this differently.
        # The goal is to prevent any machine from becoming a bottleneck.
        # If assigning op i to its best machine results in a lower bottleneck than
        # assigning other ops, then op i is the one that "relieves" or "doesn't worsen"
        # the bottleneck the most.
        #
        # Alternative: score = min_bottleneck / (remaining_work + 1)
        # This favors ops that both reduce bottleneck AND have high urgency.
        
        remaining_work = op["remaining_work"]
        
        # The key insight: we want to pick the op that, when scheduled,
        # leads to the most balanced load profile. An op with a low min_bottleneck
        # is "safe" to schedule. An op with high remaining_work is urgent.
        #
        # Let's use: score = min_bottleneck - 0.5 * remaining_work
        # Pick the op with the LOWEST score (most balanced + most urgent).
        
        best_score[i] = min_bottleneck - 0.5 * remaining_work
    
    # Pick the operation with the lowest score (best load balance + urgency)
    return int(np.argmin(best_score))
# EVOLVE-BLOCK-END
