"""Dispatch rule for open-shop scheduling: optimize machine load balance via Hungarian assignment."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 115.0  # safety margin before 120s wall-clock limit
_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        # Fallback: pick operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0
    
    # Build cost matrix: rows = operations, cols = machines
    # We need to figure out which machines are available
    # Each op has machine_options {machine: duration}
    
    # Collect all machines that appear in any op's options
    all_machines = set()
    for op in ready_operations:
        for m in op["machine_options"]:
            all_machines.add(m)
    
    machines = sorted(all_machines)
    n_machines = len(machines)
    
    # If number of ops > number of machines, we can't assign all to distinct machines
    # In open shop, at any time we can start at most one op per machine and one per job
    # But ready_operations are all ready to start - we pick ONE to start
    # So the decision is: which single operation to start next?
    
    # The key insight: we want to pick the operation that, when started on its best machine,
    # leads to the best overall schedule. A good heuristic is to minimize the resulting
    # maximum machine load (makespan proxy).
    
    # For each operation, find the machine that would give the best outcome.
    # We don't have explicit machine load info, but we can use remaining_work as a proxy
    # and try to balance by assigning to the "least loaded" machine.
    
    # Since we don't have direct machine load, let's use a different approach:
    # Score each operation by considering:
    # 1. Its remaining_work (longer ops should be started earlier - Longest Processing Time)
    # 2. The number of machines it can run on (more flexibility = lower priority)
    # 3. remaining_ops (jobs with more remaining ops should be prioritized to avoid blocking)
    
    # Actually, let's think about this more carefully.
    # In open-shop scheduling, a common effective rule is:
    # - Prioritize operations from jobs with most remaining work
    # - Among those, pick the operation with the longest processing time on its assigned machine
    # - Assign to the least-loaded machine
    
    # Since we pick ONE operation to start, let's score each candidate:
    # score = remaining_work * weight1 + remaining_ops * weight2 - flexibility * weight3
    
    scores = []
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        n_machines_option = len(op["machine_options"])
        
        # Longest processing time on the best (shortest duration) machine
        min_duration = min(op["machine_options"].values())
        
        # Score: prioritize long remaining work, many remaining ops, short processing time
        # This combines LPT with job-completion urgency
        s = rem_work * 2.0 + rem_ops * 1.0 - min_duration * 0.5
        scores.append(s)
    
    scores = np.array(scores)
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
