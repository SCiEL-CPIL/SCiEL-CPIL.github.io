"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global start time for the evaluation run
_eval_start_time = time.time()
# Safety margin: 120s limit, stop at 115s to avoid timeout
_DEADLINE = 115.0

def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    Returns (index, machine) to explicitly select the fastest machine for the chosen operation.
    Heuristic: Most Work Remaining (MWR) + Longest Processing Time (LPT) of current op.
    """
    # Deadline guard
    if time.time() - _eval_start_time > _DEADLINE:
        # Fallback to a simple, fast choice if we are running out of time
        if ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    best_index = 0
    best_score = -1
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Get remaining work for the job
        remaining_work = op.get("remaining_work", 0)
        
        # Get machine options for this operation
        machine_opts = op.get("machine_options", {})
        
        if not machine_opts:
            # Should not happen, but handle gracefully
            score = remaining_work
            min_dur = 0
            best_mach = None
        else:
            # Find the machine with the minimum duration for this operation
            # This is the "LPT" component for the current step (actually SPT on machine, but prioritizing long ops overall)
            # We want to prioritize operations that are long, so we add the min duration.
            min_dur = min(machine_opts.values())
            best_mach = min(machine_opts, key=machine_opts.get)
            
            # Composite Score:
            # 1. remaining_work: Prioritize jobs with more total work left (MWR)
            # 2. min_dur: Prioritize operations that take a long time to process (LPT)
            # This helps clear bottlenecks early.
            score = remaining_work + min_dur

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_mach

    # Return the index and the best machine for that operation
    # The evaluator accepts (index, machine) optionally
    if best_machine is not None:
        return (best_index, best_machine)
    else:
        return best_index
# EVOLVE-BLOCK-END
