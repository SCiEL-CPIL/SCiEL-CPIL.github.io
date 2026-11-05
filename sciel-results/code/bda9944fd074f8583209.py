"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import defaultdict

# Deadline guard: the evaluator kills after 150s wall-clock.
_DEADLINE = _time.monotonic() + 140.0

# Track machine busy-until times (when each machine becomes free)
_machine_busy_until = defaultdict(float)
# Track last seen time to detect resets between instances
_last_time = -1.0
# Track job-level remaining work for critical path estimation
_job_remaining_work = defaultdict(float)


def dispatch(ready_operations, time):
    """
    Dispatch rule using look-ahead critical path estimation.
    
    Structural change: Instead of a weighted sum of local features, we estimate
    the earliest completion time (makespan contribution) for each candidate
    operation by computing a lower bound on when the job's remaining work can
    finish, considering current machine contention. We pick the operation whose
    selection minimizes the projected makespan increase.
    """
    global _last_time, _machine_busy_until, _job_remaining_work
    
    if not ready_operations:
        return 0
    
    # Reset state if time goes backwards (new instance)
    if time < _last_time - 1e-9:
        _machine_busy_until = defaultdict(float)
        _job_remaining_work = defaultdict(float)
    _last_time = time
    
    # Check deadline
    if _time.monotonic() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # For each ready operation, estimate the projected makespan if we start it now
    best_idx = 0
    best_projected_makespan = float('inf')
    best_machine_for_op = None
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        
        # Estimate when this job can finish its remaining work
        # The key insight: the critical path through remaining operations
        # is bounded by max(sequential work / 1, parallel work / num_machines_available)
        
        # For each eligible machine, compute the earliest start and finish
        best_op_score = float('inf')
        best_machine = None
        
        for machine, duration in op["machine_options"].items():
            # When can this machine start this op?
            machine_free_at = max(time, _machine_busy_until.get(machine, 0.0))
            op_finish = machine_free_at + duration
            
            # After this op finishes, estimate remaining job completion
            # remaining_ops includes current op, so after this op:
            ops_after = remaining_ops - 1
            work_after = remaining_work - duration
            
            if ops_after > 0:
                # Lower bound: remaining work must be done sequentially or in parallel
                # Simple lower bound: work_after / max_parallelism
                # But we don't know parallelism, so use a heuristic:
                # Assume remaining work takes at least work_after / remaining_machines
                # where remaining_machines is estimated from machine options
                # For now, use a conservative estimate: remaining work / 1 (sequential)
                # but discounted by number of machines that can handle future ops
                # Approximation: remaining work / sqrt(num_eligible_machines)
                num_machines = len(op["machine_options"])
                if num_machines > 1:
                    # Some parallelism possible
                    estimated_remaining_time = work_after / max(1, num_machines * 0.5)
                else:
                    estimated_remaining_time = work_after
            else:
                estimated_remaining_time = 0.0
            
            # Projected job completion time
            projected_job_completion = op_finish + estimated_remaining_time
            
            # Also consider: this machine is now busy until op_finish
            # Other jobs waiting for this machine are delayed
            # Estimate contention penalty: how many other ready ops want this machine?
            contention = 0
            for j, other_op in enumerate(ready_operations):
                if j != i and machine in other_op["machine_options"]:
                    contention += 1
            
            # Contention penalty: each competing op adds estimated delay
            contention_penalty = contention * 0.5  # scaled down
            
            total_score = projected_job_completion + contention_penalty
            
            if total_score < best_op_score:
                best_op_score = total_score
                best_machine = machine
        
        if best_machine is None and op["machine_options"]:
            best_machine = min(op["machine_options"].keys(), 
                              key=lambda m: op["machine_options"][m])
            best_op_score = float('inf')
        
        # The operation that minimizes projected makespan is best
        if best_op_score < best_projected_makespan:
            best_projected_makespan = best_op_score
            best_idx = i
            best_machine_for_op = best_machine
    
    # Update machine busy-until for the selected operation
    selected_op = ready_operations[best_idx]
    if best_machine_for_op is not None:
        duration = selected_op["machine_options"].get(best_machine_for_op, 0)
        machine_free_at = max(time, _machine_busy_until.get(best_machine_for_op, 0.0))
        _machine_busy_until[best_machine_for_op] = machine_free_at + duration
    
    # Return (index, machine) to also specify machine assignment
    if best_machine_for_op is not None:
        return (int(best_idx), best_machine_for_op)
    else:
        return int(best_idx)
# EVOLVE-BLOCK-END
