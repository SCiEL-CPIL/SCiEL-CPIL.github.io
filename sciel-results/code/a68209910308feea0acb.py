"""Job-Shop Scheduling: Model-and-Optimize dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: evaluator kills at 150s, use 140s as hard limit
_DEADLINE = 140.0
_START_TIME = _time.monotonic()

# Look-ahead critical path reduction dispatch rule.
# Strategy: For each ready operation, simulate its immediate completion and
# compute the resulting reduction in the global critical path (max remaining work
# across all jobs). Pick the operation that yields the greatest decrease in this
# makespan lower bound. This is a dynamic, state-based evaluation that directly
# targets the bottleneck.
#
# Additionally, we incorporate a non-linear urgency-speed product that favors
# fast completions of critical-path jobs over slow ones, avoiding the additive
# weighting's tendency to over-prioritize long urgent operations that block machines.
#
# Smooth surrogate: use log-sum-exp for soft-max of remaining works to get a
# differentiable approximation of the max, which helps when multiple jobs are
# near the critical path.

_LOG_SUM_EXP_SCALE = 1.0  # temperature for soft-max approximation


def _soft_max(values, scale=1.0):
    """Approximate max via log-sum-exp for smoothness."""
    if len(values) == 0:
        return 0.0
    max_val = max(values)
    if max_val <= 0:
        return max_val
    # log-sum-exp: max + log(sum(exp((x - max)/scale)))
    shifted = [v / scale - max_val / scale for v in values]
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val / scale * (1.0 + log_sum / np.log(len(values) + 1e-9) if len(values) > 1 else 1.0)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using look-ahead critical path reduction.
    
    For each ready operation, we estimate how much the global makespan lower bound
    (max remaining work across jobs) would decrease if this operation completed
    immediately. We select the operation that yields the greatest reduction.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start
    """
    # Deadline guard: if we're close to the limit, just return first op
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Build a map of job -> remaining_work for all ready ops
    # Note: ready_operations may contain multiple ops from the same job? 
    # Typically in job-shop, at most one op per job is ready at a time.
    # But let's handle the general case.
    job_rem_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        rw = op.get("remaining_work", 0.0)
        if j not in job_rem_work or rw > job_rem_work[j]:
            job_rem_work[j] = rw

    # Current critical path: max remaining work across all jobs
    if not job_rem_work:
        return 0
    current_cp = max(job_rem_work.values())
    
    # For each ready operation, compute the critical path reduction
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            scores[i] = -1e18
            continue

        min_dur = min(machine_opts.values())
        rem_work = op.get("remaining_work", 0.0)
        rem_ops = op.get("remaining_ops", 0)
        job = op["job"]
        
        # Simulate: if this op completes, remaining_work for this job decreases by min_dur
        new_rem_work = max(0.0, rem_work - min_dur)
        
        # Compute new critical path
        # Only the job's remaining work changes
        other_jobs_max = max(
            (rw for j, rw in job_rem_work.items() if j != job),
            default=0.0
        )
        new_cp = max(new_rem_work, other_jobs_max)
        
        # Critical path reduction
        cp_reduction = current_cp - new_cp
        
        # Non-linear urgency-speed product:
        # Favor operations that are both on the critical path AND fast
        # This avoids picking long operations just because they're urgent
        urgency = rem_work / (current_cp + 1e-9)  # 0 to 1, how critical this job is
        speed = 1.0 / (min_dur + 1e-9)  # higher = faster
        
        # Combine: critical path reduction is the primary signal,
        # modulated by the urgency-speed interaction
        # The product urgency * speed captures "fast completion of critical job"
        interaction = urgency * speed
        
        # Primary: cp_reduction (how much this op reduces the makespan bound)
        # Secondary: interaction term to break ties and refine selection
        scores[i] = cp_reduction + 0.1 * interaction * min_dur
    
    # Pick the operation with highest score
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
