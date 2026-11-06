"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock.
# We track a global start time and enforce a safety margin.
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds

def _elapsed():
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # Extract key features for each ready operation
    # We'll compute a score for each operation and pick the best one.
    # The score should balance:
    # 1. Constraint tightness: operations with fewer machine options are more constrained
    # 2. Remaining work (urgency: more work = higher priority)
    # 3. Minimum processing time (shortest processing time first reduces makespan)
    # 4. Number of remaining operations in the job (critical path consideration)
    
    best_score = float('-inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            # No machines available; assign lowest priority
            score = float('-inf')
        else:
            # Compute features
            durations = list(machine_options.values())
            min_duration = min(durations) if durations else 1.0
            avg_duration = sum(durations) / len(durations) if durations else 1.0
            
            # Use soft-min as a smooth surrogate for min duration
            # This penalizes operations where all machines are slow, not just the fastest one
            if durations:
                d_arr = np.array(durations, dtype=float)
                min_d = np.min(d_arr)
                # logsumexp(-d_arr) = -min_d + log(sum(exp(-(d_arr - min_d))))
                # soft_min = -logsumexp(-d_arr) = min_d - log(sum(exp(-(d_arr - min_d))))
                exp_terms = np.exp(-(d_arr - min_d))
                log_sum = np.log(np.sum(exp_terms))
                soft_min_duration = min_d - log_sum
            else:
                soft_min_duration = 1.0

            if min_duration <= 0:
                min_duration = 1.0
            if avg_duration <= 0:
                avg_duration = 1.0
            if soft_min_duration <= 0:
                soft_min_duration = 1.0
            
            # Feature 1: Constraint tightness - fewer machine options = higher priority
            # This prevents future starvation by prioritizing constrained operations
            num_options = len(machine_options)
            # Use inverse squared to amplify the effect of tight constraints
            f1 = 1.0 / (num_options ** 2)
            
            # Feature 2: Remaining work (urgency)
            f2 = remaining_work
            
            # Feature 3: Shortest processing time (SPT) using soft-min
            f3 = 1.0 / soft_min_duration
            
            # Feature 4: Remaining ops in job
            if remaining_ops > 0:
                f4 = 1.0 / remaining_ops
            else:
                f4 = 1.0
            
            # Feature 5: Weighted Shortest Processing Time (WSPT) proxy
            f5 = f2 * f3
            
            # Feature 6: Bottleneck mitigation - spread of processing times across machines
            # A large spread means the operation is flexible; a small spread means it's constrained
            # to similar-duration machines, making it more likely to cause bottlenecks
            if len(durations) > 1:
                d_arr = np.array(durations, dtype=float)
                spread = np.std(d_arr)
                # Normalize: smaller spread relative to mean = more constrained
                f6 = 1.0 / (1.0 + spread / max(avg_duration, 1e-9))
            else:
                # Single machine option: maximally constrained
                f6 = 1.0
            
            # Feature 7: Projected machine idle time consideration
            # Estimate which machine would have the least projected idle time
            # by looking at the ratio of min duration to average duration
            # Operations where the best machine is much faster than alternatives
            # are less likely to cause bottlenecks on that specific machine
            if len(durations) > 1:
                f7 = min_duration / avg_duration
            else:
                f7 = 1.0
            
            # Weighted combination of features
            # Increased weight on constraint tightness (f1), bottleneck mitigation (f6),
            # and WSPT (f5) to better model bottleneck mitigation
            w1 = 5.0  # constraint tightness (squared inverse)
            w2 = 1.0  # remaining work
            w3 = 2.0  # SPT (soft)
            w4 = 0.5  # remaining ops
            w5 = 4.0  # work * SPT combo
            w6 = 3.0  # bottleneck mitigation (spread-based)
            w7 = 1.0  # projected idle time consideration
            
            score = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5 + w6 * f6 + w7 * f7
            
            # Track the best machine for this operation (minimum duration)
            if score > best_score:
                best_score = score
                best_index = i
                # Find the machine with minimum duration for this operation
                best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
    
    return int(best_index)
# EVOLVE-BLOCK-END
