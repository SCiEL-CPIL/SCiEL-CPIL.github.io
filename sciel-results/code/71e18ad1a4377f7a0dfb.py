import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 148.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0, 0 # Fallback, shouldn't happen

    # Deadline guard: if we're running out of time, return the first operation on its first machine
    if _time.time() - _start_time > _DEADLINE:
        return 0, list(ready_operations[0]['machine_options'].keys())[0]

    # Calculate contention for each machine among ready operations
    # We count how many ready operations can use each machine.
    # This is a proxy for future congestion.
    machine_contention = {}
    for op in ready_operations:
        for m in op['machine_options']:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Constraint tightness: number of machine options (fewer = more constrained)
        num_options = len(op['machine_options'])
        
        # Use remaining_work as a measure of urgency/criticality
        remaining_work = op.get('remaining_work', 1.0)
        remaining_ops = op.get('remaining_ops', 1)
        
        # Ensure we don't divide by zero or have negative work
        if remaining_work <= 0:
            remaining_work = 1.0
        if remaining_ops <= 0:
            remaining_ops = 1
        
        # Constraint tightness factor: operations with fewer options are more critical
        constraint_factor = 1.0 + (5.0 - num_options) * 0.3
        
        # Critical path estimate
        avg_work_per_op = remaining_work / remaining_ops

        # Strategy: Model as dynamic resource contention.
        # We want to prioritize operations that can be executed most efficiently
        # on the least congested resources.
        
        # For this operation, we look at all its eligible machines.
        # We compute a "smooth minimum" of the effective processing times.
        # Effective time = duration * (1 + congestion_penalty)
        # We use log-sum-exp to approximate the min, which is smooth and differentiable.
        
        min_eff_time = float('inf')
        best_m_for_op = None
        best_m_eff_time = float('inf')

        # Collect effective times for log-sum-exp
        eff_times = []
        machines = []
        
        for m, duration in op['machine_options'].items():
            if duration <= 0:
                continue
            
            contention = machine_contention.get(m, 0)
            
            # Congestion penalty: 
            # As contention increases, the effective time increases.
            # We use a linear or sqrt penalty. Let's use a slightly stronger penalty 
            # to encourage spreading load.
            # load_factor = 1.0 + 0.5 * np.sqrt(contention)
            # Or perhaps a linear penalty is better for "least congested" strategy?
            # Let's try: effective_time = duration * (1 + contention * alpha)
            # If alpha is small, it's soft. If alpha is large, it's hard avoidance.
            # Let's stick with the sqrt from the parent but maybe tweak the coefficient.
            # The directive suggests: "priority score inversely proportional to the sum of the minimum available processing times"
            # Wait, the directive says: "inversely proportional to the sum of the minimum available processing times across all its eligible machines"
            # This is slightly ambiguous. Does it mean sum over machines of min(time, something)?
            # Or does it mean: for each machine, take the min time? No, duration is fixed per machine.
            # "sum of the minimum available processing times" might imply looking at global state?
            # No, "across all its eligible machines".
            # Let's interpret: We want to pick the op that has the smallest "best possible time" considering congestion.
            
            # Let's compute effective time for this machine.
            # We want to penalize busy machines.
            load_factor = 1.0 + 0.5 * np.sqrt(contention)
            eff_time = duration * load_factor
            
            eff_times.append(eff_time)
            machines.append(m)

        if not eff_times:
            continue

        eff_times_arr = np.array(eff_times)
        
        # Smooth min using log-sum-exp:
        # min(x) approx -log(sum(exp(-x)))
        # To avoid overflow, we can scale. But durations are usually small.
        # Let's just use the actual min for machine selection, but use the smooth min for scoring?
        # The directive says: "Select the machine for the chosen operation that minimizes the resulting pr..." (cut off, likely "processing time" or "priority").
        # And: "assigning each ready operation a priority score inversely proportional to the sum of the minimum available processing times"
        
        # Let's re-read carefully: "priority score inversely proportional to the sum of the minimum available processing times across all its eligible machines"
        # This might mean: Score = 1 / (Sum over machines of min(duration, estimated_wait + duration)?)
        # Or simply: Score = 1 / (Smooth Min of Effective Times)
        
        # Let's calculate the smooth minimum effective time for this operation.
        # This represents the "best case" time this operation can take if we pick the best machine.
        # We want to prioritize operations that have a SMALL smooth min effective time?
        # No, usually we want to prioritize operations that are CRITICAL (high remaining work) 
        # but can be done QUICKLY (low effective time).
        
        # If we use Score = (Urgency * Constraint) / SmoothMinEffTime, 
        # we prioritize urgent, constrained ops that can be done quickly on a free machine.
        
        # Let's compute smooth min.
        # To be safe against overflow in exp, we subtract the max.
        # But for min, we want exp(-x). So we add the max of -x (which is -min).
        # Let's just use the raw min for now to keep it simple and robust, 
        # as the "smoothness" might not be the main driver of improvement here, 
        # but the "least congested resource" selection is.
        
        # Actually, let's stick to the raw min for machine selection to be deterministic and fast,
        # but use the average or sum of effective times for the score to reflect total resource burden?
        
        # Let's try: Score = (remaining_work * constraint_factor * avg_work_per_op) / (min_effective_time)
        # This favors ops that are urgent and have a very fast, low-contention option.
        
        min_eff_time_val = np.min(eff_times_arr)
        # Find the machine corresponding to this min
        best_m_idx = np.argmin(eff_times_arr)
        candidate_machine = machines[best_m_idx]
        
        # Calculate score
        # We want high score for:
        # 1. High remaining work (urgency)
        # 2. High constraint factor (fewer options)
        # 3. High avg work per op (critical path)
        # 4. Low min effective time (efficiency on best available machine)
        
        # To avoid division by zero or very large scores if time is tiny, we can add a small epsilon.
        score = (remaining_work * constraint_factor * np.sqrt(avg_work_per_op)) / (min_eff_time_val + 1e-9)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = candidate_machine

    # Fallback if no valid option found
    if best_machine is None:
        return 0, list(ready_operations[0]['machine_options'].keys())[0]

    return best_idx, best_machine
