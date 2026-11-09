"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level state to track machine loads across dispatch calls
_machine_loads = {}  # machine -> estimated completion time
_last_time = 0.0
_initialized = False


def _reset_state():
    """Reset state when simulation restarts."""
    global _machine_loads, _last_time, _initialized
    _machine_loads = {}
    _last_time = 0.0
    _initialized = False


def _update_state(ready_operations, time):
    """Update machine load estimates based on current time and dispatched operations."""
    global _machine_loads, _last_time, _initialized
    
    # Detect if simulation restarted (time went backwards or significantly forward)
    if not _initialized or time < _last_time - 1.0:
        _reset_state()
        _initialized = True
    
    _last_time = time
    
    # Update machine loads: any machine not in ready_operations' options may be busy
    # We estimate loads by tracking when machines were last used
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine in machine_options:
            if machine not in _machine_loads:
                _machine_loads[machine] = time  # Assume available now if never seen
    
    # Machines that are NOT in any ready operation's options are likely busy
    # Their load doesn't change (we can't observe them directly)
    # But we can decay our estimate slightly as time passes
    for machine in list(_machine_loads.keys()):
        if _machine_loads[machine] < time:
            # Machine was available at some point, but we don't know current state
            # Keep the last known time as a lower bound
            pass
        else:
            # Machine is still busy from our last observation
            pass


def dispatch(ready_operations, time):
    """
    Critical path-based dispatch rule with machine selection.
    
    Strategy:
    1. Compute a "critical path score" for each ready operation:
       - Longest remaining path (sum of minimum durations across remaining ops in job)
       - Weighted by flexibility (number of machine options)
       - Plus a load-balancing term based on machine availability
    2. Select the operation with the highest priority score.
    3. Optionally return (index, machine) where machine minimizes completion time.
    
    The rule aims to:
    - Prioritize jobs on the critical path to reduce makespan
    - Choose machines that minimize completion time (load balancing)
    - Reward flexibility to avoid bottlenecks
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Track elapsed time for safety
    t_start = _time.time()
    deadline = 150.0  # seconds, with safety margin
    safety_margin = 10.0  # return before this many seconds remaining
    
    # Update state with current time
    _update_state(ready_operations, time)
    
    # Build set of machines that are currently eligible (appear in at least one ready op)
    eligible_machines = set()
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        eligible_machines.update(machine_options.keys())
    
    # Estimate effective completion time for each machine considering current load
    def get_machine_completion(machine, duration):
        """Estimate when an operation would complete on a given machine."""
        current_load = _machine_loads.get(machine, time)
        # If machine is busy until current_load, operation starts at max(time, current_load)
        start_time = max(time, current_load)
        completion = start_time + duration
        return completion
    
    # Compute scores for each ready operation
    scores = np.zeros(n_ops)
    
    # First pass: compute base critical path score with load-aware machine selection
    for i, op in enumerate(ready_operations):
        # Longest remaining path: sum of minimum durations across all remaining ops in this job
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Critical path estimate: use remaining_work as proxy for longest path
        # This is the sum of durations along the critical path
        crit_path = remaining_work
        
        # Flexibility: number of machine options
        machine_options = op.get("machine_options", {})
        n_machines = len(machine_options)
        flexibility = np.log1p(n_machines) if n_machines > 0 else 0.0
        
        # Minimum duration on any machine (for this operation)
        if n_machines > 0:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
        
        # Average duration across machines
        if n_machines > 0:
            avg_dur = np.mean(list(machine_options.values()))
        else:
            avg_dur = 0
        
        # Load-aware: find the best machine considering current load
        best_completion = np.inf
        best_machine = None
        for machine, duration in machine_options.items():
            completion = get_machine_completion(machine, duration)
            if completion < best_completion:
                best_completion = completion
                best_machine = machine
        
        # If no valid machine found, use min duration as fallback
        if best_completion == np.inf:
            best_completion = time + min_dur if min_dur > 0 else time
        
        # Score components:
        # 1. Critical path importance (higher = more urgent)
        w_cp = 0.5
        # 2. Flexibility (more options = less bottleneck risk)
        w_flex = 0.2
        # 3. Load-aware completion time (earlier completion = better)
        w_load = 0.15
        # 4. Remaining ops count (jobs with more remaining work are more critical)
        w_ops = 0.15
        
        # Normalize completion time relative to current time
        time_horizon = max(best_completion - time, 1.0)
        load_score = 1.0 / (1.0 + time_horizon)  # Earlier completion = higher score
        
        score = (
            w_cp * (crit_path / (1.0 + avg_dur)) +  # Normalize by avg duration
            w_flex * flexibility +
            w_load * load_score +  # Prefer operations that can complete soon
            w_ops * (remaining_ops / (1.0 + remaining_ops))
        )
        
        scores[i] = score
    
    # Check time
    if _time.time() - t_start > deadline - safety_margin:
        # Return best found so far
        return int(np.argmax(scores))
    
    # Machine selection: for the top candidate, pick machine that minimizes completion time
    best_idx = int(np.argmax(scores))
    
    # Try to improve by checking machine selection for top candidates
    top_k = min(5, n_ops)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    best_score = -np.inf
    best_idx_final = best_idx
    best_machine = None
    best_completion_final = np.inf
    
    for idx in top_indices:
        op = ready_operations[idx]
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        # For each machine, estimate completion time using load tracking
        best_machine_for_op = None
        best_completion = np.inf
        
        for machine, duration in machine_options.items():
            completion = get_machine_completion(machine, duration)
            if completion < best_completion:
                best_completion = completion
                best_machine_for_op = machine
        
        # Adjust score based on machine selection quality
        # Stronger penalty for congested machines
        adjusted_score = scores[idx]
        if best_machine_for_op is not None:
            # Bonus for early completion, scaled by how much earlier than worst case
            time_horizon = max(best_completion - time, 1.0)
            adjusted_score += 0.15 * (1.0 / (1.0 + time_horizon))
        
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_idx_final = int(idx)
            best_machine = best_machine_for_op
            best_completion_final = best_completion
    
    # Update machine load for the selected operation
    if best_machine is not None and best_idx_final is not None:
        selected_op = ready_operations[best_idx_final]
        selected_duration = selected_op.get("machine_options", {}).get(best_machine, 0)
        if selected_duration > 0:
            _machine_loads[best_machine] = best_completion_final
    
    # Check time again
    if _time.time() - t_start > deadline - safety_margin:
        return int(best_idx_final)
    
    # Return index (and optionally machine)
    # The interface says "optionally (index, machine)", so return just index for safety
    return int(best_idx_final)
# EVOLVE-BLOCK-END
