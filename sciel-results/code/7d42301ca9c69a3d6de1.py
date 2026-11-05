"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import sys

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

# State tracking for Machine Load Balancing
# We track the estimated total load on each machine to prefer less congested ones.
# Since we don't get direct machine availability signals, we approximate load
# by the sum of durations of operations assigned to that machine so far.
_machine_load = {}
_machine_count = 0

def dispatch(ready_operations, time):
    """
    Dispatch rule: Dynamic Machine Load Balancing with SPT Tie-Breaking.
    
    Strategy:
    1. For each ready operation, identify the best machine (shortest duration).
    2. Estimate the 'congestion' of that machine based on historical load.
    3. Prefer operations that can be assigned to the least congested machine.
    4. If congestion is similar, prefer the shortest processing time (SPT).
    
    This differs from the previous greedy CPM/SPT approach by explicitly
    modeling the system-wide resource contention. By steering jobs towards
    less loaded machines, we aim to balance the load and reduce the makespan
    caused by a single overloaded bottleneck machine.
    """
    global _machine_load, _machine_count
    
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    best_idx = 0
    best_score = float('inf')
    
    # We need to pick the operation that leads to the best "system state".
    # Proxy: Minimize (Machine_Load + Duration).
    # If we assign op i to machine m, the new load on m increases by duration.
    # We want to minimize the resulting load on the machine we are using,
    # or more generally, we want to avoid adding to already heavy machines.
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the best machine for this operation
        # "Best" is defined as the one with the minimum (current_load + duration)
        # This naturally prefers shorter durations (SPT) but also avoids
        # machines that are already heavily loaded.
        
        best_machine_score = float('inf')
        best_machine = None
        
        for m, d in machine_options.items():
            current_load = _machine_load.get(m, 0)
            # The "cost" of assigning this op to this machine now
            # is the resulting load on that machine.
            # Using (current_load + d) as a proxy for future waiting time
            # on that machine.
            score_candidate = current_load + d
            
            # We also want to consider the total system load impact?
            # No, local machine load is the primary bottleneck driver in JSS.
            
            if score_candidate < best_machine_score:
                best_machine_score = score_candidate
                best_machine = m

        if best_machine is None:
            continue

        # The score for this operation is the estimated completion time
        # or the resulting load on the chosen machine.
        # To break ties between operations that choose the same "good" machine,
        # we can look at the remaining work or just the duration.
        
        # Let's use the resulting load as the primary score.
        # Lower is better.
        score = best_machine_score
        
        # Add a small penalty for long remaining work? 
        # The previous heuristic used remaining_ops. 
        # Let's stick to the load balancing core first.
        # If two ops have the same best_machine_score, prefer the one with 
        # shorter remaining work to clear the job? 
        # Actually, standard SPT is a strong baseline. 
        # The "current_load + d" term effectively makes SPT the tie-breaker 
        # when loads are equal.
        
        if score < best_score:
            best_score = score
            best_idx = i

    # Update the global state for the chosen operation
    # Note: This is a heuristic simulation of the state.
    # We assume the chosen operation will indeed be started.
    chosen_op = ready_operations[best_idx]
    machine_options = chosen_op.get("machine_options", {})
    
    # Re-find the best machine for the chosen op to update state
    best_machine_score = float('inf')
    best_machine = None
    for m, d in machine_options.items():
        current_load = _machine_load.get(m, 0)
        score_candidate = current_load + d
        if score_candidate < best_machine_score:
            best_machine_score = score_candidate
            best_machine = m

    if best_machine is not None:
        d = machine_options[best_machine]
        _machine_load[best_machine] = _machine_load.get(best_machine, 0) + d
        _machine_count += 1

    return int(best_idx)
# EVOLVE-BLOCK-END
