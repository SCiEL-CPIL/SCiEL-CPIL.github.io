"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    _start = _time.perf_counter()
    _deadline = _start + 115.0

    # 1. Estimate current machine loads based on remaining work of all ready operations
    # This approximates the "contention" each machine will face.
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            # Use average duration as a proxy for load contribution if multiple options
            # Actually, better to just sum the min duration for each machine?
            # No, let's sum the duration for each machine option as potential load.
            # But an operation only runs on ONE machine.
            # A better proxy: For each machine, sum the durations of all operations that COULD run on it.
            # This represents the "potential" work.
            if m not in machine_loads:
                machine_loads[m] = 0.0
            machine_loads[m] += dur

    best_pair_score = float('inf')
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        remaining_work = op.get("remaining_work", 0)
        
        if not machine_options:
            continue

        # For each eligible machine, calculate the projected finish time
        # if we start this operation now on that machine.
        # Projected Finish Time = current_time + machine_current_load + op_duration
        # Wait, the simulation is event-driven. `time` is the current event time.
        # The machine is likely free at `time` if it's in ready_operations? 
        # Not necessarily. But `ready_operations` usually implies the machine is available or the op is ready.
        # In FJSP dispatching, we usually assume the machine is available at `time` for the chosen op.
        # However, other ops might be queued.
        
        # A common heuristic: Select the pair (i, m) that minimizes the resulting makespan contribution.
        # Approximation: minimize (machine_loads[m] + duration).
        # This balances the load across machines.
        
        for m, dur in machine_options.items():
            load = machine_loads.get(m, 0.0)
            
            # We want to minimize the completion time of this specific operation on this machine,
            # but also consider the impact on the global makespan.
            # Minimizing the max projected completion time is hard to do greedily per step.
            # Instead, let's minimize the "critical path" contribution.
            
            # Heuristic: Prioritize operations that reduce the longest tail.
            # Score = projected_finish_time - remaining_work
            # If remaining_work is high, we want to start it early.
            
            # Let's use a weighted score:
            # 1. Projected finish time (lower is better)
            # 2. Urgency (remaining work / duration)
            
            # Simple Load Balancing Heuristic:
            # Choose the pair that results in the lowest "completion time" relative to the machine's load.
            # Actually, just minimizing the projected finish time (load + dur) is a good proxy for balancing.
            
            projected_finish = load + dur
            
            # Add a small tie-breaker for remaining work to prioritize jobs with more total work left
            # This helps in clearing long jobs earlier.
            tie_breaker = -remaining_work * 0.01
            
            score = projected_finish + tie_breaker

            if score < best_pair_score:
                best_pair_score = score
                best_index = i
                best_machine = m

        # Deadline check
        if i % 50 == 0 and _time.perf_counter() > _deadline:
            break

    # If no valid machine was found (should not happen), return first
    if best_machine is None:
        return 0

    return best_index
# EVOLVE-BLOCK-END
