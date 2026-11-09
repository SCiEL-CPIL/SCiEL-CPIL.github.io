"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation to start.
    
    Load-balancing heuristic inspired by bipartite assignment:
    For each ready operation, estimate the resulting machine load if scheduled
    on its best (shortest duration) machine. We prefer operations that:
    1. Go to the currently least-loaded machine (load balancing)
    2. Have high remaining work (critical path reduction)
    3. Have many remaining operations (job complexity)
    
    The key insight: in open-shop scheduling, makespan is driven by the most
    loaded machine. By routing operations to less-loaded machines and prioritizing
    high-work operations, we balance the load distribution.
    """
    if not ready_operations:
        return -1

    best_idx = 0
    best_score = -float('inf')

    # Deadline guard to prevent timeout
    start_t = _time.time()
    deadline = start_t + 10.0

    # Pre-compute machine loads from all ready operations' options
    # This gives us a rough estimate of current machine congestion
    machine_loads = {}
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if isinstance(opts, dict):
            for m, d in opts.items():
                d = float(d)
                if m not in machine_loads:
                    machine_loads[m] = 0.0
                # Use duration as a proxy for load contribution
                if d > 0:
                    machine_loads[m] += d

    for i, op in enumerate(ready_operations):
        if _time.time() > deadline:
            break

        # Extract machine options
        opts = op.get("machine_options", {})
        mor = float(op.get("remaining_ops", 0))
        rw = float(op.get("remaining_work", 0))

        if isinstance(opts, dict):
            if not opts:
                min_dur = 0.0
                best_machine = None
                min_machine_load = 0.0
            else:
                # Find the machine with minimum (current_load + duration)
                # This is a greedy approximation of the Hungarian assignment
                best_machine = None
                min_combined = float('inf')
                min_dur = float('inf')
                min_machine_load = 0.0
                for m, d in opts.items():
                    d = float(d)
                    if d <= 0:
                        continue
                    if d < min_dur:
                        min_dur = d
                    current_load = machine_loads.get(m, 0.0)
                    combined = current_load + d
                    if combined < min_combined:
                        min_combined = combined
                        best_machine = m
                        min_machine_load = current_load

                if best_machine is None:
                    # All durations were zero or invalid
                    min_dur = 0.0
                    min_machine_load = 0.0
                    best_machine = None
        else:
            min_dur = float(opts) if opts else 0.0
            min_machine_load = 0.0
            best_machine = None

        # Scoring:
        # Primary: minimize the resulting machine load (load balancing)
        # We want to pick operations that go to less-loaded machines
        # Secondary: maximize remaining work (critical path)
        # Tertiary: maximize remaining operations (job depth)
        #
        # The load balance term is most important: by consistently routing
        # to the least-loaded machine, we flatten the load distribution.
        
        # Normalize machine load to [0, 1] range for comparison
        max_load = max(machine_loads.values()) if machine_loads else 1.0
        if max_load > 0:
            normalized_min_load = min_machine_load / max_load
        else:
            normalized_min_load = 0.0

        # Lower normalized_min_load is better (less congested machine)
        # Higher rw and mor are better
        
        # Weight tuning: load balance is the dominant factor
        score = (-normalized_min_load * 1000.0) + (rw * 1.0) + (mor * 50.0)

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
