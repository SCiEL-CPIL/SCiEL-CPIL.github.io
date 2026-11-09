"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation to start.
    
    Bipartite-graph-inspired heuristic:
    For each ready operation, compute the maximum "work density" across all
    machine options: remaining_work / duration_on_machine. This measures how
    much remaining work the operation can clear per unit of machine time on
    its best machine. Combine with remaining_ops and remaining_work.
    
    The idea: operations that clear a lot of remaining work quickly (high
    density) reduce the critical path more effectively per unit of time.
    """
    if not ready_operations:
        return -1

    best_idx = 0
    best_score = -1.0

    # Deadline guard to prevent timeout
    start_t = _time.time()
    deadline = start_t + 10.0

    for i, op in enumerate(ready_operations):
        if _time.time() > deadline:
            break

        # Extract machine options
        opts = op.get("machine_options", {})
        mor = float(op.get("remaining_ops", 0))
        rw = float(op.get("remaining_work", 0))

        # Compute work density: max over machines of (rw / duration)
        # This is the "critical path reduction per unit machine time"
        if isinstance(opts, dict):
            if not opts:
                density = 0.0
                min_dur = 0.0
            else:
                # Compute rw/dur for each machine, take max
                # Guard against zero duration
                best_density = 0.0
                min_dur = float('inf')
                for m, d in opts.items():
                    d = float(d)
                    if d > 0:
                        if d < min_dur:
                            min_dur = d
                        dens = rw / d
                        if dens > best_density:
                            best_density = dens
                density = best_density
                if min_dur == float('inf'):
                    min_dur = 0.0
        else:
            density = 0.0
            min_dur = float(opts) if opts else 0.0

        # Scoring:
        # Primary: work density (critical path reduction per unit time)
        # Secondary: remaining work (total load)
        # Tertiary: remaining ops (job depth)
        #
        # Weight tuning: density captures efficiency, rw captures scale,
        # mor captures job complexity.
        score = (density * 100.0) + (rw * 1.0) + (mor * 50.0)

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
