"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_idx = -1
    best_score = -1.0

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op.get("remaining_work", 0)
        
        # Find the minimum duration among available machines for this operation
        # This represents the "fastest" way to process this operation.
        min_dur = float('inf')
        for m, dur in machine_options.items():
            if dur < min_dur:
                min_dur = dur
        
        if min_dur == float('inf'):
            min_dur = 0.0

        # Heuristic: Most Work Remaining (MWR) for the job, tie-break by Longest Processing Time (LPT) for the op.
        # We want to prioritize jobs with more work left to avoid leaving them stranded.
        # Among jobs with similar work, process the longer operation first to get it out of the way.
        
        # Scale factors: remaining_work is usually larger than min_dur?
        # Let's check typical scales. Durations are likely 1-100. Remaining work is sum of durations, so larger.
        # We want MWR to be dominant.
        score = remaining_work * 10.0 + min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
