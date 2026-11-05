import numpy as np
import time as _time

_dispatch_start_time = None

def dispatch(ready_operations, time):
    global _dispatch_start_time
    
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > 140.0:
        return 0
    
    # Machine-load balancing strategy:
    # For each ready operation, estimate its impact on the most constrained machine.
    # We use the minimum duration (best machine) as the processing time if we start it now.
    # But we also consider the "constraint" of the machine.
    
    # A simple proxy for machine load/constraint:
    # If an operation has very few options, it's more constrained.
    # If an operation has a high min duration, it takes longer even on the best machine.
    
    # Let's try: score = min_duration / num_options?
    # Or score = min_duration * (1 / num_options)?
    # Higher score = more critical?
    
    # Alternative: prioritize operations that have the shortest processing time on their
    # most constrained machine. "Most constrained" could mean the machine with the fewest
    # available operations that can use it? But we don't have that info directly.
    
    # Let's use a simpler heuristic:
    # For each op, calculate min_duration.
    # Among ops with the same min_duration, prefer the one with fewer options (more constrained).
    # Or: score = min_duration - epsilon * num_options?
    
    # Let's try: sort by (min_duration, -num_options) and pick the first?
    # No, we want to minimize makespan.
    
    # Let's go back to the inspiration: "minimum remaining workload on its most constrained machine"
    # Since we don't have machine load, let's approximate "constrained" by the number of options.
    # An operation with fewer options is more constrained.
    # An operation with a longer min_duration is more impactful.
    
    # Let's try: score = min_duration / max(1, num_options)
    # Higher score = longer processing on a constrained machine.
    
    scores = []
    for op in ready_operations:
        options = op["machine_options"]
        if not options:
            scores.append(0)
        else:
            min_dur = min(options.values())
            num_opts = len(options)
            # Higher score for longer min duration and fewer options
            score = min_dur / num_opts
            scores.append(score)
            
    idx = int(np.argmax(scores))
    return idx
