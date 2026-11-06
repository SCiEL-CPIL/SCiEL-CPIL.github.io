import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    machine_work = defaultdict(float)
    for op in ready_operations:
        for machine, duration in op["machine_options"].items():
            machine_work[machine] += duration

    best_index = 0
    best_score = float('inf')
    best_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        min_work_machine = None
        min_work_val = float('inf')
        best_machine_for_op = None
        best_dur_for_op = 0
        
        for machine, duration in op["machine_options"].items():
            w = machine_work.get(machine, 0)
            # We want the machine that results in the lowest new load.
            # new_load = w - duration.
            # To minimize new_load, we want to maximize (duration) for a given w.
            # So if w is lower, it's better.
            # If w is equal, higher duration is better (reduces load more).
            
            # Score for this machine choice: (w, -duration) -> minimize w, maximize duration
            # We can just track the best candidate
            if min_work_machine is None:
                min_work_machine = machine
                min_work_val = w
                best_machine_for_op = machine
                best_dur_for_op = duration
            else:
                if w < min_work_val:
                    min_work_val = w
                    min_work_machine = machine
                    best_machine_for_op = machine
                    best_dur_for_op = duration
                elif w == min_work_val:
                    if duration > best_dur_for_op:
                        best_machine_for_op = machine
                        best_dur_for_op = duration
                        # min_work_machine stays the same value, but we update the specific machine reference?
                        # Actually min_work_machine is just for the value. We need the machine key.
                        # Let's just store the machine key directly.
                        min_work_machine = machine # This is redundant if we use best_machine_for_op

        if best_machine_for_op is None:
            continue

        duration_on_min_machine = best_dur_for_op
        new_load_on_min_machine = min_work_val - duration_on_min_machine
        
        max_other_load = 0
        for machine, work in machine_work.items():
            if machine != best_machine_for_op:
                if work > max_other_load:
                    max_other_load = work
        
        resulting_max_load = max(new_load_on_min_machine, max_other_load)
        
        if resulting_max_load < best_score:
            best_score = resulting_max_load
            best_duration = duration_on_min_machine
            best_index = i
        elif resulting_max_load == best_score:
            if duration_on_min_machine < best_duration:
                best_duration = duration_on_min_machine
                best_index = i

    return int(best_index)
