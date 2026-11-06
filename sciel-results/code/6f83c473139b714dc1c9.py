"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    scores = []
    best_machines = []

    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]

        # Pick the machine with the shortest processing time for this operation
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        best_duration = machine_options[best_machine]

        # Composite urgency score:
        # - More remaining work in the job => more critical (higher priority)
        # - Shorter processing time => finishes sooner, frees machine
        # - Fewer remaining ops => closer to completion, reduces tail latency
        score = remaining_work * 1.0 - best_duration * 0.5 - remaining_ops * 0.1
        scores.append(score)
        best_machines.append(best_machine)

    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
