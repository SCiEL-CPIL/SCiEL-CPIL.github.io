"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Track machine loads across the simulation to balance work.
_machine_load = {}


def dispatch(ready_operations, time):
    global _machine_load

    if not ready_operations:
        return 0

    # For each ready operation, evaluate the best machine option and a composite score.
    scores = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]

        # Choose the machine with the lowest current load among options.
        best_machine_load = None
        best_duration = None
        for m, dur in machine_options.items():
            load = _machine_load.get(m, 0.0)
            # We want low load; break ties by shorter duration.
            if best_machine_load is None or load < best_machine_load or (
                load == best_machine_load and (best_duration is None or dur < best_duration)
            ):
                best_machine_load = load
                best_duration = dur

        if best_machine_load is None:
            best_machine_load = 0.0
        if best_duration is None:
            best_duration = 0.0

        # Composite score: higher is better.
        # - Prefer operations on less-loaded machines (negative load).
        # - Prefer jobs with fewer remaining ops (urgency, helps finish jobs).
        # - Slight preference for shorter processing time on the chosen machine.
        score = (
            -best_machine_load
            - 2.0 * remaining_ops          # urgency weight
            - 0.5 * best_duration          # short processing preference
            + 0.1 * remaining_work         # mild work consideration
        )
        scores.append(score)

    scores = np.array(scores)
    idx = int(np.argmax(scores))

    # Update machine load bookkeeping for the chosen operation's best machine.
    chosen_op = ready_operations[idx]
    machine_options = chosen_op["machine_options"]
    # Recompute best machine (same logic as above) to update load.
    best_machine = None
    best_machine_load = None
    best_duration = None
    for m, dur in machine_options.items():
        load = _machine_load.get(m, 0.0)
        if best_machine_load is None or load < best_machine_load or (
            load == best_machine_load and (best_duration is None or dur < best_duration)
        ):
            best_machine = m
            best_machine_load = load
            best_duration = dur

    if best_machine is not None and best_duration is not None:
        _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + best_duration

    return idx
# EVOLVE-BLOCK-END
