"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s kill


def _elapsed():
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're running long, return a quick heuristic
    if _elapsed() > _DEADLINE:
        best_idx = 0
        op = ready_operations[best_idx]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (best_idx, best_machine)

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)

    scores = []
    best_machines = []

    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]

        # Constraint tightness: fewer machine options = more constrained
        num_options = len(machine_options)
        constraint_score = 1.0 / max(num_options, 1)

        # Pick the machine with the shortest processing time
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        best_duration = machine_options[best_machine]

        # Soft-min surrogate: harmonic mean of durations (smooths the min,
        # sensitive to the distribution of machine options)
        durations = list(machine_options.values())
        if len(durations) == 1:
            soft_min = durations[0]
        else:
            # Harmonic mean: (n / sum(1/d))
            inv_sum = sum(1.0 / d for d in durations if d > 0)
            soft_min = len(durations) / max(inv_sum, 1e-9)

        # Urgency ratio: how much work this op clears per unit of (soft-min) time.
        # Higher = more critical bottleneck-clearing operation.
        urgency_ratio = remaining_work / max(soft_min, 1e-9)

        # Composite score:
        # Primary: urgency ratio (work cleared per unit time)
        # Secondary: constraint tightness (fewer options = higher priority)
        # Tertiary: more remaining work = more critical
        # Quaternary: fewer remaining ops = closer to completion
        score = (
            urgency_ratio * 3.0
            + constraint_score * 5.0
            + remaining_work * 0.5
            - remaining_ops * 0.1
        )
        scores.append(score)
        best_machines.append(best_machine)

    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
