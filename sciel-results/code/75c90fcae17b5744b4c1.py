"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard: 150s wall-clock limit, safety margin 5s
_DEADLINE = _time.time() + 145.0


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: A composite heuristic that considers:
    1. Estimated critical path (sum of min processing times for all remaining ops in job)
    2. Shortest processing time on the best machine option
    3. Machine contention: prefer operations whose preferred machine has fewer competing ready ops
    4. Proximity to completion: jobs with fewer remaining ops get slight priority
    
    This balances urgency (critical path), efficiency (SPT), and resource contention.
    """
    # Deadline check
    if _time.time() > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Count how many ready operations prefer each machine (for contention)
    machine_contention = {}
    for op in ready_operations:
        mo = op["machine_options"]
        if not mo:
            continue
        # Find the machine with minimum duration (preferred machine)
        best_m = min(mo.keys(), key=lambda m: mo[m])
        machine_contention[best_m] = machine_contention.get(best_m, 0) + 1

    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores.append(0.0)
            continue

        # Get min duration and identify preferred machine
        min_duration = min(machine_options.values())
        preferred_machine = min(machine_options.keys(), key=lambda m: machine_options[m])

        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Contention penalty: higher contention on preferred machine is worse
        contention = machine_contention.get(preferred_machine, 1)
        # Normalize: contention ranges from 1 to n, so penalty scales with contention
        contention_penalty = 0.3 * (contention - 1)

        # Critical path weight: prioritize jobs with more remaining work
        # SPT weight: prefer shorter current operations
        # Contention weight: avoid machines with many competing operations
        
        # Composite score: higher is better
        score = (
            2.0 * remaining_work -            # Critical path: prioritize more remaining work
            1.2 * min_duration +              # SPT: prefer shorter processing
            0.4 * (remaining_ops - 1) -       # Prefer fewer remaining ops (closer to done)
            contention_penalty                 # Penalty for high machine contention
        )
        scores.append(score)

    # Pick the operation with the highest composite score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
