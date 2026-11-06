"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine runs it.
    
    Strategy: Bottleneck mitigation via weighted constraint tightness and urgency.
    1. Calculate a priority score for each operation based on:
       - Constraint Tightness: Inverse of number of machine options.
       - Urgency: Average remaining work per remaining operation.
    2. Select the operation with the highest priority score.
    3. For the selected operation, choose the machine with the shortest processing time
       to minimize the operation's completion time, thereby reducing job flow time.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if len(machine_options) == 1:
            return (0, list(machine_options.keys())[0])
        # Pick machine with shortest duration
        best_m = min(machine_options, key=lambda m: machine_options[m])
        return (0, best_m)

    # Calculate priority scores for each operation
    # We want to maximize this score.
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        num_machines = len(machine_options)
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", 0)

        # 1. Constraint Tightness: Fewer options -> Higher tightness
        # Scale it significantly so it dominates when options are very limited
        tightness = 1.0 / max(num_machines, 1)
        
        # 2. Urgency: How much work is left per operation on average
        # This ensures we don't starve jobs with huge remaining work just because they have more machine options
        urgency = remaining_work / max(remaining_ops, 1)
        
        # Combined Score:
        # Weight tightness heavily (e.g., 10x) to enforce the "bottleneck mitigation" directive,
        # but allow urgency to break ties or slightly override if the difference in tightness is small.
        # If num_machines is 1, tightness is 1.0. If 5, it's 0.2.
        # If urgency is large (e.g., 100), it should matter.
        # Let's use: Score = tightness * 50.0 + urgency
        score = tightness * 50.0 + urgency
        scores.append(score)

    # Find the operation with the highest score
    best_idx = np.argmax(scores)
    
    # Select the best machine for the chosen operation
    op = ready_operations[best_idx]
    machine_options = op["machine_options"]

    if len(machine_options) == 1:
        return (best_idx, list(machine_options.keys())[0])

    # Machine Selection Heuristic:
    # Choose the machine with the minimum processing time.
    # This minimizes the time the job is occupied, allowing subsequent operations to start earlier.
    # This is a standard "fastest machine" heuristic for makespan reduction in flexible job shops
    # when machine load information is not available in the dispatch function.
    best_machine = min(machine_options, key=lambda m: machine_options[m])

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
