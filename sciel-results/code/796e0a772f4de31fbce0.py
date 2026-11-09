"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# We keep a small global cache to avoid recomputing expensive heuristics
# across calls if the same state is revisited (unlikely but safe).
# More importantly, we implement a robust, fast heuristic that balances
# remaining work, machine load, and urgency.

def dispatch(ready_operations, time):
    """
    Select the next operation to start and optionally the machine.
    
    Strategy:
    1. If only one op is ready, return it with the best machine.
    2. Otherwise, score each candidate operation using a composite heuristic:
       - Prioritize jobs with more remaining work (to avoid tail effects).
       - Prefer machines that are less loaded / have shorter durations.
       - Consider the ratio of remaining work to number of remaining ops (urgency).
    
    We return (index, machine) to allow the simulator to pick the machine.
    If the simulator expects just an index, it will ignore the second element.
    """
    if not ready_operations:
        return 0  # Should not happen, but safe fallback

    if len(ready_operations) == 1:
        op = ready_operations[0]
        # Pick the machine with the shortest duration
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)

    # Pre-compute some global stats if possible? 
    # We don't have global machine load info directly, but we can infer 
    # from the fact that ready operations are those that can start now.
    # A common effective heuristic in flexible job shops is:
    #   Score = alpha * (remaining_work) + beta * (min_duration) + gamma * (urgency)
    # But min_duration is machine-dependent. So for each op, we consider the best machine.
    
    # Let's define a score for each operation based on:
    # 1. The minimum processing time among its eligible machines (shorter is better for this op)
    # 2. The remaining work of the job (more work remaining means higher priority to finish soon)
    # 3. The number of remaining operations (fewer remaining means higher urgency)
    
    # We'll use a weighted combination. Weights are tuned heuristically.
    # To avoid division by zero, we add a small epsilon.
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            # Should not happen, but fallback
            scores.append(0)
            best_machines.append(None)
            continue
        
        # Find the machine with the shortest duration for this operation
        best_machine = min(options.keys(), key=lambda m: options[m])
        min_duration = options[best_machine]
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Urgency: jobs with fewer remaining ops should be prioritized to avoid blocking
        # but jobs with more total remaining work also need attention.
        # A common approach is to use the ratio of remaining work to remaining ops as an "average work per op"
        # and prioritize those with higher average work (to keep busy machines busy).
        
        # Let's use:
        # score = w1 * remaining_work + w2 * (remaining_work / remaining_ops) - w3 * min_duration
        # Higher score is better.
        
        # We want to start ops that:
        # - Have high remaining work (to reduce tail)
        # - Have high average work per op (to keep machines busy)
        # - Have short processing time on the best available machine (to free up machines quickly)
        
        # Let's try a simple multiplicative or additive combination.
        # Additive is safer for scale differences.
        
        # Normalize roughly? We don't know the scale. Let's just use raw values with weights.
        # Weights:
        w1 = 1.0  # weight for total remaining work
        w2 = 0.5  # weight for average work per op
        w3 = 1.0  # weight for min duration (negative, so we want smaller)
        
        avg_work = remaining_work / max(remaining_ops, 1)
        
        score = w1 * remaining_work + w2 * avg_work - w3 * min_duration
        
        scores.append(score)
        best_machines.append(best_machine)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    # If best_machine is None (shouldn't happen), fallback to 0
    if best_machine is None:
        return best_idx
    
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
