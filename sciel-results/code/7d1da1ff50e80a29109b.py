"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Critical path-based dispatch rule with machine selection.
    
    Strategy:
    1. For each ready operation, compute the critical path length:
       - The minimum duration of the current operation (best case for this op)
       - Plus the sum of minimum durations of all remaining operations in the job
    2. Prioritize operations with the highest critical path length.
    3. Among tied or close candidates, use a tie-breaker based on remaining work.
    4. Return (index, machine) where machine is the one that minimizes completion time
       (i.e., the machine with the shortest duration for that operation).
    """
    if not ready_operations:
        return 0
    
    # Compute critical path score for each ready operation
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        # Minimum duration for the current operation
        min_dur_current = min(machine_options.values())
        
        # Sum of minimum durations for all remaining operations
        # remaining_ops is the list of operations yet to be processed (including current? 
        # Typically in these simulations, remaining_ops includes the current op or just future ones)
        # Based on common conventions, remaining_ops often includes the current operation.
        # Let's assume remaining_ops contains all operations that still need to be done, including the current one.
        # If remaining_work is provided, it might be a pre-computed sum. Let's use remaining_work if available,
        # but the task says "remaining_work" is a field. Let's check if it's the sum of mins.
        # To be safe, let's compute the sum of minimums from remaining_ops if possible, 
        # but the interface says op has "remaining_ops" and "remaining_work".
        # Let's use remaining_work as a proxy for total remaining effort, but refine with critical path logic.
        
        # Critical path = min duration of current op + sum of min durations of subsequent ops
        # If remaining_work is the sum of min durations of ALL remaining ops (including current), 
        # then critical path for prioritization is essentially remaining_work.
        # However, the current operation's duration on the chosen machine matters for the actual start.
        
        # Let's define the score as:
        # score = (sum of min durations of all remaining ops in the job)
        # This is equivalent to op["remaining_work"] if that field represents the sum of minimums.
        
        # Let's use remaining_work as the primary metric for critical path length.
        # Tie-breaker: minimize the current operation's duration (to finish this step faster).
        
        score = op["remaining_work"]
        
        # Find the machine that gives the minimum duration for this operation
        best_machine = min(machine_options, key=machine_options.get)
        min_dur_for_best_machine = machine_options[best_machine]
        
        scores.append(score)
        best_machines.append((best_machine, min_dur_for_best_machine))
    
    # Sort by score descending (highest critical path first)
    # We want to pick the index with the highest score.
    # In case of ties, we could break by smaller current duration.
    # For now, just pick the max score. If there are ties, np.argmax picks the first one.
    
    # Let's create a composite score to handle ties better:
    # Primary: remaining_work (higher is better)
    # Secondary: -min_duration_of_current_op (lower is better, so we negate)
    
    # Actually, let's just use remaining_work as the primary key.
    # The problem with just remaining_work is that it doesn't account for the current op's specific machine choice.
    # But the critical path concept is about the longest chain of remaining work.
    
    # Let's refine: 
    # The "urgency" of an operation is determined by how much work follows it.
    # So, op["remaining_work"] is a good proxy.
    
    # However, we also need to return the machine.
    # The task says: "optionally (index, machine)".
    # Let's return (index, machine) to allow the simulator to pick the best machine.
    
    # Find the index with the highest remaining_work
    max_score = max(scores)
    
    # Find all indices with the max score
    max_indices = [i for i, s in enumerate(scores) if s == max_score]
    
    if len(max_indices) == 1:
        idx = max_indices[0]
    else:
        # Tie-breaker: pick the one with the smallest duration on its best machine
        # This helps reduce the completion time of the current step
        best_idx = min(max_indices, key=lambda i: best_machines[i][1])
        idx = best_idx
    
    machine = best_machines[idx][0]
    
    return (idx, machine)
# EVOLVE-BLOCK-END
