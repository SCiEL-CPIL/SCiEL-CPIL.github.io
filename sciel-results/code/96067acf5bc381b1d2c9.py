"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Calculates priority scores for assembly tasks using a refined heuristic.
    
    The primary signal is the 'Rank' (Longest Path to end), which identifies tasks
    on the critical path. Secondary signals include total downstream work and
    connectivity to break ties and identify bottlenecks that rank alone might miss.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    times = np.array(instance["times"], dtype=float)
    n_tasks = len(times)
    preds = instance["predecessors"]
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    for i in range(n_tasks):
        if i < len(preds):
            for p in preds[i]:
                # p is a predecessor of i, so i is a successor of p
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    indeg = in_degree[:]
    for i in range(n_tasks):
        if indeg[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
                
    # If graph is cyclic or malformed, topo_order might be incomplete. Assume valid DAG.
    # If not all tasks are in topo_order, we need a fallback or handle it.
    # For now, assume valid DAG as per task description.
    
    # Calculate Rank (Longest Path) and Total Downstream Time in reverse topological order
    rank = np.zeros(n_tasks, dtype=float)
    total_down = np.zeros(n_tasks, dtype=float)
    
    # Iterate backwards through topological order
    for i in reversed(topo_order):
        # For rank: longest path through successors
        max_succ_rank = 0.0
        # For total_down: sum of all downstream work
        sum_succ_down = 0.0
        
        if successors[i]:
            succ_ranks = [rank[j] for j in successors[i]]
            max_succ_rank = max(succ_ranks)
            succ_downs = [total_down[j] for j in successors[i]]
            sum_succ_down = sum(succ_downs)
        
        rank[i] = times[i] + max_succ_rank
        total_down[i] = times[i] + sum_succ_down
    
    # Secondary features
    num_succ = np.array([len(successors[i]) for i in range(n_tasks)], dtype=float)
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback if we ran out of time (unlikely for n=30)
        return times.copy()
    
    # Compute positional urgency: how close each task is to filling the cycle time
    # Tasks that must start early (low slack) should be prioritized
    # Slack = cycle_time - rank[i] (earliest start time if all predecessors done)
    # Higher urgency = less slack = more critical to place early
    cycle_time = instance.get("cycle_time", 40.0)
    slack = cycle_time - rank
    # Invert slack so higher values mean more urgent
    urgency = -slack
    
    # Use soft-max to create a smoother aggregation of successor pressures
    # This helps break ties between tasks with similar ranks but different successor structures
    soft_max_weight = 2.0  # temperature parameter for soft-max
    soft_max_scores = np.zeros(n_tasks, dtype=float)
    for i in range(n_tasks):
        if successors[i]:
            succ_ranks = np.array([rank[j] for j in successors[i]])
            # Soft-max: exp(w * x) / sum(exp(w * x))
            exp_vals = np.exp(soft_max_weight * succ_ranks)
            soft_max_scores[i] = np.sum(exp_vals)
        else:
            soft_max_scores[i] = 0.0
    
    # Normalize soft_max_scores to be comparable to rank
    if np.max(soft_max_scores) > 0:
        soft_max_scores = soft_max_scores / np.max(soft_max_scores)
    
    # Composite Score:
    # Primary: Rank (critical path) - most important for minimizing stations
    # Secondary: Urgency (positional pressure) - tasks with less slack should go first
    # Tertiary: Total downstream time - identifies true bottlenecks  
    # Quaternary: Soft-max successor pressure - breaks ties with smooth gradient
    # Quinary: Number of successors - final tie-breaker favoring high-connectivity tasks
    
    # Weights tuned to emphasize critical path while using positional urgency
    # and downstream work to distinguish between tasks with similar ranks.
    # Rank is typically the dominant factor, but urgency helps when multiple tasks
    # have similar ranks but different positional constraints.
    
    scores = 1.0 * rank + 0.25 * urgency + 0.10 * total_down + 0.08 * soft_max_scores + 0.03 * num_succ
    
    return scores

# EVOLVE-BLOCK-END
