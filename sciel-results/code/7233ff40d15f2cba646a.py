"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def priority(instance):
    """
    Calculates priority scores for assembly tasks using a heuristic based on 
    Longest Path (Rank) and Task Duration.
    
    The Longest Path heuristic estimates the minimum remaining time for each task 
    to complete the line, prioritizing tasks on the critical path. This is 
    generally superior to simple LPT (Longest Processing Time) for SALB as it 
    accounts for precedence constraints' impact on station count.
    """
    times = np.array(instance["times"], dtype=float)
    n_tasks = len(times)
    preds = instance["predecessors"]
    # preds is typically a list of lists, where preds[i] contains indices of predecessors for task i.
    # We need successors to compute Rank (distance to end).
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    for i in range(n_tasks):
        if i < len(preds):
            for p in preds[i]:
                # p is a predecessor of i, so i is a successor of p
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort (Kahn's Algorithm) to process nodes
    # We need the order to compute Rank. 
    # Rank(i) = times[i] + max(Rank(j) for j in successors(i))
    # We can compute this in reverse topological order.
    
    # Standard Topological Sort gives order from Start to End.
    # We want End to Start to compute Rank.
    # Alternatively, we can use DFS or just iterate until no changes (Bellman-Ford style on DAG).
    # Given n=30, O(N^2) or O(N+E) is fine.
    
    # Let's use a simple iterative relaxation (Dijkstra-like on DAG or just topological sort)
    # First, get topological order
    queue = []
    indeg = in_degree[:]
    for i in range(n_tasks):
        if indeg[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        # For deterministic behavior, we could sort, but it doesn't matter for correctness of topo order
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
                
    # If graph is cyclic or malformed, topo_order might be incomplete. Assume valid DAG.
    # Calculate Rank (Longest Path) in reverse topological order
    rank = np.zeros(n_tasks, dtype=float)
    # Calculate Total Downstream Time (sum of all downstream work)
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
    
    # Composite Score:
    # Primary: Rank (critical path) - most important for minimizing stations
    # Secondary: Total downstream time - identifies true bottlenecks
    # Tertiary: Number of successors - breaks ties favoring high-connectivity tasks
    # 
    # The total_down metric captures tasks that have significant work depending on them,
    # which is a better bottleneck indicator than just the longest path.
    # 
    # Weights: Rank dominates, total_down provides additional discrimination,
    # num_succ breaks remaining ties.
    
    scores = 1.0 * rank + 0.15 * total_down + 0.05 * num_succ
    
    return scores

# EVOLVE-BLOCK-END
