"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            if 0 <= j < n:
                successors[j].append(i)
                
    # Calculate Total Succeeding Time (Rank) for each task
    # Rank[i] = times[i] + sum(Rank[j] for j in immediate successors) is NOT correct for total sum.
    # We need the sum of times of ALL successors (transitive).
    
    # Since N is small (30), we can do a DFS/BFS from each node to find all transitive successors.
    # Or, we can compute the transitive closure matrix.
    
    # Method: For each node i, perform a DFS to collect all reachable nodes.
    ranks = np.zeros(n)
    
    for i in range(n):
        # DFS to find all successors of i
        visited = [False] * n
        stack = [i]
        visited[i] = True
        total_time = 0.0
        while stack:
            node = stack.pop()
            total_time += times[node]
            for succ in successors[node]:
                if not visited[succ]:
                    visited[succ] = True
                    stack.append(succ)
        ranks[i] = total_time
        
    # Tie-breaker: Prefer tasks with more immediate successors?
    # Or simply return ranks.
    # Let's add a small bonus for having more immediate successors to encourage unblocking.
    # This is a heuristic tweak.
    succ_counts = np.array([len(s) for s in successors])
    
    # Normalize succ_counts to a small range to act as tie-breaker
    # Max succ count is n-1. We want this to be much smaller than the difference in ranks usually.
    # But if ranks are equal, higher succ count is better.
    # Let's just use ranks. It's the standard strong heuristic.
    
    return ranks
# EVOLVE-BLOCK-END
