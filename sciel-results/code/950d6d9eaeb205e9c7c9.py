import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    # predecessors[i] is a list of indices of tasks that must be completed before i
    successors = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Calculate Rank (Longest path to end)
    # Rank[i] = times[i] + max(Rank[j] for j in successors[i])
    # We can compute this using topological sort in reverse (from sinks to sources)
    # Or simply memoized recursion / DP.
    
    # Since n is small (30), we can use a simple iterative approach or recursion with memoization.
    # Let's use a topological sort on the reversed graph or just standard topological sort and process in reverse.
    
    # Build in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    # Calculate ranks in reverse topological order
    ranks = np.zeros(n)
    # Process from last to first in topo_order (sinks first)
    for i in range(len(topo_order) - 1, -1, -1):
        node = topo_order[i]
        max_succ_rank = 0
        for succ in successors[node]:
            if ranks[succ] > max_succ_rank:
                max_succ_rank = ranks[succ]
        ranks[node] = times[node] + max_succ_rank
        
    return ranks
