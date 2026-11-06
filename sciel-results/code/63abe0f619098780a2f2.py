import numpy as np
from functools import lru_cache

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build adjacency list for successors
    # predecessors[i] = list of tasks that must come before task i
    # So if j is in predecessors[i], then j -> i (j is predecessor of i, i is successor of j)
    
    # Compute transitive successors for each task
    # transitive_successors[i] = set of all tasks that depend on i (directly or indirectly)
    
    # First, build the graph: edge from predecessor to successor
    adj = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            adj[pred].append(i)
    
    # Compute transitive successors using BFS/DFS from each node
    transitive_succ_count = [0] * n
    for i in range(n):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for succ in adj[node]:
                if succ not in visited:
                    visited.add(succ)
                    stack.append(succ)
        transitive_succ_count[i] = len(visited)
    
    # Priority: time + small bonus for transitive successors
    max_succ = max(transitive_succ_count) if transitive_succ_count else 1
    if max_succ > 0:
        bonus = 0.1 * np.array(transitive_succ_count) / max_succ
    else:
        bonus = np.zeros(n)
    
    return times + bonus
