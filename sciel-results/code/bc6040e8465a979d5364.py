# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of pred indices
    
    n = len(durations)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute transitive successor count (how many activities depend on this one)
    # Use DFS/BFS from each node - n=30 so this is fine
    def count_successors(start):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for s in successors[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        return len(visited)
    
    succ_counts = np.array([count_successors(i) for i in range(n)], dtype=float)
    
    # Resource pressure: sum over resources of demand/capacity
    # For each activity i, pressure_i = sum_r (demand[i,r] / capacity[r])
    with np.errstate(divide='ignore', invalid='ignore'):
        resource_pressure = np.sum(demands / capacities[np.newaxis, :], axis=1)
    resource_pressure = np.nan_to_num(resource_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize each component to [0,1] range for combination
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-9:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    norm_succ = normalize(succ_counts)
    norm_pressure = normalize(resource_pressure)
    norm_duration = normalize(durations)
    
    # Combine: weight successor count heavily, then resource pressure, then duration
    # Activities with many successors and high resource pressure should go first
    score = 0.5 * norm_succ + 0.3 * norm_pressure + 0.2 * norm_duration
    
    return score
# EVOLVE-BLOCK-END
