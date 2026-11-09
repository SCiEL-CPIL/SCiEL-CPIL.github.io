import numpy as np
import time

def priority(instance):
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            in_degree[i] += 1
    
    # Compute RPV (Rank Position Value) - sum of processing times of all successors
    out_degree = [len(succ[i]) for i in range(n)]
    od = out_degree[:]
    q = [i for i in range(n) if od[i] == 0]
    rpv = times.copy()
    
    while q:
        u = q.pop()
        for p in preds[u]:
            rpv[p] += rpv[u]
            od[p] -= 1
            if od[p] == 0:
                q.append(p)
    
    # Compute bottom level (longest path to any sink)
    bottom_level = times.copy()
    od2 = out_degree[:]
    q2 = [i for i in range(n) if od2[i] == 0]
    
    while q2:
        u = q2.pop()
        for p in preds[u]:
            new_bl = times[p] + bottom_level[u]
            if new_bl > bottom_level[p]:
                bottom_level[p] = new_bl
            od2[p] -= 1
            if od2[p] == 0:
                q2.append(p)
    
    # Count number of successors (direct and indirect)
    # Use BFS/DFS from each node to count all reachable successors
    # For efficiency, use DP: count[u] = 1 + sum(count[s] for s in succ[u]) but that double counts
    # Better: just use direct successor count as a proxy, or compute via reverse topo
    
    # Actually, let's compute the number of immediate successors as a simpler heuristic
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Normalize all components to [0, 1] range
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    rpv_norm = normalize(rpv)
    bl_norm = normalize(bottom_level)
    ns_norm = normalize(num_succ)
    
    # Combine with weighted sum
    # Higher priority = better
    # Weight RPV heavily, then bottom level, then number of successors
    combined = 0.5 * rpv_norm + 0.3 * bl_norm + 0.2 * ns_norm
    
    return combined
