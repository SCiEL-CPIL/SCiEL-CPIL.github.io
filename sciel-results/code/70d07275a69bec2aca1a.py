"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse instance structure
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    # Build adjacency list for predecessors
    pred_list = [list() for _ in range(n)]
    if predecessors is not None:
        if isinstance(predecessors, dict):
            for act, preds in predecessors.items():
                act_idx = int(act)
                for p in preds:
                    pred_list[act_idx].append(int(p))
        elif isinstance(predecessors, (list, tuple, np.ndarray)):
            for i, preds in enumerate(predecessors):
                if preds is not None:
                    if isinstance(preds, (list, tuple, np.ndarray)):
                        pred_list[i] = [int(p) for p in preds]
                    elif isinstance(preds, (int, np.integer)):
                        pred_list[i] = [int(preds)]
    
    # Build demands structure
    n_resources = 0
    demand_matrix = None
    cap_vector = None
    
    if demands is not None:
        if isinstance(demands, (list, tuple, np.ndarray)):
            d_arr = np.asarray(demands)
            if d_arr.ndim == 2:
                demand_matrix = d_arr.astype(float)
                n_resources = d_arr.shape[1]
            elif d_arr.ndim == 1:
                demand_matrix = d_arr.reshape(-1, 1).astype(float)
                n_resources = 1
        elif isinstance(demands, dict):
            res_keys = list(demands.keys())
            n_resources = len(res_keys)
            demand_matrix = np.zeros((n, n_resources))
            for j, rk in enumerate(res_keys):
                val = demands[rk]
                if isinstance(val, (list, tuple, np.ndarray)):
                    demand_matrix[:, j] = np.asarray(val, dtype=float)
                else:
                    demand_matrix[:, j] = float(val)
    
    if capacities is not None:
        if isinstance(capacities, (list, tuple, np.ndarray)):
            cap_vector = np.asarray(capacities, dtype=float)
            if cap_vector.ndim == 0:
                cap_vector = np.full(n_resources, float(cap_vector)) if n_resources > 0 else np.array([1.0])
        elif isinstance(capacities, dict):
            cap_vector = np.zeros(n_resources)
            if n_resources > 0:
                if len(capacities) == 1:
                    cap_vector = np.full(n_resources, float(list(capacities.values())[0]))
                else:
                    for j in range(n_resources):
                        if j in capacities:
                            cap_vector[j] = float(capacities[j])
                        elif str(j) in capacities:
                            cap_vector[j] = float(capacities[str(j)])
                        else:
                            cap_vector[j] = float(list(capacities.values())[j]) if j < len(capacities) else 1.0
    
    if demand_matrix is None:
        demand_matrix = np.zeros((n, 0))
        cap_vector = np.array([])
        n_resources = 0
    
    if cap_vector is None or len(cap_vector) != n_resources:
        cap_vector = np.ones(n_resources)
    
    # Build Successor list for faster lookup
    succ_list = [list() for _ in range(n)]
    for i in range(n):
        for p in pred_list[i]:
            succ_list[p].append(i)
            
    # Compute Topological Order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            in_degree[i] += 1
            
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for j in succ_list[node]:
            temp_in_deg[j] -= 1
            if temp_in_deg[j] == 0:
                queue.append(j)
                
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
                
    # Forward Pass: Earliest Start
    earliest_start = np.zeros(n)
    for node in topo_order:
        if pred_list[node]:
            earliest_start[node] = max(earliest_start[p] + durations[p] for p in pred_list[node])
            
    # Backward Pass: Latest Start
    project_end = max(earliest_start[i] + durations[i] for i in range(n)) if n > 0 else 0
    latest_finish = np.full(n, project_end)
    for node in reversed(topo_order):
        if succ_list[node]:
            latest_finish[node] = min(earliest_start[s] for s in succ_list[node])
        else:
            latest_finish[node] = project_end
            
    latest_start = latest_finish - durations
    total_float = latest_start - earliest_start
    total_float = np.maximum(total_float, 0)
    
    # Compute Longest Path from node to end (Critical Path Length)
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if succ_list[node]:
            longest_path[node] = durations[node] + max(longest_path[s] for s in succ_list[node])
        else:
            longest_path[node] = durations[node]
    
    # NEW STRATEGY: Resource Contention Metric
    # For each activity, compute how much resource demand it shares with its direct successors.
    # High shared demand means scheduling this activity early frees resources for many dependents.
    # This complements critical path by prioritizing "resource unlockers".
    
    resource_contention = np.zeros(n)
    if n_resources > 0:
        for i in range(n):
            if succ_list[i]:
                # Sum over successors of shared resource demand
                for s in succ_list[i]:
                    # Element-wise product of demands, summed over resources
                    resource_contention[i] += np.sum(demand_matrix[i] * demand_matrix[s])
    
    # Normalize components
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
        
    norm_duration = normalize(durations)
    norm_longest_path = normalize(longest_path)
    norm_float_inv = 1.0 - normalize(total_float)
    norm_contention = normalize(resource_contention)
    
    # Base Priority: Critical Path + Duration + Low Float + High Resource Contention
    # Weights: CP is king, but Resource Contention helps break ties in resource-constrained environments
    # and prioritizes activities that unlock resources for dependents.
    base_priority = (
        0.35 * norm_longest_path +
        0.15 * norm_duration +
        0.15 * norm_float_inv +
        0.35 * norm_contention
    )
    
    # Critical mask: activities with very low float
    max_float = np.max(total_float) if n > 0 else 0
    critical_mask = total_float < (0.1 * max_float if max_float > 0 else 0)
    
    adjusted_priority = base_priority.copy()
    
    # Boost critical, high-contention activities to free resources early
    for i in range(n):
        if critical_mask[i]:
            adjusted_priority[i] += 0.1 * norm_contention[i]
    
    return adjusted_priority
# EVOLVE-BLOCK-END
