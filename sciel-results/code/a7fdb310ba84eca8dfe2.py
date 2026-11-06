import numpy as np
from collections import deque
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Deadline guard: start tracking time
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    if len(topo_order) < n:
        return times
        
    # Compute Level (Longest Path to End)
    levels = [0.0] * n
    for u in reversed(topo_order):
        max_succ_level = 0.0
        for v in successors[u]:
            if levels[v] > max_succ_level:
                max_succ_level = levels[v]
        levels[u] = times[u] + max_succ_level
        
    # Compute transitive successors using bitmasks
    succ_mask = [0] * n
    for u in reversed(topo_order):
        mask = 0
        for v in successors[u]:
            mask |= succ_mask[v]
            mask |= (1 << v)
        succ_mask[u] = mask
        
    trans_succ_count = [bin(mask).count('1') for mask in succ_mask]
    
    # Compute total downstream time
    total_downstream_time = [0.0] * n
    for u in range(n):
        mask = succ_mask[u]
        t_sum = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            t_sum += times[idx]
            m ^= lsb
        total_downstream_time[u] = t_sum

    # Helper: simulate greedy packing given a priority order
    def simulate_packing(priority_order):
        """
        Simulate greedy packing with given priority order.
        Returns number of stations used.
        """
        assigned = [False] * n
        remaining_preds = [list(preds[i]) for i in range(n)]
        eligible = set()
        for i in range(n):
            if not remaining_preds[i]:
                eligible.add(i)
        
        stations = 0
        while assigned.count(False) > 0:
            if time.time() - start_time > deadline:
                return None  # Timeout
            
            # Find highest-priority eligible task that fits
            best_task = -1
            best_priority = -1
            for task in eligible:
                if times[task] <= cycle_time:
                    if priority_order[task] > best_priority:
                        best_priority = priority_order[task]
                        best_task = task
            
            if best_task == -1:
                # No task fits, new station
                stations += 1
                # Reset eligible to all unassigned tasks with no remaining preds
                eligible = set()
                for i in range(n):
                    if not assigned[i] and not remaining_preds[i]:
                        eligible.add(i)
                continue
            
            # Assign best_task to current station
            assigned[best_task] = True
            # Update predecessors of successors
            for v in successors[best_task]:
                remaining_preds[v].remove(best_task)
                if not remaining_preds[v]:
                    eligible.add(v)
            
            # Remove from eligible
            eligible.discard(best_task)
            
            # Check if we should start a new station
            # Simple heuristic: if no more eligible tasks fit, start new station
            any_fits = False
            for task in eligible:
                if times[task] <= cycle_time:
                    any_fits = True
                    break
            if not any_fits:
                stations += 1
                eligible = set()
                for i in range(n):
                    if not assigned[i] and not remaining_preds[i]:
                        eligible.add(i)
        
        return stations
    
    # Generate multiple priority orderings and evaluate
    # Ordering 1: Based on transitive successors + downstream time
    order1 = np.zeros(n)
    for i in range(n):
        order1[i] = trans_succ_count[i] * 10.0 + total_downstream_time[i]
    
    # Ordering 2: Based on levels (critical path)
    order2 = np.array(levels)
    
    # Ordering 3: Combined
    order3 = order1 * 0.5 + order2 * 0.5
    
    # Evaluate each ordering
    results = []
    for order in [order1, order2, order3]:
        if time.time() - start_time > deadline:
            break
        stations = simulate_packing(order)
        if stations is not None:
            results.append(stations)
    
    # If we got results, use the best ordering's pattern
    if results:
        best_idx = np.argmin(results)
        # Use the best ordering as the priority
        if best_idx == 0:
            return order1
        elif best_idx == 1:
            return order2
        else:
            return order3
    
    # Fallback: return the combined ordering
    return order3
