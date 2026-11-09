import numpy as np
import heapq
import time

def priority(instance):
    start_time = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Topological Sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.pop(0) 
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        # Cycle or invalid, fallback to simple
        return np.zeros(n)

    # 1. Compute Head Rank (Earliest Start)
    head_rank = np.zeros(n, dtype=float)
    for u in topo_order:
        if not predecessors[u]:
            head_rank[u] = 0
        else:
            head_rank[u] = max(head_rank[p] + durations[p] for p in predecessors[u])
            
    # 2. Compute Tail Rank (Longest Path to End)
    tail_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            tail_rank[u] = durations[u]
        else:
            tail_rank[u] = durations[u] + max(tail_rank[v] for v in successors[u])
            
    # 3. Simulate Schedule using Tail Rank as Priority
    # Returns finish times
    sim_finish = np.zeros(n, dtype=float)
    
    # State for simulation
    remaining_preds = {i: len(predecessors[i]) for i in range(n)}
    eligible = [i for i in range(n) if remaining_preds[i] == 0]
    
    # Heap for eligible activities: (-priority, index)
    # Priority is Tail Rank
    heap = [(-tail_rank[i], i) for i in eligible]
    heapq.heapify(heap)
    
    # Active activities: (finish_time, index)
    active = []
    
    # Resource availability
    available_res = capacities.copy()
    
    current_time = 0.0
    scheduled_count = 0
    
    while scheduled_count < n:
        # Check if any active activity has finished
        # We need to advance time to the next event if no eligible activity can start
        
        # Find best eligible activity that can start at current_time
        best_act = -1
        best_priority = -np.inf
        
        # We need to check if resources are available for any eligible activity.
        # Since we pop from heap, we check the top ones.
        # But an activity at the top might not fit.
        # We need to find the highest priority activity that fits.
        
        # Optimization: Since N is small, we can just scan the heap?
        # Or pop and push back?
        # Let's pop and push back non-fitting ones.
        
        candidates = []
        while heap:
            neg_prio, idx = heapq.heappop(heap)
            prio = -neg_prio
            
            # Check if resources are available
            if np.all(demands[idx] <= available_res + 1e-9):
                best_act = idx
                best_priority = prio
                break
            else:
                candidates.append((neg_prio, idx))
        
        # Push back non-fitting candidates
        for item in candidates:
            heapq.heappush(heap, item)
        
        if best_act != -1:
            # Schedule best_act
            sim_finish[best_act] = current_time + durations[best_act]
            available_res -= demands[best_act]
            heapq.heappush(active, (sim_finish[best_act], best_act))
            scheduled_count += 1
            
            # Update eligible set
            for v in successors[best_act]:
                remaining_preds[v] -= 1
                if remaining_preds[v] == 0:
                    heapq.heappush(heap, (-tail_rank[v], v))
        else:
            # No eligible activity can start at current_time
            # Advance time to next finish event
            if not active:
                # Deadlock? Should not happen in valid DAG with capacities
                break
            
            next_finish_time, finished_act = heapq.heappop(active)
            current_time = next_finish_time
            available_res += demands[finished_act]
            
            # Note: We don't re-add finished_act to heap, it's done.
            # We just updated resources. Loop continues to try scheduling again at new time.
            
    # 4. Compute Delay Criticality
    # cpm_finish = head_rank + duration
    cpm_finish = head_rank + durations
    delay = sim_finish - cpm_finish
    
    # Propagate delay backwards
    # crit[i] = delay[i] + max(crit[succ])
    crit = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            crit[u] = delay[u]
        else:
            crit[u] = delay[u] + max(crit[v] for v in successors[u])
            
    # 5. Final Score
    # Combine Tail Rank and Criticality
    # Normalize or weight?
    # Tail Rank is in time units. Crit is in time units (delay).
    # Weight 1.0 for both seems reasonable.
    scores = tail_rank + 1.0 * crit
    
    return scores
