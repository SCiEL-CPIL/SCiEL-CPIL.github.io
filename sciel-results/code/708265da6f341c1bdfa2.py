"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random
from collections import deque

def _get_elapsed(start_time):
    return time.time() - start_time

def _remaining_time(start_time, limit=150.0, safety_margin=10.0):
    return limit - _get_elapsed(start_time) - safety_margin

def _compute_successors(preds, n):
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
    return successors

def _compute_longest_path_downstream(times, preds, successors, n):
    """
    Compute the longest path from each node to any sink (downstream).
    This is the 'critical path' length if we start at the node.
    """
    # Topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
    
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        # Cycle handling: append remaining
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # DP in reverse topological order
    longest_path = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            longest_path[u] = times[u] + max(longest_path[v] for v in successors[u])
        else:
            longest_path[u] = times[u]
            
    return longest_path

def _compute_longest_path_upstream(times, preds, n):
    """
    Compute the longest path from any source to each node (upstream).
    """
    # Reverse graph logic: longest path from source to i
    # This is equivalent to longest path in reverse graph from i to source? 
    # Actually, let's just compute rank or depth.
    # Depth of node i is max(depth[p] + times[p] for p in preds[i]) + times[i]? No, depth is usually just path length.
    # Let's compute the longest path ending at i.
    
    # Topological sort
    in_degree = [0] * n
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
            successors[p].append(i)
            
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    longest_path_up = np.zeros(n)
    for u in topo_order:
        if preds[u]:
            longest_path_up[u] = times[u] + max(longest_path_up[p] for p in preds[u])
        else:
            longest_path_up[u] = times[u]
            
    return longest_path_up

def _simulate(times, preds, successors, cycle_time, priorities, n):
    """
    Simulate the greedy assembly line balancing.
    Returns the number of stations used.
    """
    # State
    assigned = [False] * n
    available = []
    # Track successors count to know when a task becomes available
    remaining_preds = [len(preds[i]) for i in range(n)]
    
    # Initialize available tasks
    for i in range(n):
        if remaining_preds[i] == 0:
            available.append(i)
            
    stations = 0
    current_station_time = 0.0
    
    # We need to select the highest priority task from available that fits.
    # Since N is small (30), we can just sort available by priority each time or scan.
    # Scanning is O(N). Sorting is O(K log K). K is small.
    
    while any(not a for a in assigned):
        # If no available tasks, we must start a new station
        if not available:
            stations += 1
            current_station_time = 0.0
            # This shouldn't happen if we initialized correctly and graph is acyclic,
            # but if we just finished a station and there are no available tasks, 
            # it means we are stuck? No, if available is empty, it means all remaining tasks have unmet predecessors.
            # This implies a cycle or error in simulation state. 
            # However, in a valid DAG, if we have assigned some tasks, there must be available tasks 
            # unless all remaining tasks depend on each other (cycle) or we haven't started.
            # Actually, if available is empty at the start of a station, it's an error.
            # But if available becomes empty *after* assigning a task, it means no new tasks became available.
            # We just close the current station and open a new one? 
            # No, if available is empty, we can't assign anything. We must have finished the current station's capacity 
            # or we are in a deadlock. In valid ALB, we just start a new station if we can't fit anything? 
            # Wait, the greedy rule is: "stations are filled with the highest-priority eligible task that fits".
            # If no task fits, we start a new station.
            # If no task is eligible (available), we start a new station? 
            # Actually, if available is empty, it means we have processed all tasks that could be processed.
            # If there are still unassigned tasks, there is a cycle. 
            # Assuming valid input, available should not be empty if there are unassigned tasks, 
            # UNLESS the current station is full and we haven't checked for new availability?
            # No, availability is independent of station fullness.
            # If available is empty, we are stuck. 
            # Let's assume the simulation loop handles "no task fits" by starting a new station.
            # But if available is empty, we can't even pick a task to check if it fits.
            # This case only happens if there's a cycle. We'll break to avoid infinite loop.
            if not any(not a for a in assigned):
                break
            else:
                # Force break on invalid state
                break

        # Find the best task to assign
        best_task = -1
        best_priority = -1e9
        
        # Filter available tasks that fit in current station
        candidates = []
        for t in available:
            if times[t] <= cycle_time - current_station_time:
                if priorities[t] > best_priority:
                    best_priority = priorities[t]
                    best_task = t
        
        if best_task == -1:
            # No task fits in the current station
            stations += 1
            current_station_time = 0.0
            # Retry with new station
            # The loop will continue and pick the best available task for the new station
            continue
        
        # Assign best_task
        assigned[best_task] = True
        current_station_time += times[best_task]
        
        # Remove from available
        available.remove(best_task)
        
        # Update successors
        for s in successors[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                available.append(s)
                
    return stations

def priority(instance):
    """
    Compute priority for each task.
    Returns an array of scores, one per task.
    """
    start_time = time.time()
    
    try:
        times = np.asarray(instance["times"], dtype=float)
        preds = instance["predecessors"]
        cycle_time = instance["cycle_time"]
        n = len(times)
        
        # Pre-compute graph properties
        successors = _compute_successors(preds, n)
        
        # 1. Compute Heuristic Priorities
        # Critical Path Downstream (urgency)
        lp_down = _compute_longest_path_downstream(times, preds, successors, n)
        # Longest Path Upstream (position)
        lp_up = _compute_longest_path_upstream(times, preds, n)
        
        # Successor Count
        succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
        
        def normalize(arr):
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())
            
        lp_down_norm = normalize(lp_down)
        lp_up_norm = normalize(lp_up)
        sc_norm = normalize(succ_count)
        
        # Initial Priority: Weighted sum
        # Heuristic: Downstream CP is most important, then Upstream, then Succ Count
        init_priorities = 0.6 * lp_down_norm + 0.3 * lp_up_norm + 0.1 * sc_norm
        
        # 2. Local Search (Simulated Annealing)
        current_priorities = init_priorities.copy()
        best_priorities = init_priorities.copy()
        
        current_stations = _simulate(times, preds, successors, cycle_time, current_priorities, n)
        best_stations = current_stations
        
        # Parameters for SA
        T = 1.0
        alpha = 0.95
        # Number of tasks to perturb
        k = max(1, n // 5)
        
        # Time limit check: We have 150s total. We want to stop at ~140s.
        # _simulate is very fast. Let's run as many iterations as possible.
        
        while _remaining_time(start_time) > 0:
            # Generate neighbor
            neighbor = current_priorities.copy()
            # Perturb k random tasks
            indices = random.sample(range(n), k)
            for idx in indices:
                # Add Gaussian noise
                neighbor[idx] += random.gauss(0, 0.1)
            
            # Evaluate
            neighbor_stations = _simulate(times, preds, successors, cycle_time, neighbor, n)
            
            # Acceptance Criterion
            if neighbor_stations < current_stations:
                current_priorities = neighbor
                current_stations = neighbor_stations
                if current_stations < best_stations:
                    best_stations = current_stations
                    best_priorities = current_priorities.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = neighbor_stations - current_stations
                if delta == 0:
                    # Tie, accept with 50% probability to explore
                    if random.random() < 0.5:
                        current_priorities = neighbor
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_priorities = neighbor
                        current_stations = neighbor_stations
            
            T *= alpha
            if T < 0.01:
                # Restart with new random initial or reheat
                # Let's just reheat slightly or restart from best
                # Reheat
                T = 0.5
                
    except Exception as e:
        # Fallback to simple LPT if anything goes wrong
        try:
            times = np.asarray(instance["times"], dtype=float)
            return times
        except:
            return np.zeros(len(instance["times"]))
            
    return best_priorities

# EVOLVE-BLOCK-END
