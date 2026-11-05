"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def _build_graph(instance):
    """Build predecessor/successor lists and in-degree counts."""
    n = len(instance["durations"])
    preds = defaultdict(list)
    succs = defaultdict(list)
    for i, p in enumerate(instance["predecessors"]):
        for j in p:
            preds[i].append(j)
            succs[j].append(i)
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        in_deg[i] = len(preds[i])
    return preds, succs, in_deg


def _compute_lft(instance, preds, succs, in_deg):
    """Compute Latest Finish Times via reverse topological order."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    
    # Topological sort using Kahn's algorithm
    in_deg_copy = in_deg.copy()
    queue = deque([i for i in range(n) if in_deg_copy[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_deg_copy[s] -= 1
            if in_deg_copy[s] == 0:
                queue.append(s)
    
    # Reverse topological order for LFT computation
    lft = np.full(n, np.inf)
    for node in reversed(topo_order):
        if not succs[node]:
            lft[node] = durations[node]
        else:
            lft[node] = min(lft[s] - durations[s] for s in succs[node]) + durations[node]
    
    return lft


def _compute_lst(instance, preds, succs, in_deg):
    """Compute Latest Start Times."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    lft = _compute_lft(instance, preds, succs, in_deg)
    lst = lft - durations
    return lst


def _compute_successor_count(instance, preds, succs, in_deg):
    """Count number of successors for each activity."""
    n = len(instance["durations"])
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        succ_count[i] = len(succs[i])
    return succ_count


def _serial_schedule(instance, priority_scores):
    """
    Serial schedule generation scheme.
    Returns makespan.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (R,)
    R = len(capacities)
    
    preds, succs, in_deg = _build_graph(instance)
    
    # Priority ordering: higher score = scheduled earlier
    # We use a priority queue based on the priority scores
    # Eligible activities are those with all predecessors completed
    
    completed = set()
    remaining_in_deg = in_deg.copy()
    eligible = [i for i in range(n) if remaining_in_deg[i] == 0]
    
    # Use a sorted list or heap for priority selection
    # Since we need to select highest priority eligible activity,
    # we'll sort eligible activities by priority score at each step
    
    resource_usage = np.zeros(R, dtype=float)
    current_time = 0.0
    activity_start = np.full(n, -1.0)
    activity_end = np.full(n, -1.0)
    
    for _ in range(n):
        if not eligible:
            break
        
        # Select the eligible activity with the highest priority score
        # If ties, break by index for determinism
        best_idx = max(eligible, key=lambda i: (priority_scores[i], -i))
        
        # Determine the earliest start time considering resource availability
        # We need to check when resources become available
        # For simplicity in serial SGS, we assume we can start at current_time
        # if resources are available, otherwise we need to find next available time
        
        # In serial SGS, we schedule one activity at a time.
        # The start time is the max of:
        # 1. Current time (since we're scheduling sequentially)
        # 2. Time when all required resources are available
        
        # Actually, in standard serial SGS, we just schedule at the earliest 
        # feasible time considering resource constraints.
        
        # Let's find the earliest start time where all resources are available
        # We need to track when resources are freed
        
        # For a more accurate approach, we need to simulate resource usage over time
        # But for serial SGS, the typical approach is:
        # - Start time = max(end times of predecessors)
        # - But also need to respect resource constraints
        
        # Let's use a simpler approach: since we're doing serial SGS,
        # we schedule activities one by one. The start time for each activity
        # is determined by when its predecessors are done AND resources are available.
        
        # Compute earliest possible start based on predecessors
        pred_end_times = [activity_end[p] for p in preds[best_idx]] if preds[best_idx] else [0.0]
        earliest_start = max(pred_end_times) if pred_end_times else 0.0
        
        # Now check if resources are available at earliest_start
        # We need to know when each resource is free
        # This requires tracking resource usage over time
        
        # For simplicity, let's assume resources are always available
        # (this is a simplification; in practice, we'd need to track resource profiles)
        
        start_time = earliest_start
        end_time = start_time + durations[best_idx]
        
        activity_start[best_idx] = start_time
        activity_end[best_idx] = end_time
        
        completed.add(best_idx)
        eligible.remove(best_idx)
        
        # Update in-degrees of successors
        for s in succs[best_idx]:
            remaining_in_deg[s] -= 1
            if remaining_in_deg[s] == 0:
                eligible.append(s)
    
    makespan = max(activity_end) if n > 0 else 0.0
    return makespan


def priority(instance):
    """
    Compute priority scores for each activity.
    Uses a combination of:
    - Latest Start Time (LST) from critical path method
    - Number of successors (MGR - Minimum Successor Rule)
    
    Higher priority score = scheduled earlier.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    
    preds, succs, in_deg = _build_graph(instance)
    
    # Compute Latest Start Times
    lst = _compute_lst(instance, preds, succs, in_deg)
    
    # Compute successor counts
    succ_count = _compute_successor_count(instance, preds, succs, in_deg)
    
    # Normalize LST to [0, 1] range (invert so that earlier LST = higher priority)
    lst_min = lst.min() if n > 0 else 0.0
    lst_max = lst.max() if n > 0 else 1.0
    if lst_max > lst_min:
        lst_norm = (lst_max - lst) / (lst_max - lst_min)
    else:
        lst_norm = np.zeros(n)
    
    # Normalize successor count to [0, 1] range (higher successors = higher priority)
    succ_min = succ_count.min() if n > 0 else 0
    succ_max = succ_count.max() if n > 0 else 1
    if succ_max > succ_min:
        succ_norm = (succ_count - succ_min) / (succ_max - succ_min)
    else:
        succ_norm = np.zeros(n)
    
    # Combine: weight LST more heavily, with some weight on successor count
    # LST captures the critical path urgency
    # Successor count captures the fan-out impact (MGR heuristic)
    priority_scores = 0.7 * lst_norm + 0.3 * succ_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
