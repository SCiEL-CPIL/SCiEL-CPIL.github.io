"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def _build_graph(instance):
    """Build adjacency and in-degree structures."""
    n = len(instance["times"])
    preds = instance.get("predecessors", [set() for _ in range(n)])
    # Normalize predecessors to lists of ints
    pred_lists = []
    for i in range(n):
        if isinstance(preds[i], (list, tuple, set)):
            pred_lists.append(list(preds[i]))
        else:
            pred_lists.append([])
    
    successors = defaultdict(list)
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    return pred_lists, successors, in_degree


def _compute_longest_path_from_start(instance):
    """Compute longest path from start to each task (earliest start + duration)."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    pred_lists, successors, in_degree = _build_graph(instance)
    
    # Topological order
    topo_order = []
    in_deg = in_degree[:]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    
    # Longest path from start (including task duration)
    longest_from_start = np.zeros(n)
    for node in topo_order:
        if pred_lists[node]:
            longest_from_start[node] = max(longest_from_start[p] for p in pred_lists[node]) + times[node]
        else:
            longest_from_start[node] = times[node]
    
    return longest_from_start


def _compute_longest_path_to_end(instance):
    """Compute longest path from each task to end (including task duration)."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    pred_lists, successors, in_degree = _build_graph(instance)
    
    # Reverse topological order
    topo_order = []
    in_deg = in_degree[:]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    
    # Longest path to end (including task duration)
    longest_to_end = np.zeros(n)
    for node in reversed(topo_order):
        if successors[node]:
            longest_to_end[node] = times[node] + max(longest_to_end[s] for s in successors[node])
        else:
            longest_to_end[node] = times[node]
    
    return longest_to_end


def _compute_rank(instance):
    """Compute rank (longest path to end, standard rank)."""
    return _compute_longest_path_to_end(instance)


def _greedy_assign(instance, priority_scores):
    """
    Greedy station filling: repeatedly pick highest-priority eligible task that fits.
    Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance.get("cycle_time", 40)
    n = len(times)
    pred_lists, successors, in_degree = _build_graph(instance)
    
    # Track remaining predecessors
    remaining_preds = [in_degree[i] for i in range(n)]
    assigned = [False] * n
    station_load = 0.0
    num_stations = 1
    
    # Use a list for eligible tasks (tasks with all predecessors assigned)
    eligible = [i for i in range(n) if remaining_preds[i] == 0]
    
    while any(not assigned[i] for i in range(n)):
        # Find best eligible task that fits
        best_task = -1
        best_priority = -np.inf
        for i in eligible:
            if not assigned[i] and times[i] <= cycle_time - station_load:
                if priority_scores[i] > best_priority:
                    best_priority = priority_scores[i]
                    best_task = i
        
        if best_task == -1:
            # No task fits in current station, start new station
            num_stations += 1
            station_load = 0.0
            # Re-check
            best_task = -1
            best_priority = -np.inf
            for i in eligible:
                if not assigned[i] and times[i] <= cycle_time - station_load:
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task = i
        
        if best_task == -1:
            # Should not happen if all tasks fit in cycle time
            break
        
        # Assign best_task to current station
        assigned[best_task] = True
        station_load += times[best_task]
        
        # Remove from eligible and update successors
        eligible.remove(best_task)
        for succ in successors[best_task]:
            remaining_preds[succ] -= 1
            if remaining_preds[succ] == 0:
                eligible.append(succ)
    
    return num_stations


def _evaluate_priority(instance, priority_scores):
    """Evaluate a priority scheme by counting stations."""
    return _greedy_assign(instance, priority_scores)


def priority(instance):
    """
    Return priority scores for each task.
    Uses a composite of critical path length and task duration.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Compute critical path components
    lp_start = _compute_longest_path_from_start(instance)
    lp_end = _compute_longest_path_to_end(instance)
    
    # Composite priority: emphasize tasks on critical path, with tie-break by duration
    # Higher lp_end means more downstream work depends on this task
    # Higher lp_start means more upstream work feeds into this task
    # Use lp_end as primary (standard rank), with small weight on duration for tie-breaking
    priority_scores = lp_end + 0.1 * times
    
    return priority_scores
# EVOLVE-BLOCK-END
