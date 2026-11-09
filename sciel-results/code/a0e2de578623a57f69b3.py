"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import heapq
import time


def _build_graph(instance):
    """Build adjacency and in-degree arrays from the instance."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # preds is a list of lists, where preds[i] is the list of predecessor indices of task i
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    return times, adj, in_deg


def _topological_sort(adj, in_deg, n):
    """Return a topological ordering of tasks."""
    queue = deque()
    for i in range(n):
        if in_deg[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)
    
    return order


def _compute_rank(instance, direction="downstream"):
    """
    Compute rank of each task.
    'downstream': longest path from task to any successor (including task's own time)
    'upstream': longest path from any predecessor to task (including task's own time)
    """
    times, adj, in_deg = _build_graph(instance)
    n = len(times)
    
    if direction == "downstream":
        # Reverse graph
        radj = [[] for _ in range(n)]
        rin_deg = np.zeros(n, dtype=int)
        for i in range(n):
            for j in adj[i]:
                radj[j].append(i)
                rin_deg[i] += 1
        
        # Process in reverse topological order (from sinks to sources)
        # Use Kahn's on reversed graph
        queue = deque()
        for i in range(n):
            if rin_deg[i] == 0:
                queue.append(i)
        
        rank = times.copy()
        while queue:
            node = queue.popleft()
            for pred in radj[node]:
                rank[pred] = max(rank[pred], rank[node] + times[pred])
                rin_deg[pred] -= 1
                if rin_deg[pred] == 0:
                    queue.append(pred)
        
        # Wait, this is wrong. Let me redo.
        # For downstream rank: rank[i] = times[i] + max(rank[j] for j in successors of i)
        # We need to process in reverse topological order.
        
        # Let's use a different approach: compute longest path from each node to any sink.
        # Use memoized DFS or iterative from sinks.
        
        # Actually, let's just compute it properly.
        # rank_down[i] = times[i] + max(rank_down[j] for j in adj[i])
        # Process nodes in reverse topological order.
        
        order = _topological_sort(adj, in_deg.copy(), n)
        rank = np.zeros(n, dtype=float)
        for i in reversed(order):
            max_succ = 0.0
            for j in adj[i]:
                max_succ = max(max_succ, rank[j])
            rank[i] = times[i] + max_succ
        
        return rank
    
    else:  # upstream
        # rank_up[i] = times[i] + max(rank_up[j] for j in predecessors of i)
        # Process nodes in topological order.
        order = _topological_sort(adj, in_deg.copy(), n)
        rank = np.zeros(n, dtype=float)
        for i in order:
            max_pred = 0.0
            for j in preds_of(i, instance):
                max_pred = max(max_pred, rank[j])
            rank[i] = times[i] + max_pred
        
        return rank


def preds_of(i, instance):
    return instance["predecessors"][i]


def _simulate(instance, priority_scores):
    """
    Simulate the greedy list scheduling with given priority scores.
    Returns number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    # Track state
    completed = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining_in_deg = in_deg.copy()
    
    # Eligible tasks: those with all predecessors completed
    # We'll use a priority queue (max-heap via negation)
    
    while True:
        # Find eligible tasks: not completed, in_deg == 0
        # For efficiency, we could maintain a set, but for small n it's fine
        eligible = []
        for i in range(n):
            if not completed[i] and remaining_in_deg[i] == 0:
                eligible.append(i)
        
        if not eligible:
            # No more tasks, but this shouldn't happen if we have a valid DAG
            break
        
        # Sort eligible by priority (highest first)
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        # Try to fit tasks into current station
        placed_any = False
        for i in eligible:
            if station_load + times[i] <= cycle_time:
                # Place task i
                completed[i] = True
                station_load += times[i]
                placed_any = True
                # Update in-degrees of successors
                for j in adj[i]:
                    remaining_in_deg[j] -= 1
        
        if not placed_any:
            # Move to next station
            num_stations += 1
            station_load = 0.0
            # Continue trying to place from eligible (they're still eligible)
            # But we need to re-sort? No, the eligible set is the same, just station_load is reset.
            # Actually, we should try again with the same eligible set.
            # But the loop above already tried all eligible tasks. If none fit, we go to next station.
            # And then we try the same eligible tasks again.
            # This could loop infinitely if a task is larger than cycle_time, but we assume valid input.
            continue
        
        # Check if all tasks are completed
        if all(completed):
            break
    
    return num_stations


def priority(instance):
    """
    Score each assembly task. The evaluator uses these scores to greedily fill stations.
    We want to minimize the mean number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Compute downstream rank (longest path to any successor)
    preds = instance["predecessors"]
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    order = _topological_sort(adj, in_deg.copy(), n)
    
    # Downstream rank
    rank_down = np.zeros(n, dtype=float)
    for i in reversed(order):
        max_succ = 0.0
        for j in adj[i]:
            max_succ = max(max_succ, rank_down[j])
        rank_down[i] = times[i] + max_succ
    
    # Upstream rank
    rank_up = np.zeros(n, dtype=float)
    for i in order:
        max_pred = 0.0
        for p in preds[i]:
            max_pred = max(max_pred, rank_up[p])
        rank_up[i] = times[i] + max_pred
    
    # Combine: use downstream rank as primary priority
    # Higher downstream rank means the task has more work after it, so it should be scheduled earlier
    return rank_down

# EVOLVE-BLOCK-END
