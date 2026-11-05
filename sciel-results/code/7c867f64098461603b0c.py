"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _greedy_assign(times, preds, cycle_time, priority_scores):
    """Simulate the greedy station filling with given priority scores."""
    n = len(times)
    if n == 0:
        return 0

    # Precompute successors for each task (which tasks depend on it)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)

    remaining_preds = [len(preds[i]) for i in range(n)]
    eligible = [i for i in range(n) if remaining_preds[i] == 0]

    stations = 0
    remaining = n
    # Use a simple list; we'll re-sort eligible each time we need the best
    # But for efficiency, we can maintain a heap. However, priority can change? No, priority is fixed.
    # Wait, the priority is fixed per task. So we can just sort eligible tasks by priority.
    # But eligible set changes. Let's use a heap of (-priority, task_index).
    import heapq
    heap = [(-priority_scores[i], i) for i in eligible]
    heapq.heapify(heap)

    while remaining > 0:
        stations += 1
        load = 0.0
        # We need to pick the highest priority eligible task that fits.
        # But the set of eligible tasks that fit depends on current load.
        # Standard greedy: at each step within a station, pick the eligible task with highest priority that fits.
        # However, the problem says "stations are filled with the highest-priority eligible task that fits the cycle time".
        # This implies a sequential process:
        # 1. Start new station.
        # 2. While there is an eligible task that fits in remaining capacity:
        #    a. Among all eligible tasks that fit, pick the one with highest priority.
        #    b. Assign it, update load, update successors.
        # 3. If no eligible task fits, close station.
        
        # To do this efficiently, we need to repeatedly find the best eligible task that fits.
        # Since n is small (30), we can just scan the heap or eligible list.
        # Let's keep a list of eligible tasks and scan it.
        # Actually, the heap contains all eligible tasks. We need to find the one with highest priority that fits.
        # We can pop from heap until we find one that fits? No, because if it doesn't fit, it might fit in a later station.
        # But if it doesn't fit in the current station, we stop filling this station.
        # So we need to find the max priority task among those that fit.
        
        # Let's just iterate through the heap and find the best one that fits.
        # Since heap is a min-heap of (-priority), the first element is the highest priority.
        # But it might not fit. We need to check all?
        # No, if the highest priority one doesn't fit, no lower priority one with larger time will fit either?
        # Not necessarily. A lower priority task might have a smaller time and fit.
        # So we must check all eligible tasks to find the one with highest priority that fits.
        
        # For n=30, scanning all eligible tasks each time is O(n^2) per station, which is fine.
        
        best_task = -1
        best_prio = -np.inf
        # We need to know which tasks are currently eligible.
        # The heap contains all eligible tasks.
        # Let's extract all, find the best fitting, and re-heapify the rest? That's expensive.
        # Alternatively, since n is small, let's just keep a list of eligible tasks and sort/filter.
        
        # Let's revert to a simpler approach for correctness and speed (n is tiny).
        # Maintain a list `eligible_list`.
        # At each step, find the task in `eligible_list` with max priority such that times[task] <= cycle_time - load.
        
        # I'll rewrite the loop to use a list.
        eligible_list = [i for i in range(n) if len(preds[i]) == 0]
        remaining_preds = [len(preds[i]) for i in range(n)]
        
        stations = 0
        remaining = n
        while remaining > 0:
            stations += 1
            load = 0.0
            # Fill current station
            while True:
                # Find best eligible task that fits
                best_task = -1
                best_prio = -np.inf
                for i in eligible_list:
                    if times[i] <= cycle_time - load + 1e-9:
                        if priority_scores[i] > best_prio:
                            best_prio = priority_scores[i]
                            best_task = i
                if best_task == -1:
                    break
                # Assign best_task
                load += times[best_task]
                remaining -= 1
                # Remove from eligible_list
                eligible_list.remove(best_task)
                # Update successors
                for s in succ[best_task]:
                    remaining_preds[s] -= 1
                    if remaining_preds[s] == 0:
                        eligible_list.append(s)
        
        return stations


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Compute rank (number of predecessors)
    rank = np.zeros(n, dtype=int)
    for i in range(n):
        rank[i] = len(preds[i])
    
    # Compute bottom-level rank (longest path to end)
    # Topological order
    in_degree = [len(preds[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in range(n):
            if node in preds[s]:
                temp_in_degree[s] -= 1
                if temp_in_degree[s] == 0:
                    queue.append(s)
    
    # Reverse topological order for bottom-level rank
    bottom_level = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        # bottom_level[node] = times[node] + max(bottom_level[s] for s in successors)
        max_succ = 0.0
        for s in range(n):
            if node in preds[s]:
                if bottom_level[s] > max_succ:
                    max_succ = bottom_level[s]
        bottom_level[node] = times[node] + max_succ
    
    # Compute number of immediate successors
    num_succ = np.zeros(n, dtype=int)
    for i in range(n):
        for s in range(n):
            if i in preds[s]:
                num_succ[i] += 1
    
    # Compute a combined priority score
    # Higher is better.
    # We want tasks that are "critical" to have high priority.
    # Criticality can be measured by bottom_level rank.
    # Also, tasks with many successors should be prioritized.
    # And tasks with long times should be prioritized.
    
    # Let's try a weighted sum:
    # score = w1 * bottom_level + w2 * times + w3 * num_succ
    # Normalize each component to [0, 1] for stability.
    
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    bl_norm = normalize(bottom_level)
    t_norm = normalize(times)
    ns_norm = normalize(num_succ.astype(float))
    
    # Weights
    w1, w2, w3 = 1.0, 0.5, 0.5
    score = w1 * bl_norm + w2 * t_norm + w3 * ns_norm
    
    # Try to improve by local search?
    # Given the time limit, we can try a few random perturbations or hill climbing.
    # But n=30, and greedy is O(n^3) ~ 27000 ops. Very fast.
    # We can afford to try a few variations.
    
    best_score = score
    best_stations = _greedy_assign(times, preds, cycle_time, score)
    
    # Try a few different weight combinations
    weights_to_try = [
        (1.0, 0.5, 0.5),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.5, 1.0, 0.5),
        (1.0, 0.2, 0.2),
        (2.0, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ]
    
    for w1, w2, w3 in weights_to_try:
        s = w1 * bl_norm + w2 * t_norm + w3 * ns_norm
        st = _greedy_assign(times, preds, cycle_time, s)
        if st < best_stations:
            best_stations = st
            best_score = s
    
    return best_score
# EVOLVE-BLOCK-END
