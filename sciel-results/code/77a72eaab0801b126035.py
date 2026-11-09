"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    n = len(instance["times"])
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", [])
    # preds is a list of lists; preds[i] = list of predecessor indices for task i
    succs = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            indeg[i] += 1
    return n, times, preds, succs, indeg


def _compute_downstream_work(n, times, succs):
    """
    downstream_work[i] = times[i] + sum of downstream_work[j] for j in succs[i]
    This is the total work in the subtree rooted at i (including i itself).
    We compute via topological order (reverse postorder or just topological sort).
    """
    # We need a topological order. Since we have predecessors, we can do a simple approach:
    # Process nodes in reverse topological order.
    # Get indegrees for a standard topo sort
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for j in succs[i]:
            pass  # succs already built
    # Actually, let's just compute indegrees from preds
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        indeg[i] = len(preds[i]) if i < len(preds) else 0
    
    # Wait, preds is passed. Let me restructure.
    # Actually, let me just use a simple iterative approach with a stack for topological sort.
    # Or simpler: since n is small (30), we can just do a DFS-based postorder.
    
    visited = [False] * n
    postorder = []
    
    def dfs(u):
        visited[u] = True
        for v in succs[u]:
            if not visited[v]:
                dfs(v)
        postorder.append(u)
    
    for i in range(n):
        if not visited[i]:
            dfs(i)
    
    # postorder is in reverse topological order (sinks first)
    downstream_work = np.zeros(n, dtype=float)
    for i in range(n):
        for j in succs[i]:
            downstream_work[i] += downstream_work[j]
        downstream_work[i] += times[i]
    
    return downstream_work


def _compute_critical_path(n, times, preds, succs):
    """
    Compute the longest path from any source to each node (earliest start time).
    Also compute the longest path from each node to any sink (latest finish contribution).
    """
    # Longest path from sources to i (forward)
    longest_from_source = np.zeros(n, dtype=float)
    # We need topological order
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        indeg[i] = len(preds[i])
    
    from collections import deque
    queue = deque()
    for i in range(n):
        if indeg[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    
    # Forward pass
    for u in topo_order:
        for v in succs[u]:
            if longest_from_source[u] + times[u] > longest_from_source[v]:
                longest_from_source[v] = longest_from_source[u] + times[u]
    
    # Longest path from i to sinks (backward)
    longest_to_sink = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if times[v] + longest_to_sink[v] > longest_to_sink[u]:
                longest_to_sink[u] = times[v] + longest_to_sink[v]
    
    return longest_from_source, longest_to_sink, topo_order


def _greedy_assign(priority_scores, n, times, preds, cycle_time):
    """
    Greedy station assignment: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    assigned = [False] * n
    remaining = set(range(n))
    stations = 0
    
    while remaining:
        # Find eligible tasks: all predecessors must be assigned
        eligible = []
        for i in remaining:
            all_preds_done = True
            for p in preds[i]:
                if not assigned[p]:
                    all_preds_done = False
                    break
            if all_preds_done:
                eligible.append(i)
        
        if not eligible:
            # Should not happen if instance is valid, but break to avoid infinite loop
            break
        
        # Pick the highest priority task that fits in cycle_time
        best_task = -1
        best_priority = -np.inf
        for i in eligible:
            if times[i] <= cycle_time:
                if priority_scores[i] > best_priority:
                    best_priority = priority_scores[i]
                    best_task = i
        
        if best_task == -1:
            # No eligible task fits; pick the one with smallest time among eligible
            # This shouldn't happen if all times <= cycle_time, but handle gracefully
            best_task = min(eligible, key=lambda i: times[i])
        
        assigned[best_task] = True
        remaining.discard(best_task)
        stations += 1
        
        # Wait, this is wrong. Each station can hold MULTIPLE tasks as long as they fit
        # and precedence is respected. Let me redo this.
        pass
    
    # Redo: proper greedy station filling
    assigned = [False] * n
    remaining = set(range(n))
    stations = 0
    station_time = 0.0
    station_tasks = []
    
    while remaining:
        # Find eligible tasks
        eligible = []
        for i in remaining:
            all_preds_done = True
            for p in preds[i]:
                if not assigned[p]:
                    all_preds_done = False
                    break
            if all_preds_done:
                eligible.append(i)
        
        if not eligible:
            break
        
        # Sort eligible by priority descending
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        # Try to add tasks to current station
        added_any = False
        for i in eligible:
            if times[i] <= (cycle_time - station_time):
                # Add this task to current station
                assigned[i] = True
                remaining.discard(i)
                station_time += times[i]
                station_tasks.append(i)
                added_any = True
                # Re-evaluate eligible since adding i might make its successors eligible
                # But we should continue trying to add more tasks to this station
                # Let's recompute eligible after each addition for correctness
                # Actually, let's just break and re-loop to get updated eligible set
                break
        
        if not added_any:
            # No task fits in current station; start a new station
            if station_tasks:
                stations += 1
            station_time = 0.0
            station_tasks = []
            # Try again with fresh eligible set
            continue
        
        # Check if all tasks are assigned
        if not remaining:
            if station_tasks:
                stations += 1
            break
    
    if station_tasks:
        stations += 1
    
    return stations


def _greedy_assign_v2(priority_scores, n, times, preds, cycle_time):
    """
    Proper greedy: fill one station at a time. For each station, repeatedly pick the
    highest-priority eligible task that fits, until no more fit.
    """
    assigned = [False] * n
    remaining = set(range(n))
    stations = 0
    
    while remaining:
        station_time = 0.0
        added_any = False
        
        while True:
            # Find eligible tasks
            eligible = []
            for i in remaining:
                all_preds_done = True
                for p in preds[i]:
                    if not assigned[p]:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(i)
            
            if not eligible:
                break
            
            # Pick highest priority task that fits
            best_task = -1
            best_priority = -np.inf
            for i in eligible:
                if times[i] <= (cycle_time - station_time):
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task = i
            
            if best_task == -1:
                break
            
            assigned[best_task] = True
            remaining.discard(best_task)
            station_time += times[best_task]
            added_any = True
        
        if not added_any:
            # No task could be added to this station; this shouldn't happen if
            # all remaining tasks have time <= cycle_time and have eligible predecessors
            # Force progress: pick any eligible task (smallest time) to avoid infinite loop
            eligible = []
            for i in remaining:
                all_preds_done = True
                for p in preds[i]:
                    if not assigned[p]:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(i)
            if eligible:
                best_task = min(eligible, key=lambda i: times[i])
                assigned[best_task] = True
                remaining.discard(best_task)
                added_any = True
            else:
                # Deadlock; break
                break
        
        stations += 1
    
    return stations


def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, times, preds, succs, indeg = _build_graph(instance)
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Compute downstream work
    downstream_work = _compute_downstream_work(n, times, succs)
    
    # Compute critical path info
    longest_from_source, longest_to_sink, topo_order = _compute_critical_path(n, times, preds, succs)
    
    # Work Content Density: time[i] / downstream_work[i]
    # Avoid division by zero
    wcd = np.zeros(n, dtype=float)
    for i in range(n):
        if downstream_work[i] > 0:
            wcd[i] = times[i] / downstream_work[i]
    
    # Critical path priority: longest_to_sink[i] (how much work remains after i)
    # Or use longest_from_source + times[i] + longest_to_sink[i] as total path through i
    
    # Combine: use a weighted combination
    # Priority = wcd * alpha + (longest_to_sink[i] / max_longest) * beta
    # Let's normalize longest_to_sink
    max_lts = np.max(longest_to_sink) if np.max(longest_to_sink) > 0 else 1.0
    norm_lts = longest_to_sink / max_lts
    
    # Try different combinations
    candidates = []
    
    # Candidate 1: Pure WCD
    candidates.append(wcd.copy())
    
    # Candidate 2: WCD + normalized longest_to_sink
    candidates.append(0.5 * wcd + 0.5 * norm_lts)
    
    # Candidate 3: WCD + normalized (longest_from_source + times)
    max_lfs = np.max(longest_from_source + times) if np.max(longest_from_source + times) > 0 else 1.0
    norm_lfs = (longest_from_source + times) / max_lfs
    candidates.append(0.5 * wcd + 0.5 * norm_lfs)
    
    # Candidate 4: Product of WCD and norm_lts
    candidates.append(wcd * (1.0 + norm_lts))
    
    # Candidate 5: times / (downstream_work + 1) to avoid zero division issues
    wcd_safe = times / (downstream_work + 1.0)
    candidates.append(wcd_safe)
    
    # Candidate 6: times * norm_lts
    candidates.append(times * norm_lts)
    
    # Evaluate each candidate with greedy assignment
    best_priority = None
    best_stations = float('inf')
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        stations = _greedy_assign_v2(cand, n, times, preds, cycle_time)
        if stations < best_stations:
            best_stations = stations
            best_priority = cand
    
    # If we still have time, try some perturbed versions
    if time.time() < deadline - 20.0 and best_priority is not None:
        # Try small random perturbations of the best candidate
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            perturbed = best_priority + rng.normal(0, 0.01, size=n)
            stations = _greedy_assign_v2(perturbed, n, times, preds, cycle_time)
            if stations < best_stations:
                best_stations = stations
                best_priority = perturbed
    
    if best_priority is None:
        # Fallback: just use times
        best_priority = times.copy()
    
    return best_priority
# EVOLVE-BLOCK-END
