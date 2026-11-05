"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict, deque

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 120s limit


def _time_remaining():
    return max(0.0, _DEADLINE - time.time())


def _build_graph(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", [])
    n = len(times)
    # preds: list of lists, preds[i] = list of predecessor indices for task i
    # Build successor map and in-degree
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1
    return times, succ, indeg, preds


def _longest_path_to_end(times, succ, n):
    """Compute longest path from each node to any sink (reverse topological)."""
    # Use memoized DFS or topological sort
    # Since we have DAG, do reverse topological order
    # First get topological order
    indeg_tmp = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg_tmp[s] += 1
    
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg_tmp[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg_tmp[v] -= 1
            if indeg_tmp[v] == 0:
                q.append(v)
    
    # longest_path[i] = max(times[j] + longest_path[j] for j in succ[i])
    longest_path = np.zeros(n, dtype=float)
    for i in reversed(topo):
        best = 0.0
        for s in succ[i]:
            if times[s] + longest_path[s] > best:
                best = times[s] + longest_path[s]
        longest_path[i] = times[i] + best
    return longest_path, topo


def _longest_path_from_start(times, preds, n):
    """Compute longest path from any source to each node."""
    indeg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1
    
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    longest_from_start = np.zeros(n, dtype=float)
    for i in topo:
        best = 0.0
        for p in preds[i]:
            if longest_from_start[p] + times[p] > best:
                best = longest_from_start[p] + times[p]
        longest_from_start[i] = best + times[i]
    return longest_from_start, topo


def _greedy_balance(instance, priority_scores):
    """
    Greedy assembly line balancing using the given priority scores.
    Returns number of stations.
    """
    times, succ, indeg, preds = _build_graph(instance)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)
    
    if n == 0:
        return 0
    
    # Track pred_count and eligible tasks
    pred_count = np.array([len(p) for p in preds], dtype=int)
    eligible = []  # min-heap of (-priority, -time, task_index) for tie-breaking
    
    for i in range(n):
        if pred_count[i] == 0:
            # Use (-priority, -time, index) so higher priority first, then larger time
            eligible.append((-priority_scores[i], -times[i], i))
    heapq.heapify(eligible)
    
    # Station assignment
    station_load = 0.0
    num_stations = 1
    assigned_count = 0
    
    while assigned_count < n:
        # Try to fill current station
        while True:
            # Find best eligible task that fits
            best_task = -1
            best_key = None
            candidates = []
            
            while eligible:
                neg_prio, neg_time, task = heapq.heappop(eligible)
                if times[task] <= cycle_time - station_load + 1e-9:
                    # This task fits
                    key = (neg_prio, neg_time, task)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_task = task
                    candidates.append((neg_prio, neg_time, task))
                else:
                    candidates.append((neg_prio, neg_time, task))
            
            # Put all candidates back
            for item in candidates:
                heapq.heappush(eligible, item)
            
            if best_task == -1:
                # No eligible task fits in current station
                if station_load > 0:
                    # Move to next station
                    num_stations += 1
                    station_load = 0.0
                    continue
                else:
                    # Shouldn't happen, break to avoid infinite loop
                    break
            
            # Assign best_task to current station
            station_load += times[best_task]
            assigned_count += 1
            
            # Update successors
            for s in succ[best_task]:
                pred_count[s] -= 1
                if pred_count[s] == 0:
                    heapq.heappush(eligible, (-priority_scores[s], -times[s], s))
            
            # Check if station is full
            if station_load >= cycle_time - 1e-9:
                break
    
    return num_stations


def _compute_priority(instance):
    """
    Compute priority scores for tasks.
    Uses a combination of:
    1. Longest path to end (critical path length) - dominant
    2. Rank (transitive successor count) - Buck's method
    3. Longest path from start
    4. Task time
    5. Tightness: how close task time is to cycle_time
    """
    times, succ, indeg, preds = _build_graph(instance)
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Compute longest path to end
    lp_end, topo = _longest_path_to_end(times, succ, n)
    
    # Compute longest path from start
    lp_start, _ = _longest_path_from_start(times, preds, n)
    
    # Number of successors (direct)
    num_succ = np.array([len(s) for s in succ], dtype=float)
    
    # Total work in subtree (all successors transitively)
    subtree_work = np.zeros(n, dtype=float)
    for i in reversed(topo):
        total = times[i]
        for s in succ[i]:
            total += subtree_work[s]
        subtree_work[i] = total
    
    # Rank: count of all transitive successors (Buck's method)
    # rank[i] = 1 + sum(rank[s] for s in succ[i]) but that double counts
    # Better: rank[i] = number of nodes reachable from i (excluding i)
    # Use topological order: rank[i] = sum of (1 + rank[s]) for s in succ[i]
    # Actually the standard Buck rank is: rank(i) = number of immediate + transitive successors
    # We can compute: for each node, count all nodes reachable from it
    # Efficient way: process in reverse topo, rank[i] = sum over succ of (1 + rank[s])... no that overcounts
    # Correct: reachable[i] = union of {s} U reachable[s] for s in succ[i]
    # For counting: rank[i] = sum_{s in succ[i]} (1 + rank[s]) is wrong due to overlap
    # Simple approximation: use subtree_work / min_time or just count direct+indirect via DP
    # Let's do: rank[i] = number of distinct successors (transitive). 
    # For DAG, we can compute this with bitsets for small n, but n=30 so use sets
    # Actually for n=30, just do BFS/DFS from each node
    rank = np.zeros(n, dtype=float)
    for i in range(n):
        # Count all nodes reachable from i
        visited = set()
        stack = list(succ[i])
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                stack.extend(succ[u])
        rank[i] = len(visited)
    
    # Normalize each component
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_lp_end = normalize(lp_end)
    norm_lp_start = normalize(lp_start)
    norm_rank = normalize(rank)
    norm_subtree = normalize(subtree_work)
    norm_time = normalize(times)
    
    # Tightness: tasks with time close to cycle_time should be placed early
    # to avoid fragmentation. Use (time / cycle_time) as a proxy.
    # Tasks that are "large" relative to cycle time are harder to fit.
    tightness = times / cycle_time if cycle_time > 0 else np.ones(n)
    norm_tightness = normalize(tightness)
    
    # Weighted combination based on ALB literature:
    # - Critical path (lp_end) is the strongest single heuristic
    # - Rank (Buck) is also very strong  
    # - lp_start helps with early placement
    # - subtree_work captures total downstream workload
    # - tightness helps avoid fragmentation
    priority = (
        0.40 * norm_lp_end +
        0.25 * norm_rank +
        0.15 * norm_lp_start +
        0.10 * norm_subtree +
        0.10 * norm_tightness
    )
    
    return priority


def priority(instance):
    """
    Score each assembly task. Higher score = higher priority.
    The greedy algorithm picks the highest-priority eligible task that fits.
    """
    if time.time() > _DEADLINE:
        # Return a simple fallback
        return np.asarray(instance["times"], dtype=float)
    
    try:
        return _compute_priority(instance)
    except Exception:
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
