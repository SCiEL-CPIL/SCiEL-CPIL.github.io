"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the ALB greedy assignment.
    
    Uses a combination of:
      1. Longest processing time (primary)
      2. Number of successors (rank position - well-known ALB heuristic)
      3. Weighted sum of successor times (look-ahead)
      4. Small random tie-breaker to explore different orderings
    
    The evaluator fills stations with the highest-priority eligible task that fits.
    Lower mean number of stations is better.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        if preds is not None:
            for p in preds:
                successors[p].append(i)
    
    # Compute number of successors (direct and transitive via topological order)
    # First, get topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        if preds is not None:
            in_degree[i] = len(preds)
    
    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topological sort didn't cover all nodes (cycle), fall back to index order
    if len(topo_order) != n:
        topo_order = list(range(n))
    
    # Compute transitive successor counts and weighted successor times
    # Process in reverse topological order
    succ_count = np.zeros(n, dtype=float)
    succ_weighted_time = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        for succ in successors[node]:
            succ_count[node] += 1 + succ_count[succ]
            succ_weighted_time[node] += times[succ] + succ_weighted_time[succ]
    
    # Direct successor count (immediate)
    direct_succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Normalize components to [0, 1]
    def safe_normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_time = safe_normalize(times)
    norm_succ_count = safe_normalize(succ_count)
    norm_direct_succ = safe_normalize(direct_succ_count)
    norm_succ_weighted = safe_normalize(succ_weighted_time)
    
    # Combine with weights. 
    # Longest processing time is most important for filling stations efficiently.
    # Successor count helps with "rank" - tasks with many successors should go early
    # to not block downstream work.
    # 
    # We use a weighted combination. The key insight: in ALB, we want to place
    # long tasks early to avoid wasting cycle time, but also respect precedence.
    # The greedy algorithm already handles eligibility, so our priority just needs
    # to rank eligible tasks well.
    
    # Primary: processing time (longer = higher priority)
    # Secondary: number of successors (more = higher priority, to unblock downstream)
    # Tertiary: weighted successor time
    
    w_time = 0.6
    w_succ = 0.25
    w_direct = 0.1
    w_weighted = 0.05
    
    scores = (w_time * norm_time + 
              w_succ * norm_succ_count + 
              w_direct * norm_direct_succ + 
              w_weighted * norm_succ_weighted)
    
    # Add a tiny deterministic tie-breaker based on index to ensure stability
    # but also slight variation to help with different instance structures
    tie_break = np.arange(n, dtype=float) / (n + 1) * 0.01
    
    # We want higher priority = assigned first. The evaluator picks highest-priority
    # eligible task. So we return the score where higher = more preferred.
    # But wait - let me re-read: "stations are filled with the highest-priority 
    # eligible task that fits the cycle time."
    
    # So higher score = picked first. We want long tasks to be picked first.
    # Our scores already have that property (norm_time is high for long tasks).
    
    scores = scores + tie_break
    
    return scores
# EVOLVE-BLOCK-END
