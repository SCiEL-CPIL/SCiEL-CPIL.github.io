"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque


def _simulate_greedy(times, preds, cycle_time, priority_scores):
    """
    Simulate the greedy station filling with given priority scores.
    Returns the number of stations used.
    """
    n = len(times)
    if n == 0:
        return 0

    # Precompute successors
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)

    remaining_preds = [len(preds[i]) for i in range(n)]
    eligible_set = set(i for i in range(n) if remaining_preds[i] == 0)

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
            # Iterate over eligible tasks
            # For n=30, this is fast
            for i in eligible_set:
                if times[i] <= cycle_time - load + 1e-9:
                    if priority_scores[i] > best_prio:
                        best_prio = priority_scores[i]
                        best_task = i
            
            if best_task == -1:
                break
            
            # Assign best_task
            load += times[best_task]
            remaining -= 1
            eligible_set.remove(best_task)
            
            # Update successors
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible_set.add(s)

    return stations


def priority(instance):
    """
    Score each assembly task using a refined Critical Path and Weighted Rank approach
    with extensive local search over weight parameters.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    predecessors = instance.get("predecessors", [])

    if n == 0:
        return np.array([])

    # Build adjacency: successors[i] = list of tasks that depend on i
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)

    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort
    topo_order = []
    in_deg = in_degree.copy()
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    # If not all nodes are in topo_order, there's a cycle; fall back to simple ordering
    if len(topo_order) < n:
        return times.copy()

    # Compute longest downstream path for each node (reverse topological order)
    longest_downstream = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            if longest_downstream[succ] > max_succ:
                max_succ = longest_downstream[succ]
        longest_downstream[node] = times[node] + max_succ

    # Compute Weighted Rank: Sum of times of all transitive successors
    # We can compute this by processing in reverse topological order
    # weighted_rank[i] = sum(times[s] + weighted_rank[s] for s in successors[i])? 
    # No, that double counts. 
    # Weighted Rank is typically: Sum of durations of all successors (immediate and transitive).
    # Let's compute transitive successors count and sum of times.
    
    # For n=30, we can compute transitive closure easily or just iterate.
    # Let's compute "total downstream time" (sum of times of all nodes reachable from i)
    total_downstream_time = np.zeros(n, dtype=float)
    total_downstream_count = np.zeros(n, dtype=int)
    
    # Reverse topo order
    for node in reversed(topo_order):
        max_td_t = 0.0
        max_td_c = 0
        # This is not quite right for "sum of all successors" if there are merges.
        # We need the sum of times of ALL nodes in the subgraph reachable from node.
        # Since it's a DAG, we can sum the values of successors, but if a successor is reachable via two paths, 
        # standard "rank" counts it once. "Weighted rank" in ALB usually refers to the sum of times of immediate successors 
        # or the longest path. 
        # Let's stick to features that are robust:
        # 1. Longest Path (LDP)
        # 2. Immediate Fan-out
        # 3. Immediate Fan-in
        # 4. Time
        # 5. "Potential": How much it constrains the rest.
        
        pass # We will use simpler features to avoid overfitting to a specific definition of WR

    # Let's use these features:
    # 1. LDP (Longest Downstream Path)
    # 2. Time
    # 3. Fan-out (immediate successors)
    # 4. Fan-in (immediate predecessors)
    # 5. Upstream Path (Longest path from start to node)
    
    fan_out = np.array([len(successors[i]) for i in range(n)], dtype=float)
    fan_in = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)
    
    longest_upstream = np.zeros(n, dtype=float)
    for node in topo_order:
        max_pred = 0.0
        for p in predecessors[node]:
            if longest_upstream[p] + times[p] > max_pred:
                max_pred = longest_upstream[p] + times[p]
        longest_upstream[node] = max_pred

    # Normalize components
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_ld = normalize(longest_downstream)
    norm_time = normalize(times)
    norm_fanout = normalize(fan_out)
    norm_fanin = normalize(fan_in)
    norm_upstream = normalize(longest_upstream)
    
    # Add a "Balance" feature: How central is the task?
    # Centrality = (upstream + downstream) / total_path_length?
    total_path = np.max(longest_downstream) # Approx
    centrality = (longest_upstream + longest_downstream) / (total_path + 1e-9)
    norm_central = normalize(centrality)

    # Base priority score
    # We want to prioritize tasks that are on the critical path AND have high fan-out
    # Initial weights
    w_ld, w_time, w_fanout, w_fanin, w_upstream, w_central = 1.0, 0.5, 0.3, 0.2, 0.0, 0.1
    base_score = (w_ld * norm_ld + w_time * norm_time + w_fanout * norm_fanout + 
                  w_fanin * norm_fanin + w_upstream * norm_upstream + w_central * norm_central)

    # Evaluate base score
    best_score = base_score.copy()
    best_stations = _simulate_greedy(times, predecessors, cycle_time, base_score)

    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_score

    # Local search over weight parameters
    # We will use a random search with a larger number of iterations if time permits
    rng = np.random.RandomState(42)
    
    # Define a grid of "base" weight vectors to explore
    # Focus on LDP and Time as primary drivers
    weight_combos = [
        (1.0, 0.5, 0.3, 0.2, 0.0, 0.1),
        (1.0, 0.3, 0.5, 0.2, 0.0, 0.1),
        (1.0, 0.5, 0.5, 0.0, 0.0, 0.1),
        (1.0, 0.7, 0.3, 0.0, 0.0, 0.1),
        (1.0, 0.5, 0.3, 0.0, 0.3, 0.1),
        (1.0, 0.5, 0.3, 0.0, 0.5, 0.1),
        (1.2, 0.5, 0.3, 0.2, 0.0, 0.1),
        (0.8, 0.5, 0.5, 0.2, 0.0, 0.1),
        (1.0, 1.0, 0.5, 0.0, 0.0, 0.1),
        (1.0, 0.5, 0.3, 0.5, 0.0, 0.1),
        (1.5, 0.5, 0.0, 0.0, 0.0, 0.0), # Pure LDP
        (0.5, 1.5, 0.0, 0.0, 0.0, 0.0), # Pure Time
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0), # LDP + Fanout
    ]

    # Add random perturbations around the best found so far
    for _ in range(50):
        w_ld_r = rng.uniform(0.5, 2.0)
        w_time_r = rng.uniform(0.0, 1.5)
        w_fanout_r = rng.uniform(0.0, 1.0)
        w_fanin_r = rng.uniform(0.0, 0.5)
        w_upstream_r = rng.uniform(0.0, 0.5)
        w_central_r = rng.uniform(0.0, 0.5)
        weight_combos.append((w_ld_r, w_time_r, w_fanout_r, w_fanin_r, w_upstream_r, w_central_r))

    for w_ld, w_time, w_fanout, w_fanin, w_upstream, w_central in weight_combos:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break

        score = (w_ld * norm_ld + w_time * norm_time + w_fanout * norm_fanout + 
                 w_fanin * norm_fanin + w_upstream * norm_upstream + w_central * norm_central)
        
        # Optional: Add a small random noise to break ties
        # score += rng.uniform(-0.01, 0.01, size=n)
        
        stations = _simulate_greedy(times, predecessors, cycle_time, score)

        if stations < best_stations:
            best_stations = stations
            best_score = score.copy()
            
            # If we find a new best, we can do a few local refinements around it
            # This is a simple hill-climbing step
            for _ in range(5):
                if time.time() - start_time > DEADLINE - 5:
                    break
                delta = rng.uniform(-0.2, 0.2, size=6)
                new_w = (w_ld + delta[0], w_time + delta[1], w_fanout + delta[2], 
                         w_fanin + delta[3], w_upstream + delta[4], w_central + delta[5])
                # Ensure weights are not too negative (though they can be, it's just a linear combo)
                new_score = (new_w[0] * norm_ld + new_w[1] * norm_time + new_w[2] * norm_fanout + 
                             new_w[3] * norm_fanin + new_w[4] * norm_upstream + new_w[5] * norm_central)
                new_stations = _simulate_greedy(times, predecessors, cycle_time, new_score)
                if new_stations < best_stations:
                    best_stations = new_stations
                    best_score = new_score.copy()
                    w_ld, w_time, w_fanout, w_fanin, w_upstream, w_central = new_w

    return best_score
# EVOLVE-BLOCK-END
