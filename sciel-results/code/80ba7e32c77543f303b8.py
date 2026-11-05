"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    Uses Simulated Annealing to optimize feature weights, followed by a local polish pass.
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph from predecessors
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute features in reverse topological order
    # 1. Longest path to sink (downstream work)
    dist_down = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist_down[v])
        dist_down[u] = times[u] + max_succ
    
    # 2. Longest path from source (upstream work)
    dist_up = np.zeros(n, dtype=float)
    for u in top_order:
        max_pred = 0.0
        for p in pred[u]:
            max_pred = max(max_pred, dist_up[p])
        dist_up[u] = times[u] + max_pred
    
    # 3. Number of immediate successors
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # 4. Number of immediate predecessors
    n_pred = np.array([len(pred[i]) for i in range(n)], dtype=float)
    
    # 5. Total transitive successors
    total_succ = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0
        for v in succ[u]:
            s += 1 + total_succ[v]
        total_succ[u] = s
    
    # 6. Total transitive predecessors
    total_pred = np.zeros(n, dtype=float)
    for u in top_order:
        s = 0
        for p in pred[u]:
            s += 1 + total_pred[p]
        total_pred[u] = s
    
    # 7. Rank-based position in topological order (earlier = higher priority base)
    top_rank = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        top_rank[u] = idx
    
    # 8. "Slack" or flexibility: tasks with less slack should go first
    #    Slack = cycle_time - dist_up - dist_down + times[u]  (rough)
    slack = cycle_time - dist_up - dist_down + times
    
    # 9. Critical Path Density: ratio of task duration to its longest downstream path
    cpd = np.zeros(n, dtype=float)
    for i in range(n):
        if dist_down[i] > 1e-12:
            cpd[i] = times[i] / dist_down[i]
        else:
            cpd[i] = 0.0
            
    # 10. Critical Path Density x Fanout
    cpd_fanout = cpd * (1.0 + n_succ)
    
    # 11. Critical Ratio: sum of all successor times / own time
    #     High ratio means this task gates a lot of downstream work relative to its duration
    total_downstream_time = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0.0
        for v in succ[u]:
            s += times[v] + total_downstream_time[v]
        total_downstream_time[u] = s
    # Critical ratio: downstream workload per unit of own time
    # Tasks with high ratio are bottlenecks that should be placed early to avoid idle time
    critical_ratio = np.zeros(n, dtype=float)
    for i in range(n):
        if times[i] > 1e-12:
            critical_ratio[i] = total_downstream_time[i] / times[i]
        else:
            critical_ratio[i] = 0.0
    
    # 12. Combined: critical ratio weighted by fanout
    critical_ratio_fanout = critical_ratio * (1.0 + n_succ)
    
    # Normalize features
    def normalize(x):
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Build feature matrix
    # Each feature is normalized to [0,1], higher = more priority
    features = np.column_stack([
        normalize(dist_down),       # 0: downstream path length
        normalize(dist_up),         # 1: upstream path length  
        normalize(times),           # 2: task duration
        normalize(n_succ),          # 3: immediate successors
        normalize(n_pred),          # 4: immediate predecessors
        normalize(total_succ),      # 5: transitive successors
        normalize(total_pred),      # 6: transitive predecessors
        1.0 - normalize(top_rank),  # 7: earlier in topo order
        1.0 - normalize(slack),     # 8: less slack = tighter
        normalize(cpd),             # 9: critical path density
        normalize(cpd_fanout),      # 10: critical path density x fanout
        normalize(critical_ratio),  # 11: critical ratio (downstream workload / own time)
        normalize(critical_ratio_fanout),  # 12: critical ratio x fanout
    ])
    
    n_features = features.shape[1]
    
    # Precompute predecessor sets for fast lookup
    pred_set = [set(pred[i]) for i in range(n)]
    
    # Function to simulate greedy ALB given priority scores
    def simulate(scores):
        """Simulate greedy station filling and return number of stations."""
        remaining_set = set(range(n))
        assigned = np.full(n, -1, dtype=int)
        assigned_set = set()
        stations = 0
        
        while remaining_set:
            # Find eligible tasks: all predecessors assigned
            eligible = []
            for i in remaining_set:
                if pred_set[i].issubset(assigned_set):
                    eligible.append(i)
            
            if not eligible:
                break
            
            # Sort eligible by priority (descending)
            eligible.sort(key=lambda x: -scores[x])
            
            # Fill current station
            station_load = 0.0
            to_remove = set()
            for i in eligible:
                if station_load + times[i] <= cycle_time:
                    assigned[i] = stations
                    assigned_set.add(i)
                    to_remove.add(i)
                    station_load += times[i]
            
            remaining_set -= to_remove
            
            if not remaining_set:
                break
            
            # Check if any tasks were assigned to this station
            if station_load == 0.0:
                # Force assign the highest priority eligible
                best = eligible[0]
                assigned[best] = stations
                assigned_set.add(best)
                remaining_set.discard(best)
            
            stations += 1
        
        return stations

    # Initial weights: emphasize downstream path, duration, critical ratio, and critical path density
    # Feature order: dist_down, dist_up, times, n_succ, n_pred, total_succ, total_pred, top_rank_inv, slack_inv, cpd, cpd_fanout, critical_ratio, critical_ratio_fanout
    weights = np.array([0.20, 0.08, 0.18, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.08, 0.12, 0.06])
    weights = weights / weights.sum()
    
    best_stations = simulate(features @ weights)
    best_weights = weights.copy()
    
    # Simulated Annealing
    np.random.seed(42)
    
    # Parameters - increased iterations since simulate is faster with set-based approach
    T0 = 1.0  # Initial temperature
    T_end = 0.01
    max_iters = 8000
    current_stations = best_stations
    current_weights = best_weights.copy()
    
    for i in range(max_iters):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate neighbor with adaptive noise
        noise_scale = 0.04 * (1.0 - i / max_iters) + 0.01
        new_w = current_weights.copy()
        noise = np.random.normal(0, noise_scale, n_features)
        new_w += noise
        new_w = np.clip(new_w, 0, None)
        if new_w.sum() > 1e-9:
            new_w = new_w / new_w.sum()
        else:
            new_w = current_weights
        
        score_vec = features @ new_w
        new_stations = simulate(score_vec)
        
        # Accept if better or with probability exp(-(delta)/T)
        if new_stations < current_stations:
            current_stations = new_stations
            current_weights = new_w
            if new_stations < best_stations:
                best_stations = new_stations
                best_weights = new_w.copy()
        else:
            delta = new_stations - current_stations
            if delta == 0:
                if np.random.rand() < 0.5:
                    current_weights = new_w
            else:
                T = T0 * (T_end / T0) ** (i / max_iters)
                prob = math.exp(-delta / (T + 1e-9))
                if np.random.rand() < prob:
                    current_stations = new_stations
                    current_weights = new_w
        
    # Local Polish Pass (Coordinate Descent with small steps)
    # This refines the best solution found by SA
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step_size in step_sizes:
        for coord in range(n_features):
            if time.time() - start_time > DEADLINE:
                return features @ best_weights + 1e-8 * times
            
            for delta in [step_size, -step_size]:
                try_w = best_weights.copy()
                try_w[coord] += delta
                try_w = np.clip(try_w, 0, None)
                if try_w.sum() < 1e-9:
                    continue
                try_w = try_w / try_w.sum()
                
                score_vec = features @ try_w
                st = simulate(score_vec)
                
                if st < best_stations:
                    best_stations = st
                    best_weights = try_w.copy()
    
    # Final check: also try some known good heuristics directly
    # Heuristic 1: pure downstream distance
    h1 = normalize(dist_down)
    s1 = simulate(h1)
    if s1 < best_stations:
        best_stations = s1
        final_scores = h1
    else:
        final_scores = features @ best_weights
    
    # Heuristic 2: pure task time
    h2 = normalize(times)
    s2 = simulate(h2)
    if s2 < best_stations:
        best_stations = s2
        final_scores = h2
    
    # Heuristic 3: downstream + time combination
    h3 = 0.6 * normalize(dist_down) + 0.4 * normalize(times)
    s3 = simulate(h3)
    if s3 < best_stations:
        best_stations = s3
        final_scores = h3
    
    # Heuristic 4: inverse rank (BKS-like)
    h4 = 1.0 - normalize(top_rank)
    s4 = simulate(h4)
    if s4 < best_stations:
        best_stations = s4
        final_scores = h4
    
    # Heuristic 5: Critical path density + fanout
    h5 = 0.5 * normalize(cpd) + 0.5 * normalize(cpd_fanout)
    s5 = simulate(h5)
    if s5 < best_stations:
        best_stations = s5
        final_scores = h5
    
    # Heuristic 6: downstream + cpd
    h6 = 0.5 * normalize(dist_down) + 0.5 * normalize(cpd)
    s6 = simulate(h6)
    if s6 < best_stations:
        best_stations = s6
        final_scores = h6
    
    # Heuristic 7: Critical ratio based
    h7 = 0.5 * normalize(critical_ratio) + 0.3 * normalize(dist_down) + 0.2 * normalize(times)
    s7 = simulate(h7)
    if s7 < best_stations:
        best_stations = s7
        final_scores = h7
    
    # Heuristic 8: Critical ratio + fanout + downstream
    h8 = 0.4 * normalize(critical_ratio_fanout) + 0.3 * normalize(dist_down) + 0.3 * normalize(times)
    s8 = simulate(h8)
    if s8 < best_stations:
        best_stations = s8
        final_scores = h8

    # Add tiny tie-breaker
    final_scores = final_scores + 1e-8 * times
    
    return final_scores
# EVOLVE-BLOCK-END
