"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    Uses a model-and-optimize approach: compute multiple feature components,
    then optimize weights via a lightweight local search on the generated instance
    to minimize the number of stations produced by the greedy heuristic.
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
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
    slack = cycle_time - dist_up - dist_down + times
    
    # 9. Critical Path Density: ratio of task duration to its longest downstream path
    cpd = np.zeros(n, dtype=float)
    for i in range(n):
        if dist_down[i] > 1e-12:
            cpd[i] = times[i] / dist_down[i]
        else:
            cpd[i] = 0.0
    
    # 10. Critical Path Density × Fanout
    cpd_fanout = cpd * (1.0 + n_succ)
    
    # 11. Effective downstream load: sum of all downstream task times
    eff_down = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = times[u]
        for v in succ[u]:
            s += eff_down[v]
        eff_down[u] = s
    
    # 12. Critical path membership (soft)
    max_dp = dist_down.max()
    cp_member = np.zeros(n, dtype=float)
    for i in range(n):
        path_len = dist_up[i] + dist_down[i] - times[i]
        if abs(path_len - max_dp) < 1e-6:
            cp_member[i] = 1.0
        else:
            cp_member[i] = max(0.0, 1.0 - (max_dp - path_len) / max(max_dp, 1e-9))
    
    # Normalize features
    def normalize(x):
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Build feature matrix
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
        normalize(cpd_fanout),      # 10: critical path density × fanout
        normalize(eff_down),        # 11: effective downstream load
        cp_member,                  # 12: critical path membership
    ])
    
    # Optimized simulate: maintain ready list incrementally
    def simulate(scores):
        """Simulate greedy station filling and return number of stations."""
        assigned = np.full(n, -1, dtype=int)
        remaining_count = n
        stations = 0
        
        pending_preds = np.array([len(pred[i]) for i in range(n)], dtype=int)
        ready = [i for i in range(n) if pending_preds[i] == 0]
        
        while remaining_count > 0:
            ready.sort(key=lambda x: -scores[x])
            
            station_load = 0.0
            newly_assigned = []
            still_ready = []
            
            for i in ready:
                if station_load + times[i] <= cycle_time:
                    newly_assigned.append(i)
                    station_load += times[i]
                else:
                    still_ready.append(i)
            
            if not newly_assigned:
                best = ready[0]
                newly_assigned = [best]
                still_ready = ready[1:]
            
            for i in newly_assigned:
                assigned[i] = stations
                remaining_count -= 1
                for v in succ[i]:
                    pending_preds[v] -= 1
                    if pending_preds[v] == 0:
                        still_ready.append(v)
            
            ready = still_ready
            stations += 1
        
        return stations
    
    # Initial weights
    weights = np.array([0.20, 0.08, 0.18, 0.04, 0.04, 0.04, 0.04, 0.04, 0.06, 0.08, 0.08, 0.08, 0.04])
    weights = weights / weights.sum()
    
    best_stations = simulate(features @ weights)
    best_weights = weights.copy()
    
    # Model-and-optimize: local search on weights
    np.random.seed(42)
    n_features = features.shape[1]
    
    max_evals = 5000
    eval_count = 0
    
    for iteration in range(max_evals):
        if time.time() - start_time > DEADLINE:
            break
        
        noise_scale = 0.03 if best_stations <= 8 else 0.05
        new_w = best_weights.copy()
        noise = np.random.normal(0, noise_scale, n_features)
        new_w += noise
        new_w = np.clip(new_w, 0, None)
        if new_w.sum() > 1e-9:
            new_w = new_w / new_w.sum()
        
        score_vec = features @ new_w
        stations = simulate(score_vec)
        eval_count += 1
        
        if stations < best_stations:
            best_stations = stations
            best_weights = new_w.copy()
            
            for coord in range(n_features):
                if time.time() - start_time > DEADLINE:
                    break
                for delta in [0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                    try_w = best_weights.copy()
                    try_w[coord] += delta
                    try_w = np.clip(try_w, 0, None)
                    if try_w.sum() < 1e-9:
                        continue
                    try_w = try_w / try_w.sum()
                    
                    score_vec = features @ try_w
                    st = simulate(score_vec)
                    eval_count += 1
                    
                    if st < best_stations:
                        best_stations = st
                        best_weights = try_w.copy()
    
    # Simulated Annealing polish pass
    if time.time() - start_time < DEADLINE - 15:
        sa_best_stations = best_stations
        sa_best_weights = best_weights.copy()
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        sa_iters = 0
        sa_max_iters = 4000
        
        while T > T_min and sa_iters < sa_max_iters:
            if time.time() - start_time > DEADLINE - 10:
                break
            sa_iters += 1
            
            noise_scale = 0.02 + 0.08 * T
            cand_w = sa_best_weights.copy()
            noise = np.random.normal(0, noise_scale, n_features)
            cand_w += noise
            cand_w = np.clip(cand_w, 0, None)
            if cand_w.sum() < 1e-9:
                continue
            cand_w = cand_w / cand_w.sum()
            
            score_vec = features @ cand_w
            cand_stations = simulate(score_vec)
            
            delta = cand_stations - sa_best_stations
            if delta < 0 or (delta > 0 and np.random.random() < np.exp(-delta / max(T, 1e-9))):
                sa_best_stations = cand_stations
                sa_best_weights = cand_w.copy()
            
            T *= cooling
        
        if sa_best_stations < best_stations:
            best_stations = sa_best_stations
            best_weights = sa_best_weights
    
    # Fine-grained coordinate descent polish pass
    if time.time() - start_time < DEADLINE - 5:
        polish_weights = best_weights.copy()
        polish_stations = best_stations
        fine_deltas = [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]
        
        improved = True
        while improved and time.time() - start_time < DEADLINE - 3:
            improved = False
            for coord in range(n_features):
                if time.time() - start_time > DEADLINE - 3:
                    break
                for d in fine_deltas:
                    try_w = polish_weights.copy()
                    try_w[coord] += d
                    try_w = np.clip(try_w, 0, None)
                    if try_w.sum() < 1e-9:
                        continue
                    try_w = try_w / try_w.sum()
                    
                    score_vec = features @ try_w
                    st = simulate(score_vec)
                    
                    if st < polish_stations:
                        polish_stations = st
                        polish_weights = try_w.copy()
                        improved = True
                        break
        
        if polish_stations < best_stations:
            best_stations = polish_stations
            best_weights = polish_weights
    
    # Final check: known good heuristics
    final_scores = features @ best_weights
    
    h1 = normalize(dist_down)
    s1 = simulate(h1)
    if s1 < best_stations:
        best_stations = s1
        final_scores = h1
    
    h2 = normalize(times)
    s2 = simulate(h2)
    if s2 < best_stations:
        best_stations = s2
        final_scores = h2
    
    h3 = 0.6 * normalize(dist_down) + 0.4 * normalize(times)
    s3 = simulate(h3)
    if s3 < best_stations:
        best_stations = s3
        final_scores = h3
    
    h4 = 1.0 - normalize(top_rank)
    s4 = simulate(h4)
    if s4 < best_stations:
        best_stations = s4
        final_scores = h4
    
    h5 = 0.5 * normalize(cpd) + 0.3 * normalize(dist_down) + 0.2 * normalize(cpd_fanout)
    s5 = simulate(h5)
    if s5 < best_stations:
        best_stations = s5
        final_scores = h5
    
    h6 = 0.4 * normalize(cpd_fanout) + 0.3 * normalize(dist_down) + 0.2 * normalize(times) + 0.1 * (1.0 - normalize(slack))
    s6 = simulate(h6)
    if s6 < best_stations:
        best_stations = s6
        final_scores = h6
    
    h7 = 0.5 * cp_member + 0.3 * normalize(dist_down) + 0.2 * normalize(eff_down)
    s7 = simulate(h7)
    if s7 < best_stations:
        best_stations = s7
        final_scores = h7
    
    final_scores = final_scores + 1e-8 * times
    
    return final_scores
# EVOLVE-BLOCK-END
