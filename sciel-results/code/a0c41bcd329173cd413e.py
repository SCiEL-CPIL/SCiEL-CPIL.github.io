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
    DEADLINE = 135  # safety margin before 150s kill
    
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
    #    Use the longest path length as the makespan lower bound
    makespan_lb = max(dist_down)
    slack = makespan_lb - dist_up - dist_down + times

    # New Features:
    # 9. Total time of all predecessors (sum of times of transitive preds)
    total_pred_time = np.zeros(n, dtype=float)
    for u in top_order:
        s = 0.0
        for p in pred[u]:
            s += times[p] + total_pred_time[p]
        total_pred_time[u] = s

    # 10. Total time of all successors (sum of times of transitive succs)
    total_succ_time = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0.0
        for v in succ[u]:
            s += times[v] + total_succ_time[v]
        total_succ_time[u] = s
        
    # 11. Critical Path Indicator (1 if on critical path, 0 otherwise)
    # A node is on critical path if dist_up[u] + dist_down[u] - times[u] == makespan_lb
    on_cp = np.zeros(n, dtype=float)
    for u in range(n):
        if np.isclose(dist_up[u] + dist_down[u] - times[u], makespan_lb):
            on_cp[u] = 1.0

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
        normalize(dist_down),       # downstream path length
        normalize(dist_up),         # upstream path length  
        normalize(times),           # task duration
        normalize(n_succ),          # immediate successors
        normalize(n_pred),          # immediate predecessors
        normalize(total_succ),      # transitive successors
        normalize(total_pred),      # transitive predecessors
        1.0 - normalize(top_rank),  # earlier in topo order
        1.0 - normalize(slack),     # less slack = tighter
        normalize(total_pred_time), # total pred time
        normalize(total_succ_time), # total succ time
        on_cp                       # critical path
    ])
    
    # Function to simulate greedy ALB given priority scores
    def simulate(scores):
        """Simulate greedy station filling and return number of stations."""
        remaining = set(range(n))
        assigned = np.full(n, -1, dtype=int)
        stations = 0
        station_load = 0.0
        
        # For efficiency, maintain a priority-based selection
        # Eligible tasks: all predecessors already assigned to current or earlier stations
        
        while remaining:
            # Find eligible tasks
            eligible = []
            for i in remaining:
                ok = True
                for p in pred[i]:
                    if assigned[p] < 0:
                        ok = False
                        break
                if ok:
                    eligible.append(i)
            
            if not eligible:
                # No eligible tasks - shouldn't happen if valid DAG
                break
            
            # Sort eligible by priority (descending)
            eligible.sort(key=lambda x: -scores[x])
            
            # Fill current station
            station_load = 0.0
            for i in eligible:
                if station_load + times[i] <= cycle_time:
                    assigned[i] = stations
                    station_load += times[i]
                    remaining.discard(i)
            
            if not remaining:
                break
            
            # Check if any tasks were assigned to this station
            if station_load == 0.0:
                # Force assign the highest priority eligible
                best = eligible[0]
                assigned[best] = stations
                remaining.discard(best)
            
            stations += 1
        
        return stations
    
    # Initial weights: emphasize downstream path and duration
    # We have 12 features now
    weights = np.array([0.25, 0.10, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    weights = weights / weights.sum()
    
    best_stations = simulate(features @ weights)
    best_weights = weights.copy()
    
    # Model-and-optimize: local search on weights
    # Use Multi-Start Local Search
    np.random.seed(42)
    n_features = features.shape[1]
    
    # Budget for optimization
    max_evals = 5000
    eval_count = 0
    
    # Define a local search function
    def local_search(w_init, max_iter=20):
        w = w_init.copy()
        best_w = w.copy()
        best_st = simulate(features @ w)
        
        for _ in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            # Try coordinate descent
            improved = False
            for coord in range(n_features):
                for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                    try_w = best_w.copy()
                    try_w[coord] += delta
                    try_w = np.clip(try_w, 0, None)
                    if try_w.sum() < 1e-9:
                        continue
                    try_w = try_w / try_w.sum()
                    
                    score_vec = features @ try_w
                    st = simulate(score_vec)
                    eval_count[0] += 1
                    
                    if st < best_st:
                        best_st = st
                        best_w = try_w.copy()
                        improved = True
                        if time.time() - start_time > DEADLINE:
                            break
                if time.time() - start_time > DEADLINE:
                    break
            if not improved:
                # Try a random perturbation
                noise = np.random.normal(0, 0.1, n_features)
                new_w = best_w + noise
                new_w = np.clip(new_w, 0, None)
                if new_w.sum() > 1e-9:
                    new_w = new_w / new_w.sum()
                    st = simulate(features @ new_w)
                    eval_count[0] += 1
                    if st < best_st:
                        best_st = st
                        best_w = new_w.copy()
        return best_w, best_st

    # Multi-start
    n_starts = 10
    eval_counter = [0]
    
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate random initial weights
        if s == 0:
            w_init = best_weights.copy()
        else:
            w_init = np.random.dirichlet(np.ones(n_features))
            
        w_opt, st_opt = local_search(w_init, max_iter=50)
        
        if st_opt < best_stations:
            best_stations = st_opt
            best_weights = w_opt.copy()
            
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
        
    # Heuristic 4: Critical Path + Downstream
    h4 = 0.5 * on_cp + 0.5 * normalize(dist_down)
    s4 = simulate(h4)
    if s4 < best_stations:
        best_stations = s4
        final_scores = h4

    # Add tiny tie-breaker
    final_scores = final_scores + 1e-8 * times
    
    return final_scores
# EVOLVE-BLOCK-END
