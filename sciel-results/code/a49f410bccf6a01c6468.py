"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _simulate_fast(priority_scores, times, pred_sets, cycle_time, n):
    """Fast simulation of greedy ALB."""
    completed = [False] * n
    remaining = set(range(n))
    num_stations = 0
    
    while remaining:
        num_stations += 1
        remaining_time = cycle_time
        
        # Find eligible tasks
        eligible = []
        for t in remaining:
            if all(completed[p] for p in pred_sets[t]):
                eligible.append(t)
        
        if not eligible:
            return num_stations + 1000000
        
        # Sort by priority descending, then time descending
        eligible.sort(key=lambda t: (-priority_scores[t], -times[t], t))
        
        # Greedily assign
        for t in eligible:
            if times[t] <= remaining_time:
                completed[t] = True
                remaining.discard(t)
                remaining_time -= times[t]
    
    return num_stations


def priority(instance):
    """Score each assembly task for the greedy ALB heuristic."""
    start_time = time.time()
    DEADLINE = 130  # Leave margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    pred = instance["predecessors"]
    
    # Build predecessor sets and successor graph
    pred_sets = [set(p) for p in pred]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort
    in_degree = [len(pred[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute features
    # 1. Downstream distance (longest path from task to end)
    dist = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist[v])
        dist[u] = times[u] + max_succ
    
    # 2. Total downstream workload (sum of all successor times)
    total_down = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0.0
        for v in succ[u]:
            s += times[v] + total_down[v]
        total_down[u] = s
    
    # 3. Upstream distance (longest path from start to task)
    up_dist = np.zeros(n, dtype=float)
    for u in top_order:
        max_pred = 0.0
        for p in pred[u]:
            max_pred = max(max_pred, up_dist[p])
        up_dist[u] = max_pred + times[u]
    
    # 4. Total upstream workload
    total_up = np.zeros(n, dtype=float)
    for u in top_order:
        s = 0.0
        for p in pred[u]:
            s += times[p] + total_up[p]
        total_up[u] = s
    
    # 5. Number of immediate successors
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # 6. Number of immediate predecessors
    n_pred = np.array([len(pred[i]) for i in range(n)], dtype=float)
    
    # 7. Total transitive successors
    total_succ_count = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0
        for v in succ[u]:
            s += 1 + total_succ_count[v]
        total_succ_count[u] = s
    
    # 8. Total transitive predecessors
    total_pred_count = np.zeros(n, dtype=float)
    for u in top_order:
        s = 0
        for p in pred[u]:
            s += 1 + total_pred_count[p]
        total_pred_count[u] = s
    
    # 9. Critical ratio: downstream workload / own time
    crit_ratio = np.ones(n, dtype=float)
    for i in range(n):
        if times[i] > 1e-9:
            crit_ratio[i] = total_down[i] / times[i]
        else:
            crit_ratio[i] = 0.0
    
    # 10. Position in topological order (normalized)
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        topo_pos[u] = idx / max(n - 1, 1)
    
    # 11. Sum of times of immediate successors
    succ_time_sum = np.zeros(n, dtype=float)
    for u in range(n):
        for v in succ[u]:
            succ_time_sum[u] += times[v]
    
    # 12. Sum of times of immediate predecessors
    pred_time_sum = np.zeros(n, dtype=float)
    for u in range(n):
        for p in pred[u]:
            pred_time_sum[u] += times[p]
    
    # 13. Bottleneck indicator: time * downstream distance
    bottleneck = times * dist
    
    # 14. Time ratio
    time_ratio = times / max(cycle_time, 1e-9)
    
    # 15. Slack: cycle_time - time (how much room left)
    slack = cycle_time - times
    
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Build feature matrix
    feats = np.column_stack([
        normalize(dist),
        normalize(times),
        normalize(total_down),
        normalize(up_dist),
        normalize(total_up),
        normalize(n_succ),
        normalize(n_pred),
        normalize(total_succ_count),
        normalize(total_pred_count),
        normalize(crit_ratio),
        normalize(topo_pos),
        normalize(succ_time_sum),
        normalize(pred_time_sum),
        normalize(bottleneck),
        normalize(time_ratio),
        normalize(slack),
    ])
    
    d = feats.shape[1]
    
    def simulate_with_weights(w):
        scores = feats @ w
        return _simulate_fast(scores, times, pred_sets, cycle_time, n)
    
    # Candidate weight vectors
    candidates = []
    
    # 1. Pure downstream distance
    w = np.zeros(d); w[0] = 1.0
    candidates.append(w)
    
    # 2. Pure times
    w = np.zeros(d); w[1] = 1.0
    candidates.append(w)
    
    # 3. Dist + times
    w = np.zeros(d); w[0] = 0.5; w[1] = 0.5
    candidates.append(w)
    
    # 4. Critical ratio focused
    w = np.zeros(d); w[9] = 0.6; w[0] = 0.4
    candidates.append(w)
    
    # 5. Total downstream + crit ratio
    w = np.zeros(d); w[2] = 0.5; w[9] = 0.5
    candidates.append(w)
    
    # 6. Balanced: dist, times, crit_ratio
    w = np.zeros(d); w[0] = 0.4; w[1] = 0.3; w[9] = 0.3
    candidates.append(w)
    
    # 7. Fan-out + downstream
    w = np.zeros(d); w[5] = 0.3; w[0] = 0.4; w[9] = 0.3
    candidates.append(w)
    
    # 8. Upstream + downstream
    w = np.zeros(d); w[0] = 0.4; w[3] = 0.3; w[1] = 0.3
    candidates.append(w)
    
    # 9. All equal
    w = np.ones(d) / d
    candidates.append(w)
    
    # 10. Heavy downstream
    w = np.zeros(d); w[0] = 0.5; w[2] = 0.3; w[1] = 0.2
    candidates.append(w)
    
    # 11. Critical ratio + bottleneck
    w = np.zeros(d); w[9] = 0.5; w[13] = 0.3; w[0] = 0.2
    candidates.append(w)
    
    # 12. Total downstream + total up
    w = np.zeros(d); w[2] = 0.4; w[4] = 0.3; w[0] = 0.3
    candidates.append(w)
    
    # 13. Slack + times
    w = np.zeros(d); w[15] = 0.5; w[1] = 0.5
    candidates.append(w)
    
    # 14. Pred count + succ count + dist
    w = np.zeros(d); w[6] = 0.3; w[5] = 0.3; w[0] = 0.4
    candidates.append(w)
    
    # 15. Total succ count + crit ratio
    w = np.zeros(d); w[7] = 0.4; w[9] = 0.6
    candidates.append(w)
    
    # Evaluate all candidates
    best_stations = float('inf')
    best_w = candidates[0].copy()
    for w in candidates:
        st = simulate_with_weights(w)
        if st < best_stations:
            best_stations = st
            best_w = w.copy()
    
    # Deadline check
    if time.time() - start_time > DEADLINE - 20:
        scores = feats @ best_w
        return scores
    
    # Simulated Annealing
    best_stations_local = best_stations
    best_w_local = best_w.copy()
    rng = np.random.RandomState(42)
    
    T_init = 0.2
    T_min = 0.005
    cooling_rate = 0.993
    T = T_init
    sa_iterations = 0
    max_sa_iterations = 5000
    
    while T > T_min and sa_iterations < max_sa_iterations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Generate neighbor
        w_new = best_w_local + rng.normal(0, T * 0.4, size=d)
        # Keep weights in reasonable range
        w_new = np.clip(w_new, -0.5, 2.0)
        
        st_new = simulate_with_weights(w_new)
        st_best = best_stations_local
        
        if st_new <= st_best:
            best_stations_local = st_new
            best_w_local = w_new.copy()
        else:
            delta = st_new - st_best
            if delta < 1.0:
                if rng.random() < np.exp(-delta / max(T, 1e-9)):
                    best_w_local = w_new.copy()
        
        T *= cooling_rate
        sa_iterations += 1
    
    # Coordinate descent polish
    if time.time() - start_time < DEADLINE - 5:
        for _ in range(3):  # Multiple passes
            for i in range(d):
                if time.time() - start_time > DEADLINE - 5:
                    break
                for delta in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
                    w_new = best_w_local.copy()
                    w_new[i] += delta
                    st = simulate_with_weights(w_new)
                    if st < best_stations_local:
                        best_stations_local = st
                        best_w_local = w_new.copy()
    
    # Final scores with tie-breaker
    scores = feats @ best_w_local
    scores = scores + 1e-8 * times
    
    return scores
# EVOLVE-BLOCK-END
