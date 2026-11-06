# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _topo_order(n, pred_lists):
    """Return topological order and successor adjacency."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order, succ


def _downstream_stats(n, order, succ, times):
    """Compute longest_down and total_descendant_time via reverse topo."""
    ld = [0.0] * n
    td = [0.0] * n
    for u in reversed(order):
        best_ld = 0.0
        tot = 0.0
        for v in succ[u]:
            if times[v] + ld[v] > best_ld:
                best_ld = times[v] + ld[v]
            tot += times[v] + td[v]
        ld[u] = best_ld
        td[u] = tot
    return ld, td


def _greedy_stations(n, times, pred_lists, succ, cycle_time, priority_vec):
    """
    Simulate the greedy station-filling.
    Returns number of stations.
    """
    done = [False] * n
    remaining_count = n
    stations = 0
    pred_count = [len(pred_lists[i]) for i in range(n)]
    pred_done = [0] * n
    
    while remaining_count > 0:
        stations += 1
        cap = cycle_time
        
        eligible = []
        for t in range(n):
            if done[t]:
                continue
            if pred_done[t] == pred_count[t] and times[t] <= cap:
                eligible.append(t)
        
        if not eligible:
            return 1000
        
        eligible.sort(key=lambda t: (-priority_vec[t], -times[t], t))
        
        assigned_any = False
        for t in eligible:
            if times[t] <= cap:
                cap -= times[t]
                done[t] = True
                remaining_count -= 1
                for v in succ[t]:
                    pred_done[v] += 1
                assigned_any = True
        
        if not assigned_any:
            return 1000
    
    return stations


def _transitive_time_sum(n, order, succ, pred_lists, times):
    """
    Compute for each task:
      - total time of all transitive predecessors (upstream_time)
      - total time of all transitive successors (downstream_time)
    via reverse topological accumulation.
    """
    # downstream_time[u] = sum of times[v] for all v in transitive successors of u
    # Compute by processing in reverse topological order
    downstream_time = np.zeros(n, dtype=float)
    for u in reversed(order):
        for v in succ[u]:
            downstream_time[u] += times[v] + downstream_time[v]
    
    # upstream_time[u] = sum of times[p] for all p in transitive predecessors of u
    # Compute by processing in topological order
    upstream_time = np.zeros(n, dtype=float)
    for u in order:
        for v in succ[u]:
            upstream_time[v] += times[u] + upstream_time[u]
    
    return upstream_time, downstream_time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    pred_lists = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)

    t0 = time.time()
    DEADLINE = 135.0

    order, succ = _topo_order(n, pred_lists)
    ld, td = _downstream_stats(n, order, succ, times)

    lu = [0.0] * n
    for u in order:
        for v in succ[u]:
            c = lu[u] + times[u]
            if c > lu[v]:
                lu[v] = c

    cp_len = max(lu[i] + times[i] + ld[i] for i in range(n)) + 1e-9
    max_t = max(times) + 1e-9
    max_ld = max(ld) + 1e-9 if max(ld) > 0 else 1.0
    max_td = max(td) + 1e-9 if max(td) > 0 else 1.0
    max_lu = max(lu) + 1e-9 if max(lu) > 0 else 1.0
    max_succ = max([len(succ[i]) for i in range(n)]) + 1e-9
    max_pred = max([len(pred_lists[i]) for i in range(n)]) + 1e-9

    lu_a = np.array(lu, dtype=float)
    ld_a = np.array(ld, dtype=float)
    td_a = np.array(td, dtype=float)
    succ_a = np.array([len(succ[i]) for i in range(n)], dtype=float)
    pred_a = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    cp_a = (lu_a + times + ld_a) / cp_len

    trans_succ = np.zeros(n, dtype=float)
    for u in reversed(order):
        for v in succ[u]:
            trans_succ[u] += 1 + trans_succ[v]
    max_trans = max(trans_succ) + 1e-9 if max(trans_succ) > 0 else 1.0

    # NEW: Compute transitive time sums for resource contention metric
    upstream_time, downstream_time = _transitive_time_sum(n, order, succ, pred_lists, times)
    
    # Normalize transitive time sums
    max_upstream = max(upstream_time) + 1e-9 if max(upstream_time) > 0 else 1.0
    max_downstream = max(downstream_time) + 1e-9 if max(downstream_time) > 0 else 1.0
    norm_upstream = upstream_time / max_upstream
    norm_downstream = downstream_time / max_downstream
    
    # Resource contention: sum of cycle-time-normalized durations of transitive predecessors and successors
    # This identifies tasks in time-dense, interconnected subgraphs
    resource_contention = norm_upstream + norm_downstream
    max_rc = max(resource_contention) + 1e-9 if max(resource_contention) > 0 else 1.0
    norm_rc = resource_contention / max_rc

    norm_succ = succ_a / max_succ
    norm_time = times / max_t
    norm_ld = ld_a / max_ld
    norm_td = td_a / max_td
    norm_lu = lu_a / max_lu

    slack_a = cp_len - (lu_a + times + ld_a)
    min_slack = min(slack_a)
    max_slack = max(slack_a)
    range_slack = max_slack - min_slack + 1e-9
    norm_slack = (slack_a - min_slack) / range_slack

    candidates = []
    # Classic heuristics
    candidates.append(times.copy())
    candidates.append(norm_succ * 0.6 + norm_time * 0.4)
    candidates.append(norm_time * 0.5 + norm_ld * 0.5)
    candidates.append(norm_time * 0.4 + cp_a * 0.4 + norm_ld * 0.2)
    candidates.append(norm_ld * 0.5 + norm_td * 0.3 + norm_time * 0.2)
    candidates.append(norm_succ * 0.3 + norm_time * 0.3 + cp_a * 0.2 + norm_ld * 0.2)
    candidates.append((1.0 - pred_a / max_pred) * 0.3 + norm_time * 0.4 + cp_a * 0.3)
    
    # Rank-based
    combined = norm_ld * 0.6 + norm_time * 0.4
    idx = np.argsort(-combined)
    ranks = np.zeros(n)
    for r, t in enumerate(idx):
        ranks[t] = r
    max_r = max(n - 1, 1)
    candidates.append(1.0 - (ranks / max_r) ** 0.7)
    
    candidates.append(norm_time * cp_a * 0.5 + norm_time * 0.3 + norm_ld * 0.2)
    candidates.append(norm_td * 0.4 + norm_time * 0.3 + cp_a * 0.3)
    candidates.append(cp_a.copy())
    candidates.append((1.0 - norm_slack) * 0.5 + norm_time * 0.5)
    candidates.append(norm_lu * 0.5 + norm_time * 0.5)
    candidates.append(norm_time * 0.3 + norm_ld * 0.3 + norm_succ * 0.2 + cp_a * 0.2)
    candidates.append((trans_succ / max_trans) * 0.5 + norm_time * 0.3 + cp_a * 0.2)
    candidates.append(norm_time * norm_succ * 0.4 + norm_time * 0.3 + norm_ld * 0.3)
    candidates.append((1.0 - norm_slack) * 0.4 + norm_time * 0.4 + norm_succ * 0.2)
    
    cp_pos = lu_a / cp_len
    candidates.append(cp_pos * 0.3 + (1 - cp_pos) * 0.2 + norm_time * 0.5)
    
    # Longest path first (LPF)
    sort_key = lu_a + times + ld_a
    idx_lpf = np.argsort(-sort_key)
    ranks_lpf = np.zeros(n)
    for r, t in enumerate(idx_lpf):
        ranks_lpf[t] = r
    max_r_lpf = max(n - 1, 1)
    candidates.append(1.0 - (ranks_lpf / max_r_lpf) ** 0.8)
    
    # Modified position weight
    sort_key2 = ld_a * 1000 + times
    idx19 = np.argsort(-sort_key2)
    ranks19 = np.zeros(n)
    for r, t in enumerate(idx19):
        ranks19[t] = r
    max_r19 = max(n - 1, 1)
    candidates.append(1.0 - (ranks19 / max_r19) ** 0.5)
    
    candidates.append((times ** 2 / (max_t ** 2)) * 0.5 + norm_succ * 0.3 + cp_a * 0.2)
    
    # New: prioritize tasks that are bottlenecks (high time, many successors)
    candidates.append(norm_time * 0.4 + norm_succ * 0.4 + norm_ld * 0.2)
    
    # New: prioritize tasks early in critical path with high downstream
    candidates.append(cp_a * 0.3 + norm_td * 0.3 + norm_time * 0.2 + norm_succ * 0.2)
    
    # New: balance between time and position
    pos_in_cp = lu_a / cp_len
    candidates.append((1.0 - pos_in_cp) * 0.3 + norm_time * 0.4 + norm_ld * 0.3)
    
    # New: successor-weighted time
    succ_w_time = times * (1 + succ_a / max_succ)
    max_swt = max(succ_w_time) + 1e-9
    candidates.append((succ_w_time / max_swt) * 0.5 + norm_ld * 0.3 + cp_a * 0.2)
    
    # New: inverse slack (tasks with less slack are more critical)
    candidates.append((1.0 - norm_slack) * 0.6 + norm_time * 0.2 + norm_succ * 0.2)
    
    # New: combined depth and width
    depth_norm = lu_a / max_lu
    candidates.append(depth_norm * 0.3 + norm_succ * 0.3 + norm_time * 0.2 + norm_ld * 0.2)
    
    # New: rank by (time + ld)
    sort_key3 = times + ld_a
    idx3 = np.argsort(-sort_key3)
    ranks3 = np.zeros(n)
    for r, t in enumerate(idx3):
        ranks3[t] = r
    max_r3 = max(n - 1, 1)
    candidates.append(1.0 - (ranks3 / max_r3) ** 0.6)
    
    # New: priority based on number of immediate successors (bottleneck tasks)
    candidates.append(norm_succ * 0.5 + norm_time * 0.3 + (1.0 - norm_slack) * 0.2)
    
    # New: geometric mean of time and criticality
    geo_mean = np.sqrt(norm_time * cp_a)
    candidates.append(geo_mean * 0.5 + norm_ld * 0.3 + norm_succ * 0.2)
    
    # NEW: Resource contention based candidates
    # Tasks in time-dense interconnected subgraphs should be prioritized to maximize parallelism
    candidates.append(norm_rc * 0.5 + norm_time * 0.3 + norm_ld * 0.2)
    candidates.append(norm_rc * 0.4 + norm_time * 0.4 + cp_a * 0.2)
    candidates.append(norm_rc * 0.3 + norm_upstream * 0.2 + norm_downstream * 0.2 + norm_time * 0.3)
    candidates.append(norm_rc * 0.6 + norm_succ * 0.2 + norm_time * 0.2)
    candidates.append((norm_rc * 0.5 + norm_ld * 0.3) * (1 + norm_time * 0.5))
    candidates.append(norm_rc * 0.4 + (1.0 - norm_slack) * 0.3 + norm_time * 0.3)
    
    # Hybrid: combine resource contention with critical path position
    pos_in_cp2 = lu_a / cp_len
    candidates.append(norm_rc * 0.4 + (1.0 - pos_in_cp2) * 0.3 + norm_time * 0.3)
    
    # Geometric mean of resource contention and time
    geo_rc = np.sqrt(norm_rc * norm_time)
    candidates.append(geo_rc * 0.5 + norm_ld * 0.3 + norm_succ * 0.2)
    
    # Rank-based resource contention
    sort_key_rc = norm_rc * 1000 + norm_time
    idx_rc = np.argsort(-sort_key_rc)
    ranks_rc = np.zeros(n)
    for r, t in enumerate(idx_rc):
        ranks_rc[t] = r
    max_r_rc = max(n - 1, 1)
    candidates.append(1.0 - (ranks_rc / max_r_rc) ** 0.7)

    best_priority = candidates[0]
    best_stations = _greedy_stations(n, times, pred_lists, succ, cycle_time, candidates[0])

    for c in candidates[1:]:
        if time.time() - t0 > DEADLINE:
            break
        s = _greedy_stations(n, times, pred_lists, succ, cycle_time, c)
        if s < best_stations:
            best_stations = s
            best_priority = c

    # Local search with adaptive step size
    if time.time() - t0 < DEADLINE - 5:
        best_p = best_priority.copy()
        best_s = best_stations
        rng = np.random.default_rng(42)
        step_size = 0.05
        no_improve = 0
        max_iter = 200
        
        for it in range(max_iter):
            if time.time() - t0 > DEADLINE - 2:
                break
            if no_improve > 30:
                step_size *= 0.7
                no_improve = 0
                if step_size < 0.001:
                    break
            
            noise = rng.normal(0, step_size, n)
            perturbed = best_p + noise
            s = _greedy_stations(n, times, pred_lists, succ, cycle_time, perturbed)
            if s < best_s:
                best_s = s
                best_p = perturbed
                no_improve = 0
            else:
                no_improve += 1
        
        if best_s < best_stations:
            best_priority = best_p
            best_stations = best_s

    # Try a few different random seeds for perturbation around best
    if time.time() - t0 < DEADLINE - 3:
        for seed in [1, 7, 123, 456, 789]:
            if time.time() - t0 > DEADLINE - 1:
                break
            rng2 = np.random.default_rng(seed)
            for _ in range(40):
                if time.time() - t0 > DEADLINE - 1:
                    break
                noise = rng2.normal(0, 0.03, n)
                perturbed = best_priority + noise
                s = _greedy_stations(n, times, pred_lists, succ, cycle_time, perturbed)
                if s < best_stations:
                    best_stations = s
                    best_priority = perturbed
    
    # Final polish: try blending the best priority with resource contention
    if time.time() - t0 < DEADLINE - 1:
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if time.time() - t0 > DEADLINE - 1:
                break
            blended = (1 - alpha) * best_priority + alpha * norm_rc
            s = _greedy_stations(n, times, pred_lists, succ, cycle_time, blended)
            if s < best_stations:
                best_stations = s
                best_priority = blended

    return best_priority
# EVOLVE-BLOCK-END
