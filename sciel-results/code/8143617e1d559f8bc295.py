"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _build_graph(instance):
    """Build adjacency, in-degree, and successor map."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj = defaultdict(list)       # task -> successors
    in_deg = np.zeros(n, dtype=int)
    
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                adj[i].append(j)
                in_deg[j] += 1
    
    return adj, in_deg


def _topo_sort(n, in_deg, adj):
    """Kahn's algorithm for topological sort."""
    in_deg_copy = in_deg.copy()
    queue = [i for i in range(n) if in_deg_copy[i] == 0]
    heapq.heapify(queue)
    topo = []
    while queue:
        u = heapq.heappop(queue)
        topo.append(u)
        for v in adj[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                heapq.heappush(queue, v)
    return topo


def _longest_path_forward(times, topo, adj):
    """Longest path from start to each node."""
    n = len(times)
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        fwd[u] = times[u]
        for v in adj[u]:
            if fwd[u] + times[v] > fwd[v]:
                fwd[v] = fwd[u] + times[v]
    return fwd


def _longest_path_backward(times, topo, adj):
    """Longest path from each node to end."""
    n = len(times)
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in adj[u]:
            if bwd[v] > best:
                best = bwd[v]
        bwd[u] = times[u] + best
    return bwd


def _count_successors(n, adj):
    """Count direct and indirect successors for each node."""
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(adj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(adj[v])
        succ_count[i] = len(visited)
    return succ_count


def _positional_weight(times, topo, adj):
    """Total processing time of node + all downstream nodes (inclusive)."""
    n = len(times)
    pos = np.zeros(n, dtype=float)
    for u in reversed(topo):
        total = times[u]
        for v in adj[u]:
            total += pos[v]
        pos[u] = total
    return pos


def _downstream_min_stations(times, topo, adj, cycle_time):
    """
    Lower bound on the number of stations needed to complete the downstream
    subtree of each node, derived via longest-path depth / total work.
    For node u: min_stations[u] = max(
        1,
        ceil(total_downstream_work[u] / cycle_time),
        max over successors v of (1 + min_stations[v])  [path depth bound]
    )
    This captures how "scarce" station capacity is downstream of u.
    """
    n = len(times)
    # First compute total downstream work (same as positional weight)
    total_work = np.zeros(n, dtype=float)
    for u in reversed(topo):
        total_work[u] = times[u]
        for v in adj[u]:
            total_work[u] += total_work[v]

    # Path-depth bound: longest chain of successors
    path_depth = np.zeros(n, dtype=int)
    for u in reversed(topo):
        best = 0
        for v in adj[u]:
            if path_depth[v] > best:
                best = path_depth[v]
        path_depth[u] = 1 + best

    # Combine: min stations >= max(1, ceil(work/cycle), path_depth)
    work_bound = np.ceil(total_work / (cycle_time + 1e-9)).astype(int)
    min_stations = np.maximum(1, np.maximum(work_bound, path_depth))
    return min_stations


def _count_predecessors(instance):
    """Count direct predecessors for each task."""
    preds = instance["predecessors"]
    n = len(instance["times"])
    pred_counts = np.zeros(n, dtype=int)
    for i in range(n):
        if i in preds:
            pred_counts[i] = len(preds[i])
    return pred_counts


def _normalize(arr):
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _assign_stations(instance, priority_scores, cycle_time):
    """
    Greedy assignment: fill stations with highest-priority eligible task that fits.
    Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj, in_deg = _build_graph(instance)
    pred_counts = in_deg.copy()
    
    assigned = np.zeros(n, dtype=bool)
    rem_preds = pred_counts.copy()
    station_time = 0.0
    num_stations = 1
    
    while assigned.sum() < n:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            break
        
        # Among eligible, find those that fit in current station
        remaining_capacity = cycle_time - station_time
        fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
        
        if fit:
            # Pick highest priority among those that fit
            best = max(fit, key=lambda i: priority_scores[i])
        else:
            # Close current station, start new one
            station_time = 0.0
            num_stations += 1
            remaining_capacity = cycle_time
            fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
            if not fit:
                # Should not happen if times[i] <= cycle_time
                best = max(eligible, key=lambda i: priority_scores[i])
            else:
                best = max(fit, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        
        # Update successor counts
        for s in adj[best]:
            rem_preds[s] -= 1
    
    return num_stations


def _generate_candidates(instance):
    """Generate multiple candidate priority vectors."""
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    adj, in_deg = _build_graph(instance)
    topo = _topo_sort(n, in_deg, adj)
    
    candidates = []
    
    # 1. Longest processing time
    candidates.append(np.asarray(times, dtype=float))
    
    # 2. Longest path to end (rank)
    bwd = _longest_path_backward(times, topo, adj)
    candidates.append(bwd)
    
    # 3. Critical path (forward + backward)
    fwd = _longest_path_forward(times, topo, adj)
    crit = fwd + bwd - times
    candidates.append(crit)
    
    # 4. Modified rank (successor count)
    succ_count = _count_successors(n, adj)
    candidates.append(succ_count.astype(float))
    
    # 5. Composite weighted
    p_time = _normalize(times)
    p_rank = _normalize(bwd)
    p_crit = _normalize(crit)
    p_succ = _normalize(succ_count.astype(float))
    composite = 0.35 * p_time + 0.3 * p_rank + 0.2 * p_crit + 0.15 * p_succ
    candidates.append(composite)
    
    # 6. Time * successor count
    candidates.append(times * (1.0 + succ_count.astype(float)))
    
    # 7. Backward path weighted by time
    candidates.append(bwd * (1.0 + 0.1 * times))
    
    # 8. Inverse predecessor count weighted
    pred_counts = _count_predecessors(instance)
    denom = 1.0 + pred_counts.astype(float)
    candidates.append(times / denom)
    
    # 9. Forward path
    candidates.append(fwd)
    
    # 10. Time * critical path
    candidates.append(times * crit)
    
    # 11. Bottleneck density: time / min_stations_downstream
    #     Penalizes long tasks where downstream station capacity is scarce
    min_stations = _downstream_min_stations(times, topo, adj, cycle_time).astype(float)
    bottleneck_density = times / (min_stations + 1e-9)
    candidates.append(bottleneck_density)
    
    # 12. Positional weight (total downstream work including self)
    pos = _positional_weight(times, topo, adj)
    candidates.append(pos)
    
    # 13. Bottleneck density * positional weight (combined scarcity + volume)
    candidates.append(bottleneck_density * (1.0 + 0.1 * _normalize(pos)))
    
    # 14. Positional weight normalized + bottleneck density normalized (blend)
    p_pos = _normalize(pos)
    p_bnd = _normalize(bottleneck_density)
    candidates.append(0.5 * p_pos + 0.5 * p_bnd)
    
    # 15. Time * (1 / min_stations) * (1 + normalized positional weight)
    candidates.append(times * (1.0 / (min_stations + 1e-9)) * (1.0 + p_pos))
    
    # 16. Critical path / min_stations (path scarcity)
    candidates.append(crit / (min_stations + 1e-9))
    
    return candidates


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Uses a model-and-optimize approach: generate parameterized priority vectors
    and select the one that minimizes station count.
    """
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Generate candidate priority vectors
    candidates = _generate_candidates(instance)
    
    # Evaluate each candidate and track the best
    best_priority = candidates[0]
    best_num_stations = _assign_stations(instance, candidates[0], cycle_time)
    
    for idx in range(1, len(candidates)):
        if time.time() > _DEADLINE - 10:
            break
        try:
            ns = _assign_stations(instance, candidates[idx], cycle_time)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = candidates[idx]
        except Exception:
            continue
    
    # If we have time, try local optimization: perturb the best priority
    # and see if we can do better
    if time.time() < _DEADLINE - 20:
        rng = np.random.RandomState(42)
        for seed in range(10):
            if time.time() > _DEADLINE - 10:
                break
            # Add small noise to the best priority
            noise_scale = 0.05 * (best_priority.max() - best_priority.min() + 1e-9)
            noisy = best_priority + rng.normal(0, noise_scale, size=n)
            try:
                ns = _assign_stations(instance, noisy, cycle_time)
                if ns < best_num_stations:
                    best_num_stations = ns
                    best_priority = noisy
            except Exception:
                continue
    
    # Second phase: finer-grained local search around the best found
    if time.time() < _DEADLINE - 10:
        rng2 = np.random.RandomState(123)
        for seed in range(15):
            if time.time() > _DEADLINE - 5:
                break
            noise_scale = 0.02 * (best_priority.max() - best_priority.min() + 1e-9)
            noisy = best_priority + rng2.normal(0, noise_scale, size=n)
            try:
                ns = _assign_stations(instance, noisy, cycle_time)
                if ns < best_num_stations:
                    best_num_stations = ns
                    best_priority = noisy
            except Exception:
                continue
    
    return best_priority
# EVOLVE-BLOCK-END
