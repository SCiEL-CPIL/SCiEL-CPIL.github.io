"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


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


def _downstream_station_requirement(times, adj, cycle_time):
    """
    For each node, compute the minimum number of stations its downstream
    subtree (including itself) would require if each station held at most
    cycle_time of work. This is an admissible lower bound on stations needed
    for that subtree.

    We approximate by: total_downstream_time / cycle_time, rounded up.
    total_downstream_time = sum of times of the node and all its successors.
    """
    n = len(times)
    # Compute total downstream time (including self) via reverse topological order
    # We need a topological order; use the adj to do a DFS post-order or use
    # the fact that successors come after in topo order.
    # Simpler: for each node, sum times of all nodes reachable (including self).
    # For n=30 this is O(n^2 * avg_succ) which is fine.
    downstream_time = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set([i])
        stack = list(adj[i])
        total = times[i]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                total += times[v]
                stack.extend(adj[v])
        downstream_time[i] = total

    # Minimum stations needed for downstream subtree
    if cycle_time <= 0:
        return np.ones(n, dtype=float)
    stations_req = np.ceil(downstream_time / cycle_time)
    # Ensure at least 1
    stations_req = np.maximum(stations_req, 1.0)
    return stations_req


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
    rem_preds = in_deg.copy()
    
    assigned = np.zeros(n, dtype=bool)
    station_time = 0.0
    num_stations = 1
    
    remaining = n
    while remaining > 0:
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
        remaining -= 1
        
        # Update successor counts
        for s in adj[best]:
            rem_preds[s] -= 1
    
    return num_stations


def _generate_candidates(instance):
    """Generate multiple candidate priority vectors."""
    times = np.asarray(instance["times"], dtype=float)
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
    
    # 11. Bottleneck density boost
    downstream_req = _downstream_station_requirement(times, adj, instance["cycle_time"])
    bottleneck_ratio = times / downstream_req
    ratio_norm = _normalize(bottleneck_ratio)
    boost_strength = 1.5
    cand_11 = times * (1.0 + boost_strength * ratio_norm)
    candidates.append(cand_11)
    
    # 12. Critical path * bottleneck ratio
    cand_12 = crit * (1.0 + 0.5 * ratio_norm)
    candidates.append(cand_12)
    
    # 13. Backward path with stronger bottleneck emphasis
    cand_13 = bwd * (1.0 + 1.0 * ratio_norm)
    candidates.append(cand_13)
    
    # 14. Downstream time (total work remaining including self)
    downstream_time = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set([i])
        stack = list(adj[i])
        total = times[i]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                total += times[v]
                stack.extend(adj[v])
        downstream_time[i] = total
    candidates.append(downstream_time)
    
    # 15. Downstream time normalized by cycle time (station pressure)
    ct = instance["cycle_time"]
    station_pressure = downstream_time / max(ct, 1e-9)
    candidates.append(station_pressure)
    
    # 16. Time squared (emphasize very long tasks)
    candidates.append(times ** 2)
    
    # 17. Rank position in topological order (earlier = higher priority)
    topo_pos = np.zeros(n, dtype=float)
    for rank, idx in enumerate(topo):
        topo_pos[idx] = float(n - rank)  # earlier tasks get higher score
    candidates.append(topo_pos)
    
    # 18. Combined: downstream_time * (1 + 0.5 * ratio_norm)
    cand_18 = downstream_time * (1.0 + 0.5 * ratio_norm)
    candidates.append(cand_18)
    
    # 19. bwd * downstream_time (both signals)
    cand_19 = bwd * downstream_time
    candidates.append(cand_19)
    
    # 20. Soft-max style: exp(times / tau) to smooth emphasis
    tau = 0.3 * times.max() + 1e-9
    cand_20 = np.exp(times / tau)
    candidates.append(cand_20)
    
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
    
    # Local optimization: swap-based search (more effective for discrete assignment)
    if time.time() < _DEADLINE - 15:
        rng = np.random.RandomState(42)
        # Sort indices by current priority (descending)
        order = np.argsort(-best_priority)
        
        # Phase 1: Try swapping pairs in the priority ordering
        improved = True
        max_swaps = 500
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < _DEADLINE - 15:
            improved = False
            for i in range(n - 1):
                if time.time() > _DEADLINE - 15:
                    break
                # Try swapping positions i and i+1 in the priority order
                a, b = order[i], order[i + 1]
                # Create modified priority
                mod_priority = best_priority.copy()
                tmp = mod_priority[a]
                mod_priority[a] = mod_priority[b]
                mod_priority[b] = tmp
                try:
                    ns = _assign_stations(instance, mod_priority, cycle_time)
                    if ns < best_num_stations:
                        best_num_stations = ns
                        best_priority = mod_priority
                        # Re-sort order
                        order = np.argsort(-best_priority)
                        improved = True
                        swap_count += 1
                        if time.time() > _DEADLINE - 15:
                            break
                except Exception:
                    pass
            # If no improvement in a full pass, try random swaps
            if not improved:
                for attempt in range(100):
                    if time.time() > _DEADLINE - 15:
                        break
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i == j:
                        continue
                    a, b = order[i], order[j]
                    mod_priority = best_priority.copy()
                    tmp = mod_priority[a]
                    mod_priority[a] = mod_priority[b]
                    mod_priority[b] = tmp
                    try:
                        ns = _assign_stations(instance, mod_priority, cycle_time)
                        if ns < best_num_stations:
                            best_num_stations = ns
                            best_priority = mod_priority
                            order = np.argsort(-best_priority)
                            improved = True
                            swap_count += 1
                    except Exception:
                        pass
                if not improved:
                    break
    
    # Phase 2: Fine-grained noise perturbation if time remains
    if time.time() < _DEADLINE - 10:
        rng2 = np.random.RandomState(123)
        for seed in range(40):
            if time.time() > _DEADLINE - 10:
                break
            noise_scale = (0.01 + 0.02 * (seed % 3) / 2.0) * (best_priority.max() - best_priority.min() + 1e-9)
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
