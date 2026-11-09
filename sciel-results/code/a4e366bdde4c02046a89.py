"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import itertools


def _simulate_priority(instance, scores):
    """Simulate the station-filling algorithm with given priority scores.
    Returns the number of stations used."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor list
    succ = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        plist = preds[i] if isinstance(preds, (list, tuple, np.ndarray)) else []
        # Handle different predecessor formats
        if isinstance(plist, (list, tuple, np.ndarray)):
            for p in plist:
                succ[int(p)].append(i)
                pred_count[i] += 1
    
    # Track state
    remaining_preds = pred_count[:]
    completed = [False] * n
    station_time = 0.0
    station_tasks = []
    num_stations = 1
    unassigned = set(range(n))
    
    while unassigned:
        # Find eligible tasks: all predecessors completed, fits in remaining cycle time
        eligible = []
        for i in unassigned:
            if remaining_preds[i] == 0 and times[i] <= cycle_time - station_time:
                eligible.append(i)
        
        if not eligible:
            # Start new station
            num_stations += 1
            station_time = 0.0
            station_tasks = []
            # Re-find eligible
            for i in unassigned:
                if remaining_preds[i] == 0 and times[i] <= cycle_time:
                    eligible.append(i)
        
        if not eligible:
            # Should not happen if valid instance, but guard against infinite loop
            break
        
        # Pick highest priority eligible task
        best_task = max(eligible, key=lambda i: scores[i])
        
        # Assign task to current station
        station_time += times[best_task]
        station_tasks.append(best_task)
        unassigned.discard(best_task)
        completed[best_task] = True
        
        # Update predecessor counts for successors
        for s in succ[best_task]:
            remaining_preds[s] -= 1
    
    return num_stations


def _compute_ranking_scores(instance):
    """Compute multiple heuristic scores for each task."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build adjacency
    succ = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        plist = preds[i] if isinstance(preds, (list, tuple, np.ndarray)) else []
        if isinstance(plist, (list, tuple, np.ndarray)):
            for p in plist:
                succ[int(p)].append(i)
                pred_count[i] += 1
    
    # 1. Longest processing time (LPT) - already the baseline
    lpt = times.copy()
    
    # 2. Number of successors (direct)
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # 3. Total number of successors (transitive) - use BFS/DFS
    total_succ = [0] * n
    for i in range(n):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for s in succ[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        total_succ[i] = len(visited)
    total_succ = np.array(total_succ, dtype=float)
    
    # 4. Longest path from task to end (critical path position)
    # Compute longest path from each node to any terminal
    # Use topological order
    in_degree = pred_count[:]
    topo_order = []
    from collections import deque
    q = deque([i for i in range(n) if in_degree[i] == 0])
    in_degree_copy = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                q.append(s)
    
    # Longest path from node to end
    longest_to_end = [0.0] * n
    for node in reversed(topo_order):
        if not succ[node]:
            longest_to_end[node] = times[node]
        else:
            longest_to_end[node] = times[node] + max(longest_to_end[s] for s in succ[node])
    longest_to_end = np.array(longest_to_end, dtype=float)
    
    # 5. Longest path from start to node
    longest_from_start = [0.0] * n
    for node in topo_order:
        if not pred_count[node] or all(p not in preds[node] for p in range(n)):
            # Check if any predecessors
            plist = preds[node] if isinstance(preds[node], (list, tuple, np.ndarray)) else []
            if len(plist) == 0:
                longest_from_start[node] = times[node]
            else:
                longest_from_start[node] = times[node] + max(longest_from_start[int(p)] for p in plist)
        else:
            plist = preds[node] if isinstance(preds[node], (list, tuple, np.ndarray)) else []
            if len(plist) == 0:
                longest_from_start[node] = times[node]
            else:
                longest_from_start[node] = times[node] + max(longest_from_start[int(p)] for p in plist)
    longest_from_start = np.array(longest_from_start, dtype=float)
    
    # 6. Combination: critical path position + successors
    critical = longest_to_end + 0.5 * longest_from_start
    successor_weighted = num_succ + 0.1 * total_succ
    
    # 7. Rank-based: sort by time, give higher rank to larger times
    sorted_idx = np.argsort(-times)
    rank = np.zeros(n)
    for r, idx in enumerate(sorted_idx):
        rank[idx] = n - r  # higher rank for larger times
    
    # 8. Time * (1 + successor_count)
    time_succ = times * (1.0 + 0.1 * num_succ)
    
    # 9. Longest to end * (1 + 0.1 * num_succ)
    lte_succ = longest_to_end * (1.0 + 0.1 * num_succ)
    
    # 10. Normalized time + normalized critical path
    norm_time = times / (np.max(times) + 1e-9)
    norm_crit = longest_to_end / (np.max(longest_to_end) + 1e-9)
    combined_norm = 0.6 * norm_time + 0.4 * norm_crit
    
    # 11. Fit-aware score: penalize tasks whose time is close to cycle_time
    #     (they waste a station if placed alone). Higher score = better fit.
    #     Use a smooth penalty: score = time * (1 - max(0, (time - 0.75*CT) / (CT - 0.75*CT)))
    #     This smoothly reduces priority for tasks that would nearly fill a station.
    ct = cycle_time
    if ct > 0:
        # Smoothly penalize: tasks with time > 0.75*CT get reduced priority
        # Use a quadratic penalty for smoothness
        excess = np.maximum(0, times - 0.70 * ct)
        fit_penalty = excess / (ct + 1e-9)
        fit_aware = times * (1.0 - 0.3 * fit_penalty)
    else:
        fit_aware = times.copy()
    
    # 12. Successor-weighted critical path: critical path length weighted by
    #     how many tasks depend on this one (transitive successors)
    lte_weighted = longest_to_end * (1.0 + 0.15 * total_succ)
    
    # 13. Time * (1 + 0.2 * normalized_total_succ)
    norm_total_succ = total_succ / (np.max(total_succ) + 1e-9)
    time_ns = times * (1.0 + 0.2 * norm_total_succ)
    
    return {
        'lpt': lpt,
        'num_succ': num_succ,
        'total_succ': total_succ,
        'longest_to_end': longest_to_end,
        'longest_from_start': longest_from_start,
        'critical': critical,
        'successor_weighted': successor_weighted,
        'rank': rank,
        'time_succ': time_succ,
        'lte_succ': lte_succ,
        'combined_norm': combined_norm,
        'fit_aware': fit_aware,
        'lte_weighted': lte_weighted,
        'time_ns': time_ns,
    }


def _try_score_combinations(instance, base_scores, budget_seconds=100):
    """Try various linear combinations of base scores to find best."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    best_stations = float('inf')
    best_scores = base_scores['lpt'].copy()
    start_time = time.time()
    
    # Try each individual score first
    for name, scores in base_scores.items():
        if time.time() - start_time > budget_seconds:
            break
        stations = _simulate_priority(instance, scores)
        if stations < best_stations:
            best_stations = stations
            best_scores = scores.copy()
    
    # Try pairwise combinations
    score_names = list(base_scores.keys())
    for i, name1 in enumerate(score_names):
        if time.time() - start_time > budget_seconds:
            break
        for j, name2 in enumerate(score_names):
            if i == j:
                continue
            if time.time() - start_time > budget_seconds:
                break
            for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                if time.time() - start_time > budget_seconds:
                    break
                combined = w * base_scores[name1] + (1.0 - w) * base_scores[name2]
                stations = _simulate_priority(instance, combined)
                if stations < best_stations:
                    best_stations = stations
                    best_scores = combined.copy()
    
    # Try some triple combinations with fixed weights
    triple_candidates = [
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.6, 0.2, 0.2),
        (0.3, 0.3, 0.4),
        (0.4, 0.3, 0.3),
        (0.5, 0.25, 0.25),
    ]
    for w1, w2, w3 in triple_candidates:
        if time.time() - start_time > budget_seconds:
            break
        for i, name1 in enumerate(score_names[:6]):  # Limit to top 6
            if time.time() - start_time > budget_seconds:
                break
            for j, name2 in enumerate(score_names[:6]):
                if i == j:
                    continue
                if time.time() - start_time > budget_seconds:
                    break
                for k, name3 in enumerate(score_names[:6]):
                    if i == k or j == k:
                        continue
                    if time.time() - start_time > budget_seconds:
                        break
                    combined = (w1 * base_scores[name1] + 
                               w2 * base_scores[name2] + 
                               w3 * base_scores[name3])
                    stations = _simulate_priority(instance, combined)
                    if stations < best_stations:
                        best_stations = stations
                        best_scores = combined.copy()
    
    # Random search over simplex for 3-4 feature combinations
    # Use a seeded RNG for reproducibility
    rng = np.random.default_rng(42)
    remaining_budget = budget_seconds - (time.time() - start_time)
    if remaining_budget > 5.0:
        # Estimate time per simulation
        n_sims_done = 1  # rough count
        elapsed = time.time() - start_time
        time_per_sim = elapsed / max(n_sims_done, 1)
        max_random_sims = int(remaining_budget / max(time_per_sim, 0.01))
        max_random_sims = min(max_random_sims, 200)
        
        # Focus on the most promising features
        top_features = ['lpt', 'longest_to_end', 'fit_aware', 'lte_weighted', 
                       'critical', 'time_ns', 'combined_norm', 'lte_succ']
        top_features = [f for f in top_features if f in base_scores]
        
        for _ in range(max_random_sims):
            if time.time() - start_time > budget_seconds - 2.0:
                break
            # Pick 2-4 features randomly
            n_feats = rng.integers(2, 5)
            n_feats = min(n_feats, len(top_features))
            chosen = rng.choice(len(top_features), size=n_feats, replace=False)
            # Random Dirichlet weights
            alpha = rng.dirichlet(np.ones(n_feats) * 0.5)
            combined = np.zeros(n)
            for idx, feat_idx in enumerate(chosen):
                combined += alpha[idx] * base_scores[top_features[feat_idx]]
            stations = _simulate_priority(instance, combined)
            if stations < best_stations:
                best_stations = stations
                best_scores = combined.copy()
    
    return best_scores


def priority(instance):
    """Score each assembly task for the station-filling algorithm."""
    start_time = time.time()
    deadline = start_time + 120.0  # Hard deadline with margin
    
    # Compute base heuristic scores
    base_scores = _compute_ranking_scores(instance)
    
    # Check if we've already exceeded deadline
    if time.time() > deadline - 5.0:
        return base_scores['lpt']
    
    # Try to find a better combination
    try:
        best_scores = _try_score_combinations(instance, base_scores, budget_seconds=110)
    except Exception:
        best_scores = base_scores['lpt']
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.asarray(base_scores['lpt'], dtype=float)
    
    return np.asarray(best_scores, dtype=float)
# EVOLVE-BLOCK-END
