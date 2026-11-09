"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    The heuristic fills stations with the highest-priority eligible task that fits.
    We want to minimize the number of stations.
    
    Strategy:
    1. Compute a "slack-based" priority that encourages packing tight tasks.
    2. Use local search (hill climbing) to refine priority weights if time allows.
    3. Deadline guard to stay within 120s.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build predecessor sets
    pred_sets = [set(preds) for preds in predecessors]
    
    # Compute successors for topological ordering awareness
    succ_sets = [set() for _ in range(n)]
    for i, preds in enumerate(predecessors):
        for p in preds:
            succ_sets[p].add(i)
    
    # Compute rank (longest path from start)
    rank = np.zeros(n)
    # Topological order
    in_degree = np.array([len(preds) for preds in predecessors])
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_sets[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
            rank[v] = max(rank[v], rank[u] + times[u])
    
    # Initial priority: combine time, rank, and a slack-like measure
    # Higher priority = should be placed earlier
    # We want to pack stations tightly, so prioritize tasks that are "hard to place"
    
    # Compute a base priority
    # Idea: prioritize tasks with more successors (critical path), longer times, 
    # and tasks that have fewer predecessors (more flexibility early)
    
    num_succ = np.array([len(s) for s in succ_sets])
    num_pred = np.array([len(p) for p in predecessors])
    
    # Base priority: time + 0.5 * rank + 0.1 * num_succ - 0.05 * num_pred
    base_priority = times + 0.5 * rank + 0.1 * num_succ - 0.05 * num_pred
    
    # Simulate the greedy algorithm to count stations
    def simulate_station_count(prio_vec):
        """Simulate greedy station filling and return number of stations."""
        assigned = np.zeros(n, dtype=bool)
        stations = 0
        remaining = set(range(n))
        
        while remaining:
            stations += 1
            # Find eligible tasks: all predecessors assigned
            eligible = [t for t in remaining if all(p in assigned_set for p in pred_sets[t])]
            if not eligible:
                # Should not happen if graph is valid
                break
            # Sort by priority descending
            eligible_sorted = sorted(eligible, key=lambda t: -prio_vec[t])
            # Fill station
            used = 0.0
            placed_this_station = []
            for t in eligible_sorted:
                if used + times[t] <= cycle_time:
                    assigned[t] = True
                    assigned_set.add(t)
                    remaining.discard(t)
                    used += times[t]
                    placed_this_station.append(t)
            if not placed_this_station:
                # No task fits, force place highest priority eligible
                t = eligible_sorted[0]
                assigned[t] = True
                assigned_set.add(t)
                remaining.discard(t)
                used += times[t]
        
        return stations
    
    assigned_set = set()
    
    # Get initial station count
    current_prio = base_priority.copy()
    current_stations = simulate_station_count(current_prio)
    
    # Local search: hill climbing on priority perturbations
    best_prio = current_prio.copy()
    best_stations = current_stations
    
    # Try random perturbations
    rng = np.random.default_rng(42)
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Perturb a random subset of priorities
        num_perturb = rng.integers(1, 5)
        indices = rng.choice(n, size=num_perturb, replace=False)
        
        new_prio = best_prio.copy()
        for idx in indices:
            new_prio[idx] += rng.normal(0, 1.0)
        
        # Normalize to keep scale similar
        new_prio = (new_prio - new_prio.min()) / (new_prio.max() - new_prio.min() + 1e-10)
        new_prio = new_prio * best_prio.max()
        
        new_stations = simulate_station_count(new_prio)
        
        if new_stations < best_stations:
            best_prio = new_prio
            best_stations = new_stations
    
    # Return the best priority found
    # Scale back to original magnitude for consistency
    if best_prio.max() > 0:
        result = best_prio / best_prio.max() * times.max()
    else:
        result = base_priority.copy()
    
    return result

# EVOLVE-BLOCK-END
