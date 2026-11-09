"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    The heuristic fills stations with the highest-priority eligible task that fits.
    We want to minimize the number of stations.
    
    Strategy:
    1. Compute structural features: rank (longest path), slack, successors.
    2. Model "criticality" using a flow-network-inspired metric (bottleneck detection).
    3. Use a composite priority combining rank, time, and criticality.
    4. Perform local search (hill climbing) on priority weights/perturbations.
    5. Deadline guard to stay within 120s.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build predecessor sets and successor lists
    pred_sets = [set(preds) for preds in predecessors]
    succ_sets = [set() for _ in range(n)]
    for i, preds in enumerate(predecessors):
        for p in preds:
            succ_sets[p].add(i)
    
    # Compute in-degree and out-degree
    in_degree = np.array([len(preds) for preds in predecessors])
    out_degree = np.array([len(s) for s in succ_sets])
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_sets[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    if len(topo_order) != n:
        # Cycle detected or invalid graph; return default priority
        return times.copy()
    
    # Compute rank (longest path from start)
    rank = np.zeros(n)
    for u in topo_order:
        for v in succ_sets[u]:
            rank[v] = max(rank[v], rank[u] + times[u])
    
    # Compute reverse rank (longest path to end) - "remaining work"
    # Process in reverse topological order
    rank_to_end = np.zeros(n)
    for u in reversed(topo_order):
        for v in succ_sets[u]:
            rank_to_end[u] = max(rank_to_end[u], times[u] + rank_to_end[v])
    
    # Compute slack: total_time - rank - rank_to_end + times[i]
    # Actually, slack = (cycle_time - rank_to_end[i]) - rank[i] ? No.
    # Standard slack in scheduling: LS - ES.
    # ES = rank[i]
    # LS = cycle_time - rank_to_end[i]
    # Slack = LS - ES = cycle_time - rank_to_end[i] - rank[i]
    slack = cycle_time - rank_to_end - rank
    
    # Compute "Criticality" based on flow/bottleneck idea
    # Idea: Tasks with high downstream demand (many successors, long paths) are bottlenecks.
    # We approximate the "capacity" of the subgraph downstream.
    # A simple proxy: sum of times of all successors (transitive).
    # Let's compute transitive successor sum (total work downstream)
    downstream_sum = np.zeros(n)
    # Process in reverse topological order to accumulate
    for u in reversed(topo_order):
        for v in succ_sets[u]:
            downstream_sum[u] += times[v] + downstream_sum[v]
    
    # Criticality: Higher if downstream_sum is large (bottleneck for capacity)
    # But we also want to prioritize tasks that are "hard to place" (long time, high rank)
    
    # Normalize features
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_rank = normalize(rank)
    norm_rank_to_end = normalize(rank_to_end)
    norm_time = normalize(times)
    norm_downstream = normalize(downstream_sum)
    norm_slack = normalize(slack) # Low slack = high priority usually
    
    # Base components
    # 1. Rank to end (critical path remaining)
    # 2. Time (LPT - Longest Processing Time first)
    # 3. Downstream sum (bottleneck/criticality)
    # 4. Inverse Slack (tight tasks first)
    
    # Initial weights
    # Weighting strategy:
    # - rank_to_end is crucial for precedence feasibility and packing.
    # - time helps pack large items.
    # - downstream_sum identifies structural bottlenecks.
    
    w_rank_end = 0.5
    w_time = 0.3
    w_downstream = 0.2
    w_inv_slack = 0.0 # Try adding this later if needed
    
    # Inverse slack: tasks with 0 slack are most critical
    # If slack can be negative (infeasible within cycle?), handle carefully.
    # Here slack is relative to cycle time.
    inv_slack = 1.0 / (1.0 + np.maximum(0, slack)) # Soft inverse
    
    base_priority = (w_rank_end * norm_rank_to_end + 
                     w_time * norm_time + 
                     w_downstream * norm_downstream)
    
    # Simulate function
    def simulate_station_count(prio_vec):
        """Simulate greedy station filling and return number of stations."""
        assigned = np.zeros(n, dtype=bool)
        assigned_set = set()
        remaining = set(range(n))
        stations = 0
        
        # Pre-compute eligible check helper? No, dynamic.
        # For speed, maintain list of available tasks.
        # But simple check is O(N^2) which is fine for N=30.
        
        while remaining:
            stations += 1
            # Find eligible tasks: all predecessors assigned
            eligible = [t for t in remaining if all(p in assigned_set for p in pred_sets[t])]
            if not eligible:
                # Should not happen if graph is valid and we assign at least one per station
                # If it happens, we are stuck. Break to avoid infinite loop.
                break
            
            # Sort by priority descending
            eligible_sorted = sorted(eligible, key=lambda t: -prio_vec[t])
            
            # Fill station
            used = 0.0
            placed_this_station = False
            for t in eligible_sorted:
                if used + times[t] <= cycle_time:
                    assigned[t] = True
                    assigned_set.add(t)
                    remaining.discard(t)
                    used += times[t]
                    placed_this_station = True
            
            if not placed_this_station:
                # No task fits in the cycle time? 
                # This implies times[t] > cycle_time for all eligible.
                # The problem statement implies tasks fit or we must handle it.
                # "stations are filled with the highest-priority eligible task that fits the cycle time"
                # If none fit, the heuristic might fail or we force place?
                # Standard ALB assumes task_time <= cycle_time.
                # If not, we might have to force place the highest priority to make progress?
                # Let's assume valid input where at least one fits.
                # If not, force place highest priority to prevent infinite loop.
                t = eligible_sorted[0]
                assigned[t] = True
                assigned_set.add(t)
                remaining.discard(t)
        
        return stations
    
    # Get initial station count
    current_prio = base_priority.copy()
    current_stations = simulate_station_count(current_prio)
    
    best_prio = current_prio.copy()
    best_stations = current_stations
    
    # Local search: Hill Climbing
    # We can perturb the weights or the priorities directly.
    # Perturbing priorities directly allows for more nuanced tuning.
    
    rng = np.random.default_rng(42)
    
    # Time budget for search
    search_deadline = deadline - 5
    
    # Phase 1: Weight optimization (coarse)
    # Try a grid of weights
    best_w = (w_rank_end, w_time, w_downstream)
    
    # Grid search for weights
    # w_rank_end in [0.3, 0.7], w_time in [0.1, 0.5], w_downstream = 1 - sum(others)
    # Ensure w_downstream >= 0
    
    w_re_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    w_t_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for w_re in w_re_vals:
        for w_t in w_t_vals:
            w_ds = 1.0 - w_re - w_t
            if w_ds < 0:
                continue
            
            if time.time() > search_deadline:
                break
                
            p_score = w_re * norm_rank_to_end + w_t * norm_time + w_ds * norm_downstream
            stations = simulate_station_count(p_score)
            
            if stations < best_stations:
                best_stations = stations
                best_prio = p_score.copy()
                best_w = (w_re, w_t, w_ds)
    
    # Phase 2: Fine-tuning via random perturbations on the best priority vector
    # This allows breaking ties and adjusting for specific instance quirks.
    
    if time.time() < search_deadline:
        max_iters = 1000
        for it in range(max_iters):
            if time.time() > search_deadline:
                break
            
            # Perturb a random subset of priorities
            num_perturb = rng.integers(1, 6)
            indices = rng.choice(n, size=num_perturb, replace=False)
            
            new_prio = best_prio.copy()
            for idx in indices:
                # Small Gaussian noise
                new_prio[idx] += rng.normal(0, 0.05)
            
            # Optional: Renormalize to keep scale consistent?
            # If we renormalize, we lose the specific scale adjustments.
            # Let's keep raw values but maybe clip to [0, 1.5] to avoid extremes?
            # Or just let it be. The simulation only cares about relative order.
            
            new_stations = simulate_station_count(new_prio)
            
            if new_stations < best_stations:
                best_stations = new_stations
                best_prio = new_prio
    
    # Return the best priority found
    # Ensure output is a numpy array of floats
    return best_prio

# EVOLVE-BLOCK-END
