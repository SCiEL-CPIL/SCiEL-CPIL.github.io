"""Capacitated vehicle routing heuristic: Dynamic Savings with Capacity Penalty."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_START_TIME = None


def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill


def _time_left():
    _ensure_deadline()
    return _DEADLINE - time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _ensure_deadline()
    if _time_left() < 0.5:
        # Safety: return depot or nearest feasible if any
        if len(unvisited_nodes) == 0:
            return int(depot)
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        return int(min(feasible, key=lambda n: distance_matrix[current_node, n]))

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return int(depot)

    dm = distance_matrix
    cur = int(current_node)
    dep = int(depot)

    # Feasibility filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return int(depot)

    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])

    n = len(feasible)
    
    # Vectorized distance calculations
    # dist_cur[i] = distance from current node to feasible node i
    # dist_dep[i] = distance from depot to feasible node i
    dist_cur = np.array([dm[cur, i] for i in feasible])
    dist_dep = np.array([dm[dep, i] for i in feasible])
    
    # Calculate "Savings" score.
    # Classical savings for connecting depot->i->cur vs depot->i + depot->cur is not directly applicable 
    # because we are building a path. 
    # Instead, we use a heuristic: 
    # Score = -distance_to_next (minimize travel) 
    #         + lambda * (distance_from_depot) (prefer nodes far from depot to group them, 
    #           BUT this is counter-intuitive if we are already far. 
    #           Actually, standard VRP heuristics prefer nodes that are close to current 
    #           but "expensive" to visit later. 
    #           Let's use the "Regret" proxy but simplified: 
    #           How much do we lose if we don't go to i now?
    #           Approximation: If we go to j instead of i, we pay dist_cur[j] + dist(j,i).
    #           If we go to i, we pay dist_cur[i].
    #           Loss = min_j (dist_cur[j] + dist(j,i)) - dist_cur[i].
    #           This is the regret.
    #           
    #   NEW STRUCTURAL CHANGE: 
    #   Use a "Dynamic Savings" approach inspired by Clarke-Wright but adapted for sequential selection.
    #   We estimate the "value" of visiting i now vs later.
    #   Value(i) = Savings(i) - Penalty(i)
    #   Savings(i) is approximated by how much i is "aligned" with the current direction or 
    #   how much it reduces the "open" distance to the depot.
    #   
    #   Let's try a simpler structural shift: 
    #   1. Compute Regret (O(N^2) is fine for N=50) as in previous version, but 
    #      add a DYNAMIC PENALTY that scales with (1 - rest_capacity / total_capacity).
    #      As capacity fills, we penalize long distances more heavily to avoid 
    #      "stranded" nodes far away that can't be connected efficiently later.
    #      
    #   Actually, the cell finding said: "Dynamic penalty weight that increases as 
    #   rest_capacity decreases, reducing the effective savings coefficient on dist_depot 
    #   and making the heuristic prefer nodes closer to the depot when capacity is tight"
    #   
    #   Let's implement:
    #   Score(i) = Regret(i) - Dynamic_Penalty * Distance(i)
    #   Where Dynamic_Penalty increases as rest_capacity drops.
    #   Wait, if we penalize distance, we prefer close nodes. 
    #   The finding said "prefer nodes closer to the depot". 
    #   So maybe: Score(i) = Regret(i) - Dynamic_Penalty * Distance_Depot(i)?
    #   Or Score(i) = Regret(i) - Dynamic_Penalty * Distance_Cur(i)?
    #   "Prefer nodes closer to the depot" implies minimizing dist_dep.
    
    # Let's compute Regret first.
    regrets = np.zeros(n)
    # Precompute distance matrix between feasible nodes for O(1) lookup in loop? 
    # Or just use dm directly.
    for idx, i in enumerate(feasible):
        if n == 1:
            regrets[idx] = 0.0
            continue
        min_skip_cost = np.inf
        # Find min cost to go to j then i
        for jdx, j in enumerate(feasible):
            if j == i:
                continue
            c = dist_cur[jdx] + dm[j, i]
            if c < min_skip_cost:
                min_skip_cost = c
        cost_take = dist_cur[idx]
        regrets[idx] = min_skip_cost - cost_take

    # Normalize Regret to [0, 1] roughly
    if regrets.max() > 0:
        regret_norm = regrets / regrets.max()
    else:
        regret_norm = regrets

    # Dynamic Penalty Component
    # We need to estimate total capacity to normalize rest_capacity.
    # We don't have total capacity directly, but we can assume it's roughly the sum of demands 
    # or a fixed value. The problem states "capacity 50". Let's assume C=50.
    # If rest_capacity is low, we should prefer nodes that allow us to return to depot 
    # or are close to the depot? 
    # "Prefer nodes closer to the depot" -> Minimize dist_dep.
    # So we subtract a penalty proportional to dist_dep.
    
    # Normalize dist_dep
    if dist_dep.max() > 0:
        dist_dep_norm = dist_dep / dist_dep.max()
    else:
        dist_dep_norm = dist_dep

    # Also consider dist_cur to avoid long jumps generally.
    if dist_cur.max() > 0:
        dist_cur_norm = dist_cur / dist_cur.max()
    else:
        dist_cur_norm = dist_cur

    # Dynamic weight: 
    # If rest_capacity is high, we can afford to go far (low penalty).
    # If rest_capacity is low, we should stick to nodes close to depot (high penalty on dist_dep).
    # Let capacity = 50 (from problem description).
    capacity = 50.0
    # Usage = 1 - rest/capacity
    usage = 1.0 - (rest_capacity / capacity)
    if usage < 0: usage = 0.0
    if usage > 1: usage = 1.0
    
    # Penalty weight increases with usage.
    # At start (usage=0), penalty is low. At end (usage=1), penalty is high.
    # Let's say penalty_weight = 0.5 * usage + 0.1 (base penalty)
    penalty_weight = 0.5 * usage + 0.1
    
    # Combine
    # Primary: Regret (urgency)
    # Secondary: Minimize distance to depot (to keep route compact when full)
    # Tertiary: Minimize current distance (efficiency)
    
    # Score = regret_norm - penalty_weight * dist_dep_norm - 0.2 * dist_cur_norm
    # We want to MAXIMIZE score.
    
    scores = regret_norm - penalty_weight * dist_dep_norm - 0.2 * dist_cur_norm
    
    best_idx = int(np.argmax(scores))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
