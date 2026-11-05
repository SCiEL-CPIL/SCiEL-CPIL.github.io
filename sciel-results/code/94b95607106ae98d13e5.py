"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Fast fallback: nearest feasible neighbor."""
    if len(unvisited_nodes) == 0:
        return depot
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])
    d = distance_matrix[current_node, feasible]
    return int(feasible[np.argmin(d)])

def _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Adaptive heuristic combining:
    - Refined savings score (strong current-proximity weight, mild depot bonus)
    - Adaptive lookahead depth based on capacity tightness
    - Smooth surrogate (log-sum-exp) for robust candidate ranking
    - Final polish: re-evaluate top candidates with deeper lookahead
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time check: if tight, use fast path
    if _time_remaining() < 1.0:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # --- Refined savings-based score ---
    # Emphasize proximity to current node (1.5x) with mild depot-proximity bonus (-0.5x)
    # This approximates the classical savings criterion in a sequential greedy context.
    d_from_current = distance_matrix[current_node, feasible]
    d_to_depot = distance_matrix[feasible, depot]

    # Normalize to [0, 1] for stable combination
    max_dc = np.max(d_from_current) if len(d_from_current) > 0 else 1.0
    max_dd = np.max(d_to_depot) if len(d_to_depot) > 0 else 1.0
    if max_dc == 0:
        max_dc = 1.0
    if max_dd == 0:
        max_dd = 1.0

    norm_dc = d_from_current / max_dc
    norm_dd = d_to_depot / max_dd

    # Weighted score: strong current-proximity, mild depot bonus
    # Lower score = better
    scores = 1.5 * norm_dc - 0.5 * norm_dd

    # --- Adaptive lookahead based on capacity tightness ---
    cap_ratio = rest_capacity / 50.0 if rest_capacity > 0 else 0.0

    if cap_ratio < 0.3:
        # Tight capacity: deep lookahead to avoid trapping
        lookahead_depth = 4
        k_candidates = min(7, len(feasible))
    elif cap_ratio < 0.6:
        # Medium capacity: moderate lookahead
        lookahead_depth = 3
        k_candidates = min(5, len(feasible))
    else:
        # High capacity: shallow lookahead, rely on score
        lookahead_depth = 2
        k_candidates = min(4, len(feasible))

    # --- Smooth surrogate for candidate ranking (log-sum-exp style) ---
    # Instead of hard argmin, use a soft ranking to avoid ties and noise
    # Temperature controls how "sharp" the ranking is
    temperature = 0.1
    shifted = scores - np.min(scores)
    weights = np.exp(-shifted / temperature)
    weights = weights / np.sum(weights)

    # Use the smooth ranking to select top-k candidates
    # Sort by score (primary) but use weights as tiebreaker for stability
    sorted_indices = np.argsort(scores)
    top_k_indices = sorted_indices[:k_candidates]
    candidates = [feasible[i] for i in top_k_indices]

    # --- Lookahead evaluation ---
    best_node = candidates[0]
    best_cost = float('inf')

    for cand in candidates:
        # Simulate a short path from cand
        sim_remaining = rest_capacity - demands[cand]
        sim_unvisited = [n for n in unvisited_nodes if n != cand]
        sim_current = cand
        sim_cost = distance_matrix[current_node, cand]

        for _ in range(lookahead_depth):
            if not sim_unvisited:
                break
            sim_feasible = [n for n in sim_unvisited if demands[n] <= sim_remaining]
            if not sim_feasible:
                break
            # Greedy nearest within simulation
            d_sim = distance_matrix[sim_current, sim_feasible]
            next_node = sim_feasible[np.argmin(d_sim)]
            sim_cost += distance_matrix[sim_current, next_node]
            sim_current = next_node
            sim_remaining -= demands[next_node]
            sim_unvisited.remove(next_node)

        # Add return to depot
        sim_cost += distance_matrix[sim_current, depot]

        if sim_cost < best_cost:
            best_cost = sim_cost
            best_node = cand

    # --- Final polish: if time permits, re-evaluate with deeper lookahead ---
    if _time_remaining() > 5.0 and len(candidates) >= 2:
        # Re-evaluate top 3 with one more step of lookahead
        polish_depth = lookahead_depth + 1
        best_polish_cost = float('inf')
        best_polish_node = best_node

        for cand in candidates[:3]:
            sim_remaining = rest_capacity - demands[cand]
            sim_unvisited = [n for n in unvisited_nodes if n != cand]
            sim_current = cand
            sim_cost = distance_matrix[current_node, cand]

            for _ in range(polish_depth):
                if not sim_unvisited:
                    break
                sim_feasible = [n for n in sim_unvisited if demands[n] <= sim_remaining]
                if not sim_feasible:
                    break
                d_sim = distance_matrix[sim_current, sim_feasible]
                next_node = sim_feasible[np.argmin(d_sim)]
                sim_cost += distance_matrix[sim_current, next_node]
                sim_current = next_node
                sim_remaining -= demands[next_node]
                sim_unvisited.remove(next_node)

            sim_cost += distance_matrix[sim_current, depot]

            if sim_cost < best_polish_cost:
                best_polish_cost = sim_cost
                best_polish_node = cand

        best_node = best_polish_node

    return int(best_node)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        return _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

    if len(unvisited_nodes) == 0:
        return depot

    return _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
