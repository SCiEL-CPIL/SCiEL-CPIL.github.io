"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0, deadline=None):
    """Run the ant colony optimization with the given heuristic."""
    n = len(prizes)
    rng = np.random.default_rng(seed)

    # Heuristic: entries that are not positive become 1e-9
    eta = np.array(heuristic, dtype=np.float64, copy=True)
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9

    # Pheromone starts at 1
    tau = np.ones((n, n), dtype=np.float64)

    d = distance_matrix
    # Precompute cost to return to 0: d[j, 0]
    d_to_0 = d[:, 0]

    best_global = 0.0

    start_time = time.time()
    for it in range(num_iters):
        # Deadline guard
        if deadline is not None and time.time() > deadline:
            break

        best_iter = 0.0

        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0

            while True:
                remaining = max_len - length
                # Valid nodes: unvisited, and length + d[cur,j] + d[j,0] <= max_len
                valid_mask = (~visited) & (d[cur, :] + d_to_0 <= remaining)

                if not np.any(valid_mask):
                    break

                # Compute probabilities proportional to tau[cur, j] * eta[cur, j]
                weights = tau[cur, :] * eta[cur, :]
                weights[~valid_mask] = 0.0

                total_w = np.sum(weights)
                if total_w <= 0:
                    break

                probs = weights / total_w

                # Sample next node
                next_node = rng.choice(n, p=probs)

                # Move
                length += d[cur, next_node]
                prize += prizes[next_node]
                visited[next_node] = True
                cur = next_node

            if prize > best_iter:
                best_iter = prize

        if best_iter > best_global:
            best_global = best_iter

        # Pheromone update: multiply by 0.9, add best_iter / 10
        tau *= 0.9
        tau += best_iter / 10.0

    return best_global


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    The heuristic is designed to favor edges that lead to high-prize nodes
    while penalizing long distances, especially when the remaining budget
    is tight.
    
    For each edge (i, j), the attractiveness should reflect:
    - How valuable node j is (prizes[j])
    - How much distance is consumed by going from i to j and then back to 0
    - The remaining budget constraint
    
    A good heuristic balances prize per unit of "effective cost" where
    effective cost accounts for the round-trip constraint.
    """
    n = len(prizes)
    d = distance_matrix
    d_to_0 = d[:, 0]

    # For each edge (i, j), the effective cost if we go i -> j -> 0
    # is d[i, j] + d[j, 0]. But we also need to consider the path so far.
    # Since we don't know the path, we use a static approximation.
    
    # Key insight: An edge (i, j) is only useful if d[i, j] + d[j, 0] <= max_len
    # (assuming we're at i with 0 length so far, which is the best case).
    # More realistically, the constraint is length_so_far + d[i,j] + d[j,0] <= max_len.
    
    # A strong heuristic: prize[j] / (d[i, j] + d[j, 0])^alpha
    # This favors high prize per unit distance.
    
    # We can also incorporate the "opportunity cost" - if d[i,j] + d[j,0] is
    # close to max_len, it's a risky edge.
    
    # Let's use: prize[j] / (d[i, j] + d[j, 0] + epsilon)
    # with a slight boost for edges that are "safe" (well within budget)
    
    epsilon = 1e-9
    effective_cost = d + d_to_0[None, :]  # shape (n, n): d[i,j] + d[j,0]
    
    # Base heuristic: prize per effective cost
    base = prizes[None, :] / (effective_cost + epsilon)
    
    # Add a bonus for edges that are well within the budget
    # If effective_cost < 0.5 * max_len, it's a "safe" edge
    safety_bonus = np.where(effective_cost < 0.5 * max_len, 1.5, 1.0)
    
    # Combine
    heuristic = base * safety_bonus
    
    # Ensure no negative or zero values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
