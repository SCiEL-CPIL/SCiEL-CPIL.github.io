import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Safety guard: return a default heuristic if time is running out or inputs are weird
    # The function is called once per instance, so we don't have a long loop here, 
    # but we ensure we don't hang on edge cases.
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))

    # 1. Base Heuristic: Prize / Cost
    # Cost to visit node j and return to depot is d[i, j] + d[j, 0] if coming from i.
    # However, the heuristic is a property of the edge (i, j) in the ACO context.
    # The ACO transition probability is proportional to pheromone[i, j] * heuristic[i, j].
    # The "cost" of moving i->j is d[i,j]. The "future cost" is d[j,0].
    # A good heuristic balances immediate prize gain against the cost of the move and the return.
    
    # Effective cost for moving to j from i: d[i,j] + d[j,0]
    # We use this to normalize the prize of node j.
    return_cost = d[:, 0] # Cost to return to 0 from any node j
    cost_matrix = d + return_cost[None, :]
    
    # Avoid division by zero
    cost_matrix = np.where(cost_matrix < 1e-9, 1e-9, cost_matrix)
    
    # Prize of destination node j
    prize_row = prizes[None, :]
    
    # Base heuristic: Efficiency of visiting j given the cost to get there and back
    H_base = prize_row / cost_matrix
    
    # 2. Tour-Aware Heuristic Injection
    # Construct a greedy tour to identify promising edges.
    # We simulate a simple greedy algorithm: always pick the next node that maximizes 
    # prize / (d[current, next] + d[next, 0]) subject to feasibility.
    
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    length_so_far = 0.0
    greedy_edges = []
    
    # We run a greedy construction to find a valid high-prize path
    for _ in range(n - 1):
        best_node = -1
        best_score = -1
        for j in range(n):
            if visited[j]:
                continue
            # Check feasibility: length_so_far + d[current, j] + d[j, 0] <= max_len
            new_len = length_so_far + d[current, j] + d[j, 0]
            if new_len <= max_len + 1e-9:
                # Score for greedy selection: Prize / incremental cost
                # Incremental cost is d[current, j] + d[j, 0] - d[current, 0] ? 
                # Actually, the constraint is on total length.
                # Let's use the same efficiency metric: prize / (d[current, j] + d[j, 0])
                # But we must ensure we can still return. The feasibility check above handles that.
                # A better greedy score might be prize / d[current, j] to prioritize close high prizes?
                # Let's stick to prize / (d[current, j] + d[j, 0]) for consistency with the base heuristic logic.
                score = prizes[j] / (d[current, j] + d[j, 0] + 1e-9)
                if score > best_score:
                    best_score = score
                    best_node = j
        
        if best_node == -1:
            break
        
        greedy_edges.append((current, best_node))
        visited[best_node] = True
        length_so_far = length_so_far + d[current, best_node]
        current = best_node
        
        # Check if we can still return to 0
        if length_so_far + d[current, 0] > max_len + 1e-9:
            # This shouldn't happen if we checked feasibility before moving, 
            # but if we are at a dead end, we stop.
            # Actually, the feasibility check `new_len <= max_len` ensures we can return.
            # So if we moved, we can return.
            pass

    # Boost the heuristic for edges in the greedy tour
    # The ACO uses heuristic[i, j]. We want to make H[i, j] larger for (i, j) in greedy_edges.
    # We multiply by a factor to make them more attractive.
    boost_factor = 1.5
    
    # Initialize H as a copy of H_base
    H = H_base.copy()
    
    # Apply boost
    for i, j in greedy_edges:
        H[i, j] *= boost_factor
    
    # Also, consider that the greedy tour might have missed some good connections.
    # We can also boost edges that are "short" and "high prize" generally.
    
    # Ensure positivity and handle non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    # Sharpen the heuristic to make ACO more exploitative
    # Raising to a power > 1 emphasizes larger values
    H = H ** 1.1
    
    # Final safety check
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
