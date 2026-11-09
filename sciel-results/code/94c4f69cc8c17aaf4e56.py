import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    n = distance_matrix.shape[0]
    
    # Helper to calculate tour length
    def tour_length(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        return total

    # Helper for 2-Opt
    def two_opt(tour):
        tour = list(tour)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a = tour[i]
                    b = tour[i + 1]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    
                    # Avoid self loops or immediate reversals that don't change
                    # dist is symmetric
                    
                    old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                    new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                    
                    if new_cost < old_cost - 1e-10:
                        # Reverse segment between i+1 and j
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        improved = True
        return tour

    # Helper for Or-Opt (1-node and 2-node moves)
    def or_opt(tour):
        tour = list(tour)
        improved = True
        while improved:
            if time.time() > deadline - 15:
                break
            improved = False
            
            # Try 1-node moves
            for i in range(n):
                node = tour[i]
                prev_node = tour[i-1]
                next_node = tour[(i+1) % n]
                
                # Try inserting node at position j
                for j in range(n):
                    if j == i or j == (i-1) % n or j == (i+1) % n:
                        continue
                    
                    old_cost = distance_matrix[prev_node, node] + distance_matrix[node, next_node]
                    new_cost = distance_matrix[prev_node, next_node] + distance_matrix[tour[j-1], node] + distance_matrix[node, tour[j]] - distance_matrix[tour[j-1], tour[j]]
                    
                    if new_cost < old_cost - 1e-10:
                        tour.pop(i)
                        if j > i:
                            j -= 1
                        tour.insert(j, node)
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-node moves if no 1-node improvement
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 15:
                        break
                    node1 = tour[i]
                    node2 = tour[(i+1) % n]
                    prev_node = tour[i-1]
                    next_node = tour[(i+2) % n]
                    
                    # Try moving both nodes together to position j
                    for j in range(n):
                        if j in [i, (i+1) % n, (i-1) % n, (i+2) % n]:
                            continue
                        
                        # Check if inserting at j would create a valid move
                        # The segment [node1, node2] is removed and inserted before tour[j]
                        j_prev = tour[j-1]
                        j_next = tour[j]
                        
                        old_cost = (distance_matrix[prev_node, node1] + distance_matrix[node1, node2] + 
                                   distance_matrix[node2, next_node])
                        new_cost = (distance_matrix[prev_node, next_node] + distance_matrix[j_prev, node1] + 
                                   distance_matrix[node1, node2] + distance_matrix[node2, j_next] - 
                                   distance_matrix[j_prev, j_next])
                        
                        if new_cost < old_cost - 1e-10:
                            # Remove both nodes
                            tour.pop((i+1) % n)  # Remove node2 first (higher index)
                            tour.pop(i)           # Remove node1
                            # Adjust j if j > i
                            if j > i:
                                j -= 2
                            elif j == i:
                                j -= 1
                            # Insert both nodes before position j
                            tour.insert(j, node1)
                            tour.insert(j+1, node2)
                            improved = True
                            break
                    if improved:
                        break
        return tour

    # Generate initial tours
    best_tour = None
    best_len = float('inf')
    
    # 1. Nearest Neighbor
    visited = [False] * n
    visited[0] = True
    nn_tour = [0]
    current = 0
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for k in range(n):
            if not visited[k]:
                d = distance_matrix[current, k]
                if d < min_dist:
                    min_dist = d
                    next_node = k
        nn_tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    improved_nn = two_opt(nn_tour)
    len_nn = tour_length(improved_nn)
    if len_nn < best_len:
        best_len = len_nn
        best_tour = improved_nn

    # 2. Farthest Insertion
    # Start with 0 and farthest point
    farthest_node = 0
    max_d = -1
    for k in range(1, n):
        if distance_matrix[0, k] > max_d:
            max_d = distance_matrix[0, k]
            farthest_node = k
            
    fi_tour = [0, farthest_node]
    remaining = list(range(1, n))
    remaining.remove(farthest_node)
    
    while remaining:
        # Find point in remaining that is farthest from the current tour
        max_min_dist = -1
        point_to_insert = -1
        for k in remaining:
            min_dist_to_tour = float('inf')
            for node_in_tour in fi_tour:
                d = distance_matrix[k, node_in_tour]
                if d < min_dist_to_tour:
                    min_dist_to_tour = d
            if min_dist_to_tour > max_min_dist:
                max_min_dist = min_dist_to_tour
                point_to_insert = k
        
        # Insert point_to_insert into fi_tour at the position that minimizes increase
        best_pos = -1
        best_increase = float('inf')
        for i in range(len(fi_tour)):
            prev_node = fi_tour[i]
            next_node = fi_tour[(i + 1) % len(fi_tour)]
            increase = distance_matrix[prev_node, point_to_insert] + distance_matrix[point_to_insert, next_node] - distance_matrix[prev_node, next_node]
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1 # Insert after i
        
        fi_tour.insert(best_pos, point_to_insert)
        remaining.remove(point_to_insert)
        
    improved_fi = two_opt(fi_tour)
    len_fi = tour_length(improved_fi)
    if len_fi < best_len:
        best_len = len_fi
        best_tour = improved_fi

    # 3. Random Restarts
    import random
    # Use a fixed seed for reproducibility? The task says "seeded per instance" for the ACO,
    # but our heuristic is computed once. We should be deterministic or use a fixed seed for our search.
    # Let's use a fixed seed for our internal search to ensure consistent heuristic.
    random.seed(42)
    
    # Use remaining time for more restarts
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() > deadline - 15:  # Leave 15s for final computation
            break
        rand_tour = list(range(n))
        random.shuffle(rand_tour)
        # Ensure it starts at 0? ACO starts at 0, but the tour is a cycle.
        # We can rotate the tour to start at 0 for consistency, though length is invariant.
        if rand_tour[0] != 0:
            idx = rand_tour.index(0)
            rand_tour = rand_tour[idx:] + rand_tour[:idx]
            
        improved_rand = two_opt(rand_tour)
        improved_rand = or_opt(improved_rand)
        len_rand = tour_length(improved_rand)
        if len_rand < best_len:
            best_len = len_rand
            best_tour = improved_rand

    # Construct Heuristic
    # Base heuristic: inverse distance
    eta = 1.0 / (distance_matrix + 1e-9)
    
    # Add Hungarian reduced-cost signal as a secondary gradient
    # The Hungarian algorithm finds the optimal assignment (perfect matching),
    # and the reduced costs indicate how much each edge "regrets" being in the solution.
    # Edges with lower reduced cost are more likely to be in a good tour.
    if time.time() < deadline - 10:
        # Solve the assignment problem (minimize total cost)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        # Compute reduced costs: for each edge (i,j), how much more expensive is it
        # compared to the optimal assignment?
        # We can compute this by looking at the dual variables or by comparing
        # to the optimal tour structure.
        
        # A simpler approach: use the optimal assignment edges as a strong signal
        # and penalize edges that are far from being in the optimal assignment.
        
        # Create a matrix of assignment costs
        assignment_cost = np.full((n, n), np.inf)
        for r, c in zip(row_ind, col_ind):
            assignment_cost[r, c] = distance_matrix[r, c]
        
        # For each edge, compute its "regret" relative to the optimal assignment
        # An edge (i,j) has low regret if it's in the optimal assignment
        # We use a soft measure: how much worse is this edge compared to the best
        # edge from i and to j in the optimal assignment
        
        # For each row i, find the optimal assignment edge
        opt_col = np.full(n, -1)
        for r, c in zip(row_ind, col_ind):
            opt_col[r] = c
        
        # For each col j, find the optimal assignment row
        opt_row = np.full(n, -1)
        for r, c in zip(row_ind, col_ind):
            opt_row[c] = r
        
        # Compute regret: for edge (i,j), how much more expensive is it compared to
        # the optimal edge from i (which goes to opt_col[i]) and to j (which comes from opt_row[j])
        regret = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    regret[i, j] = 0
                    continue
                # Cost of this edge vs optimal edge from i
                regret_from_i = distance_matrix[i, j] - distance_matrix[i, opt_col[i]]
                # Cost of this edge vs optimal edge to j
                regret_to_j = distance_matrix[i, j] - distance_matrix[opt_row[j], j]
                # Take the minimum regret (most favorable comparison)
                regret[i, j] = max(0, min(regret_from_i, regret_to_j))
        
        # Convert regret to attractiveness: lower regret -> higher attractiveness
        # Use exponential decay to smooth the signal
        regret_scale = 2.0  # Scale factor for the exponential
        regret_attractiveness = np.exp(-regret_scale * regret)
        
        # Combine: base inverse distance modulated by regret signal
        # Edges in the optimal assignment get a boost, edges with high regret get dampened
        eta = eta * (1.0 + 2.0 * regret_attractiveness)
    
    # Boost edges in best_tour with distance-aware scaling
    # Shorter edges in the tour get higher boost to reinforce local structure
    # Longer edges get lower boost but still strong signal
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        edge_len = distance_matrix[u, v]
        # Scale boost inversely with edge length: shorter edges get much higher boost
        # This encodes that the tour's local geometry is important
        boost = 5000.0 / (1.0 + 5.0 * edge_len)
        eta[u, v] = max(eta[u, v], boost)
    
    # Add explicit penalty for very long non-tour edges
    # This prevents ACO ants from making suboptimal long-distance jumps
    # Find the 95th percentile of edge lengths as a threshold
    edge_lengths = distance_matrix[~np.eye(n, dtype=bool)]
    threshold = np.percentile(edge_lengths, 95)
    
    # For edges that are very long and NOT in the best tour, reduce their attractiveness
    tour_edge_set = set()
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        tour_edge_set.add((u, v))
        tour_edge_set.add((v, u))
    
    # Apply penalty to long non-tour edges
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) in tour_edge_set:
                continue
            if distance_matrix[i, j] > threshold:
                # Scale down attractiveness for long non-tour edges
                # The longer the edge, the more we penalize
                penalty_factor = 1.0 / (1.0 + 3.0 * (distance_matrix[i, j] - threshold) / threshold)
                eta[i, j] *= penalty_factor
    
    # Ensure no negative or zero values (ACO requires positive heuristics)
    eta = np.maximum(eta, 1e-9)
    
    return eta
