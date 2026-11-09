"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _two_opt(route, dist):
    """Improve a route (list of node indices, no depot) using 2-opt."""
    if len(route) < 3:
        return route
    improved = True
    best = route[:]
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # 2-opt reversal: ... i, i+1, ..., j, j+1 ...
                # Remove edges (i-1,i) and (j,j+1), add (i-1,j) and (i,j+1)
                # But we need to handle depot connections
                # Route is: depot, best[0], best[1], ..., best[-1], depot
                a = best[i-1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j+1] if j+1 < len(best) else 0
                # Current: dist[a,b] + dist[c,d]
                # New: dist[a,c] + dist[b,d]
                if dist[a][b] + dist[c][d] > dist[a][c] + dist[b][d] + 1e-10:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best

def _savings_heuristic(dist, demands, capacity, n):
    """Clarke-Wright savings heuristic with 2-opt polish."""
    # Compute savings
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i+1, n):
            savings[i][j] = dist[0][i] + dist[0][j] - dist[i][j]
    
    # Sort savings in descending order
    pairs = []
    for i in range(1, n):
        for j in range(i+1, n):
            pairs.append((savings[i][j], i, j))
    pairs.sort(reverse=True)
    
    # Union-Find for tracking route membership
    parent = list(range(n))
    rank = [0] * n
    route_load = np.zeros(n)
    for i in range(1, n):
        route_load[i] = demands[i]
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        route_load[rx] += route_load[ry]
        return True
    
    # Build routes
    routes = {}
    for s, i, j in pairs:
        if s <= 0:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if route_load[ri] + route_load[rj] > capacity + 1e-9:
            continue
        union(ri, rj)
    
    # Collect routes
    for i in range(1, n):
        root = find(i)
        if root not in routes:
            routes[root] = []
        routes[root].append(i)
    
    # Ensure all customers are assigned
    assigned = set()
    for root, custs in routes.items():
        assigned.update(custs)
    for i in range(1, n):
        if i not in assigned:
            routes[i] = [i]
    
    # Apply 2-opt to each route
    polished_routes = []
    for root, custs in routes.items():
        polished = _two_opt(custs, dist)
        polished_routes.append(polished)
    
    return polished_routes

def _savings_with_randomization(dist, demands, capacity, n, rng, alpha=0.5):
    """Savings heuristic with randomization for diversity."""
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i+1, n):
            savings[i][j] = dist[0][i] + dist[0][j] - dist[i][j]
    
    # Add noise
    noise = rng.random((n, n)) * alpha * np.max(savings)
    savings_noisy = savings + noise
    
    pairs = []
    for i in range(1, n):
        for j in range(i+1, n):
            pairs.append((savings_noisy[i][j], i, j))
    pairs.sort(reverse=True)
    
    parent = list(range(n))
    rank = [0] * n
    route_load = np.zeros(n)
    for i in range(1, n):
        route_load[i] = demands[i]
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        route_load[rx] += route_load[ry]
        return True
    
    for s, i, j in pairs:
        if s <= 0:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if route_load[ri] + route_load[rj] > capacity + 1e-9:
            continue
        union(ri, rj)
    
    routes = {}
    for i in range(1, n):
        root = find(i)
        if root not in routes:
            routes[root] = []
        routes[root].append(i)
    
    assigned = set()
    for root, custs in routes.items():
        assigned.update(custs)
    for i in range(1, n):
        if i not in assigned:
            routes[i] = [i]
    
    polished_routes = []
    for root, custs in routes.items():
        polished = _two_opt(custs, dist)
        polished_routes.append(polished)
    
    return polished_routes

def _route_cost(route, dist):
    """Compute cost of a route (depot -> route -> depot)."""
    if not route:
        return 0.0
    cost = dist[0][route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i+1]]
    cost += dist[route[-1]][0]
    return cost

def _inter_route_2opt(routes, dist, demands, capacity, n):
    """Try to improve by moving nodes between routes."""
    improved = True
    max_iter = 50
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for r1 in range(len(routes)):
            for r2 in range(r1+1, len(routes)):
                for i in range(len(routes[r1])):
                    node = routes[r1][i]
                    # Try moving node to route r2
                    if demands[node] > capacity - sum(demands[node] for node in routes[r2]) + 1e-9:
                        continue
                    # Try inserting at each position in r2
                    best_pos = -1
                    best_cost_diff = 0.0
                    # Cost of removing node from r1
                    prev = routes[r1][i-1] if i > 0 else 0
                    next_ = routes[r1][i+1] if i+1 < len(routes[r1]) else 0
                    remove_cost = dist[prev][node] + dist[node][next_] - dist[prev][next_]
                    
                    for pos in range(len(routes[r2]) + 1):
                        p = routes[r2][pos-1] if pos > 0 else 0
                        q = routes[r2][pos] if pos < len(routes[r2]) else 0
                        insert_cost = dist[p][q] - dist[p][node] - dist[node][q]
                        total_diff = remove_cost + insert_cost
                        if total_diff < best_cost_diff - 1e-10:
                            best_cost_diff = total_diff
                            best_pos = pos
                    
                    if best_pos >= 0 and best_cost_diff < -1e-10:
                        routes[r1].pop(i)
                        routes[r2].insert(best_pos, node)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    The heuristic combines:
    1. Inverse distance (short edges are more attractive)
    2. Edge frequency from multiple diversified savings-based solutions
    3. Demand compatibility (edges between customers with compatible demands)
    """
    start_time = time.time()
    deadline = 250.0  # Safety margin before 300s limit
    
    dist = np.asarray(distance_matrix, dtype=np.float64)
    n = dist.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Edge frequency matrix from multiple diversified solutions
    edge_freq = np.zeros((n, n))
    
    # Run multiple diversified savings heuristics
    n_solutions = 15
    rng = np.random.RandomState(42)
    
    for k in range(n_solutions):
        if time.time() > deadline - 30:
            break
        if k == 0:
            routes = _savings_heuristic(dist, demands, capacity, n)
        else:
            alpha = 0.3 + 0.4 * (k / n_solutions)
            routes = _savings_with_randomization(dist, demands, capacity, n, rng, alpha=alpha)
        
        # Inter-route 2-opt
        if time.time() > deadline - 20:
            pass
        else:
            routes = _inter_route_2opt(routes, dist, demands, capacity, n)
        
        # Count edges
        for route in routes:
            if not route:
                continue
            # Depot to first
            edge_freq[0][route[0]] += 1
            edge_freq[route[0]][0] += 1
            # Between consecutive
            for i in range(len(route) - 1):
                edge_freq[route[i]][route[i+1]] += 1
                edge_freq[route[i+1]][route[i]] += 1
            # Last to depot
            edge_freq[route[-1]][0] += 1
            edge_freq[0][route[-1]] += 1
    
    # Normalize edge frequency
    max_freq = np.max(edge_freq)
    if max_freq > 0:
        edge_freq_norm = edge_freq / max_freq
    else:
        edge_freq_norm = np.ones((n, n))
    
    # Inverse distance component
    inv_dist = 1.0 / (dist + 1e-9)
    max_inv_dist = np.max(inv_dist)
    inv_dist_norm = inv_dist / max_inv_dist
    
    # Demand compatibility: penalize edges between high-demand customers
    # A customer with demand d can share a route with another if d1 + d2 <= capacity
    # So edges between customers with d1 + d2 > capacity should be penalized
    demand_compat = np.ones((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            if demands[i] + demands[j] > capacity:
                demand_compat[i][j] = 0.1
            else:
                # Scale by how much capacity is left
                remaining = capacity - demands[i] - demands[j]
                demand_compat[i][j] = 0.5 + 0.5 * (remaining / capacity)
    
    # Combine components
    # Weight: edge frequency (0.4), inverse distance (0.4), demand compat (0.2)
    w_freq = 0.4
    w_dist = 0.4
    w_demand = 0.2
    
    heuristic = (w_freq * edge_freq_norm + w_dist * inv_dist_norm + w_demand * demand_compat)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Ensure correct shape
    if heuristic.shape != dist.shape:
        heuristic = np.ones_like(dist)
    
    # Ensure finite
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.ones_like(dist)
    
    return heuristic
# EVOLVE-BLOCK-END
