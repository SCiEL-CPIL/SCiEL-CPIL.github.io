"""Different strategy: Or-opt + 2-opt with station-aware moves and time-window feasibility checks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

DEADLINE_SECONDS = 115.0

def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def _compute_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = _euclidean(coords[i], coords[j])
    return D

def _check_route_feasibility(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, capacity):
    """
    Check if a route (list of node IDs, no depot) is feasible.
    Returns (is_feasible, total_distance).
    """
    if not route:
        return True, 0.0
    
    total_demand = sum(demands[v] for v in route if not is_station[v])
    if total_demand > capacity + 1e-9:
        return False, 0.0
    
    t = 0.0
    b = battery_max
    total_dist = 0.0
    
    prev = 0
    for node in route:
        dist = D[prev, node]
        total_dist += dist
        b -= dist
        if b < -1e-9:
            return False, 0.0
        t += dist / speed
        
        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            # Customer
            start_service = max(t, ready[node])
            if start_service > due[node] + 1e-9:
                return False, 0.0
            t = start_service + service[node]
        
        prev = node
    
    # Return to depot
    dist_back = D[prev, 0]
    total_dist += dist_back
    b -= dist_back
    if b < -1e-9:
        return False, 0.0
    t += dist_back / speed
    
    if t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _construct_greedy(instance, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, capacity):
    """
    Construct initial routes using a greedy nearest-neighbor approach.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    routes = []
    unvisited = set(customers)
    
    # Sort customers by due date for better feasibility
    customers_sorted = sorted(customers, key=lambda x: due[x])
    
    for cust in customers_sorted:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            # Try all positions
            for pos in range(len(route) + 1):
                prev = 0 if pos == 0 else route[pos - 1]
                next_node = 0 if pos == len(route) else route[pos]
                
                cost = D[prev, cust] + D[cust, next_node] - D[prev, next_node]
                
                if cost < best_cost:
                    # Check feasibility
                    new_route = route[:pos] + [cust] + route[pos:]
                    feasible, _ = _check_route_feasibility(
                        new_route, D, ready, due, service, demands, is_station,
                        battery_max, recharge_rate, speed, horizon, capacity
                    )
                    if feasible:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
        
        if best_route_idx == -1:
            # Create new route
            new_route = [cust]
            feasible, _ = _check_route_feasibility(
                new_route, D, ready, due, service, demands, is_station,
                battery_max, recharge_rate, speed, horizon, capacity
            )
            if feasible:
                routes.append(new_route)
        else:
            route = routes[best_route_idx]
            routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]
        
        unvisited.discard(cust)
    
    # Add any remaining customers as single-node routes
    for cust in unvisited:
        new_route = [cust]
        feasible, _ = _check_route_feasibility(
            new_route, D, ready, due, service, demands, is_station,
            battery_max, recharge_rate, speed, horizon, capacity
        )
        if feasible:
            routes.append(new_route)
        else:
            routes.append(new_route)  # Force add, may be infeasible
    
    return routes

def _improve_with_2opt(routes, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, capacity, deadline):
    """
    Improve routes using 2-opt local search.
    """
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            best_saving = 0
            best_i, best_j = -1, -1
            best_new_route = None
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    prev_i = 0 if i == 0 else route[i - 1]
                    next_j = 0 if j == n - 1 else route[j + 1]
                    
                    old_dist = D[prev_i, route[i]] + D[route[j], next_j]
                    new_dist = D[prev_i, route[j]] + D[route[i], next_j]
                    
                    saving = old_dist - new_dist
                    if saving > best_saving:
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        
                        feasible, _ = _check_route_feasibility(
                            new_route, D, ready, due, service, demands, is_station,
                            battery_max, recharge_rate, speed, horizon, capacity
                        )
                        if feasible:
                            best_saving = saving
                            best_i, best_j = i, j
                            best_new_route = new_route
            
            if best_i != -1:
                routes[r_idx] = best_new_route
                improved = True
    
    return routes

def _improve_with_oropt(routes, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, capacity, deadline):
    """
    Improve routes using Or-opt local search (move 1-3 consecutive nodes).
    """
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            best_saving = 0
            best_move = None  # (i, j, new_pos)
            
            # Try moving 1, 2, or 3 consecutive nodes
            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len + 1):
                    segment = route[i:i + seg_len]
                    for new_pos in range(n - seg_len + 1):
                        if new_pos == i:
                            continue
                        # Skip if overlapping
                        if i < new_pos < i + seg_len:
                            continue
                        
                        # Calculate cost delta
                        # Old: D[prev_i, route[i]] + D[route[i+seg_len-1], next_i]
                        # New: D[prev_new, segment[0]] + D[segment[-1], next_new]
                        # Need to handle depot boundaries carefully
                        
                        prev_i = 0 if i == 0 else route[i - 1]
                        next_i = 0 if i + seg_len == n else route[i + seg_len]
                        
                        # Determine new position context
                        if new_pos > i:
                            new_pos -= seg_len  # Adjust for removal
                        prev_new = 0 if new_pos == 0 else route[new_pos - 1]
                        next_new = 0 if new_pos == n - seg_len else route[new_pos]
                        
                        old_dist = D[prev_i, route[i]] + D[route[i + seg_len - 1], next_i]
                        new_dist = D[prev_new, segment[0]] + D[segment[-1], next_new]
                        
                        saving = old_dist - new_dist
                        if saving > best_saving:
                            # Build new route
                            remaining = route[:i] + route[i + seg_len:]
                            new_route = remaining[:new_pos] + segment + remaining[new_pos:]
                            
                            feasible, _ = _check_route_feasibility(
                                new_route, D, ready, due, service, demands, is_station,
                                battery_max, recharge_rate, speed, horizon, capacity
                            )
                            if feasible:
                                best_saving = saving
                                best_move = (i, seg_len, new_pos)
            
            if best_move:
                i, seg_len, new_pos = best_move
                segment = route[i:i + seg_len]
                remaining = route[:i] + route[i + seg_len:]
                if new_pos > i:
                    new_pos -= seg_len
                routes[r_idx] = remaining[:new_pos] + segment + remaining[new_pos:]
                improved = True
    
    return routes

def _try_add_stations(routes, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, capacity, stations, deadline):
    """
    Try to add charging stations to routes if needed for battery feasibility.
    """
    for r_idx in range(len(routes)):
        route = routes[r_idx]
        # Check if route is already feasible
        feasible, _ = _check_route_feasibility(
            route, D, ready, due, service, demands, is_station,
            battery_max, recharge_rate, speed, horizon, capacity
        )
        if feasible:
            continue
        
        # Try inserting a station at various positions
        best_saving = float('inf')
        best_station = None
        best_pos = -1
        
        for station in stations:
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [station] + route[pos:]
                new_feasible, new_dist = _check_route_feasibility(
                    new_route, D, ready, due, service, demands, is_station,
                    battery_max, recharge_rate, speed, horizon, capacity
                )
                if new_feasible:
                    if new_dist < best_saving:
                        best_saving = new_dist
                        best_station = station
                        best_pos = pos
        
        if best_station is not None:
            routes[r_idx] = routes[r_idx][:best_pos] + [best_station] + routes[r_idx][best_pos:]
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # 1. Precompute Data
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = list(instance["stations"])
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Distance Matrix
    D = _compute_distance_matrix(coords)
    
    # 2. Construction: Greedy Cheapest Insertion
    routes = _construct_greedy(instance, D, ready, due, service, demands, is_station,
                               battery_max, recharge_rate, speed, horizon, capacity)
    
    # 3. Add stations if needed for battery
    routes = _try_add_stations(routes, D, ready, due, service, demands, is_station,
                               battery_max, recharge_rate, speed, horizon, capacity, stations, deadline)
    
    # 4. Local Search: 2-Opt
    routes = _improve_with_2opt(routes, D, ready, due, service, demands, is_station,
                                battery_max, recharge_rate, speed, horizon, capacity, deadline)
    
    # 5. Local Search: Or-opt
    routes = _improve_with_oropt(routes, D, ready, due, service, demands, is_station,
                                 battery_max, recharge_rate, speed, horizon, capacity, deadline)
    
    # 6. Final validation and cleanup
    # Ensure all customers are served
    visited = set()
    for r in routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)
    
    missing = set(range(1, n_customers + 1)) - visited
    for m in missing:
        routes.append([m])
    
    # Remove empty routes
    routes = [r for r in routes if len(r) > 0]
    
    return routes

# EVOLVE-BLOCK-END
