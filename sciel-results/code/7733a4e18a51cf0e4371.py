"""EV Routing: Cheapest Insertion + 2-Opt with Fast Prefix Validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

DEADLINE_SECONDS = 118.0

def _compute_prefixes(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon):
    """
    Compute prefix arrays for a route.
    Returns:
        p_time: arrival time at each node (before service/wait)
        p_battery: battery level at each node (before service/refill)
        p_demand: cumulative demand
        final_time: time at depot after return
        final_battery: battery at depot after return
        is_feasible: bool
    """
    n = len(route)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), 0.0, battery_max, True
    
    p_time = np.zeros(n)
    p_battery = np.zeros(n)
    p_demand = np.zeros(n)
    
    t = 0.0
    b = battery_max
    d_sum = 0.0
    prev = 0
    
    feasible = True
    
    for i, node in enumerate(route):
        dist = D[prev, node]
        t += dist / speed
        b -= dist
        
        if b < -1e-9:
            feasible = False
            # Continue to fill arrays but mark infeasible
            # We can break early if we want, but let's fill for potential debugging
            # Actually, if infeasible, we don't really care about the rest for optimization,
            # but we need to return something.
            # For speed, if infeasible, we can just return the current state and False.
            # But the caller might use the arrays. Let's just break and return infeasible.
            # However, for 2-opt, we need to know *where* it fails? No, just if it's feasible.
            # So we can break early.
            return p_time[:i+1], p_battery[:i+1], p_demand[:i+1], t, b, False
        
        p_time[i] = t
        p_battery[i] = b
        d_sum += demands[node]
        p_demand[i] = d_sum
        
        if d_sum > 100: # Capacity
            feasible = False
            return p_time[:i+1], p_battery[:i+1], p_demand[:i+1], t, b, False
        
        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            if demands[node] > 0:
                start_service = max(t, ready[node])
                if start_service > due[node] + 1e-9:
                    feasible = False
                    return p_time[:i+1], p_battery[:i+1], p_demand[:i+1], t, b, False
                t = start_service + service[node]
        
        prev = node
    
    # Return to depot
    dist_back = D[prev, 0]
    t += dist_back / speed
    b -= dist_back
    
    if b < -1e-9 or t > horizon + 1e-9:
        feasible = False
    
    return p_time, p_battery, p_demand, t, b, feasible

def _validate_route(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon):
    """Full validation returning distance and feasibility."""
    if not route:
        return 0.0, True
    
    p_time, p_battery, p_demand, final_time, final_battery, feasible = _compute_prefixes(
        route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon
    )
    
    if not feasible:
        # Still compute distance for scoring if we want, but usually we just return inf
        # Let's compute distance anyway
        dist = 0.0
        prev = 0
        for node in route:
            dist += D[prev, node]
            prev = node
        dist += D[prev, 0]
        return dist, False
    
    dist = 0.0
    prev = 0
    for node in route:
        dist += D[prev, node]
        prev = node
    dist += D[prev, 0]
    
    return dist, True

def _try_insert_station_at(route, pos, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list):
    """
    Try to insert a station at position 'pos' to fix infeasibility.
    Returns new route if feasible, else None.
    """
    # Simple heuristic: try inserting each station at pos
    # Check if it makes the route feasible
    best_route = None
    best_cost = float('inf')
    
    for s in stations_list:
        if s in route:
            # Avoid duplicate stations if possible, or allow?
            # The problem says "stations may repeat".
            pass
        
        new_route = route[:pos] + [s] + route[pos:]
        dist, feas = _validate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
        if feas and dist < best_cost:
            best_cost = dist
            best_route = new_route
            
    return best_route

def plan(instance):
    start_time = time.time()
    
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
    stations = instance["stations"]
    stations_list = list(stations)
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Distance Matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    unvisited = set(customers)
    
    # 2. Construction: Cheapest Insertion
    routes = []
    route_costs = []
    
    # Start with the most constrained customer? Or just pick one.
    # Let's pick the customer with the smallest due date or largest demand?
    # Simple: pick first customer.
    first_c = customers[0]
    routes.append([first_c])
    unvisited.remove(first_c)
    dist, feas = _validate_route(routes[0], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
    if not feas:
        # Try to fix with station
        fixed = _try_insert_station_at(routes[0], 1, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
        if fixed:
            routes[0] = fixed
            dist, feas = _validate_route(routes[0], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
    route_costs.append(dist if feas else float('inf'))
    
    # Insert remaining customers
    while unvisited:
        if time.time() - start_time > DEADLINE_SECONDS * 0.1:
            break
            
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_customer = -1
        
        # Sort unvisited by some heuristic? Or just iterate.
        # To speed up, we can pre-filter customers that don't fit in any route by capacity/demand
        # But with 25 customers, O(N^2 * R) is fine.
        
        for c in list(unvisited):
            # Check if c can be added to any existing route
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                
                # Capacity check first
                # We need cumulative demand.
                # Let's just try all positions.
                for pos in range(n + 1):
                    # Quick check: does adding c at pos keep demand <= capacity?
                    # We don't have prefix demand readily available without recomputing.
                    # But we can compute it quickly.
                    # Actually, let's just construct the new route and validate.
                    # Optimization: if current demand + demand[c] > capacity, skip.
                    # We can store total demand per route.
                    # Let's add a list route_demands.
                    pass
                
        # Refine: Store route_demands
        # Let's restart construction with better bookkeeping.
        # This is getting messy. Let's use a simpler, robust insertion.
        
        # Restart construction logic with cleaner state.
        break # Break out of this messy loop and do it properly below.
        
    # --- Proper Cheapest Insertion ---
    routes = []
    route_costs = []
    route_demands = []
    
    # Initial route with first customer
    first_c = customers[0]
    routes.append([first_c])
    route_demands.append(demands[first_c])
    unvisited.remove(first_c)
    dist, feas = _validate_route(routes[0], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
    if not feas:
        fixed = _try_insert_station_at(routes[0], 1, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
        if fixed:
            routes[0] = fixed
            dist, feas = _validate_route(routes[0], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
    route_costs.append(dist if feas else float('inf'))
    
    while unvisited:
        if time.time() - start_time > DEADLINE_SECONDS * 0.15:
            break
            
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_customer = -1
        
        # Iterate customers
        for c in list(unvisited):
            # Iterate routes
            for r_idx in range(len(routes)):
                # Capacity check
                if route_demands[r_idx] + demands[c] > capacity:
                    continue
                    
                route = routes[r_idx]
                n = len(route)
                
                # Try all positions
                for pos in range(n + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    dist, feas = _validate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    if feas:
                        if dist < best_cost:
                            best_cost = dist
                            best_route_idx = r_idx
                            best_pos = pos
                            best_customer = c
                            
        if best_customer != -1:
            # Insert
            route = routes[best_route_idx]
            new_route = route[:best_pos] + [best_customer] + route[best_pos:]
            routes[best_route_idx] = new_route
            route_costs[best_route_idx] = best_cost
            route_demands[best_route_idx] += demands[best_customer]
            unvisited.remove(best_customer)
        else:
            # Couldn't insert anywhere. Start new route.
            # Try to find a feasible single-node route for best_customer?
            # If no customer can be inserted, pick the one with smallest due date?
            # Or just pick any.
            c = min(unvisited, key=lambda x: due[x])
            new_route = [c]
            dist, feas = _validate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            if not feas:
                fixed = _try_insert_station_at(new_route, 1, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                if fixed:
                    new_route = fixed
                    dist, feas = _validate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            routes.append(new_route)
            route_costs.append(dist if feas else float('inf'))
            route_demands.append(demands[c])
            unvisited.remove(c)
            
    # 3. Local Search: 2-Opt (Intra-route and Inter-route)
    # We will do a few passes.
    # To keep it fast, we will use the prefix validation for quick checks where possible,
    # but 2-opt requires re-evaluating the whole route.
    # Given N=25, routes are short.
    
    improved = True
    passes = 0
    max_passes = 20
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() - start_time > DEADLINE_SECONDS * 0.8:
            break
            
        # Intra-route 2-Opt
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            best_saving = 0
            best_i, best_j = -1, -1
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    # New route: route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # Check feasibility and cost
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    dist, feas = _validate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    if feas:
                        saving = route_costs[r_idx] - dist
                        if saving > best_saving:
                            best_saving = saving
                            best_i, best_j = i, j
                            
            if best_i != -1:
                route = routes[r_idx]
                new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[r_idx] = new_route
                route_costs[r_idx] -= best_saving
                improved = True
                
        # Inter-route 2-Opt (Or-Opt style: move segment from one route to another)
        # For simplicity and speed, let's do 1-node moves (Or-Opt with len=1) first.
        # Move customer c from route A to position p in route B.
        
        for r_idx in range(len(routes)):
            if time.time() - start_time > DEADLINE_SECONDS * 0.9:
                break
            route = routes[r_idx]
            if len(route) < 2:
                continue
            
            for i in range(len(route)):
                c = route[i]
                # Remove c
                new_route_r = route[:i] + route[i+1:]
                
                # Check if new_route_r is feasible
                if len(new_route_r) > 0:
                    dist_r, feas_r = _validate_route(new_route_r, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    if not feas_r:
                        continue
                else:
                    dist_r = 0.0
                    feas_r = True
                    
                # Try inserting c into other routes
                for t_idx in range(len(routes)):
                    if t_idx == r_idx:
                        continue
                        
                    target_route = routes[t_idx]
                    t_n = len(target_route)
                    
                    # Capacity check
                    if route_demands[t_idx] + demands[c] > capacity:
                        continue
                        
                    for pos in range(t_n + 1):
                        new_route_t = target_route[:pos] + [c] + target_route[pos:]
                        dist_t, feas_t = _validate_route(new_route_t, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                        if not feas_t:
                            continue
                            
                        old_total = route_costs[r_idx] + route_costs[t_idx]
                        new_total = dist_r + dist_t
                        saving = old_total - new_total
                        
                        if saving > 1e-6:
                            # Apply move
                            routes[r_idx] = new_route_r
                            routes[t_idx] = new_route_t
                            route_costs[r_idx] = dist_r
                            route_costs[t_idx] = dist_t
                            route_demands[r_idx] -= demands[c]
                            route_demands[t_idx] += demands[c]
                            improved = True
                            break # Restart loop for this customer? Or just break inner loops.
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Remove empty routes
        if improved:
            new_routes = []
            new_costs = []
            new_demands = []
            for i in range(len(routes)):
                if routes[i]:
                    new_routes.append(routes[i])
                    new_costs.append(route_costs[i])
                    new_demands.append(route_demands[i])
            routes = new_routes
            route_costs = new_costs
            route_demands = new_demands
            
    # 4. Finalize
    final_routes = [r for r in routes if r]
    
    # Ensure all customers are present
    visited = set()
    for r in final_routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)
                
    missing = set(customers) - visited
    for m in missing:
        final_routes.append([m])
        
    return final_routes

# EVOLVE-BLOCK-END
