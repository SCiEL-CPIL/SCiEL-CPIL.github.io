"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _build_dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _fast_check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Fast feasibility check with early exits. Returns (feasible, total_distance)."""
    if not route:
        return True, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed
        
        if battery < -1e-9:
            return False, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed
    
    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist


def _quick_feasibility(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Very fast feasibility check - only checks battery and capacity, not time windows."""
    if not route:
        return True, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    total_dist = 0.0
    prev = 0
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0
        
        if v > n_customers:
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0
    
    return True, total_dist


def _construct_initial_routes(instance, dist):
    """Construct initial routes using a savings-based heuristic with battery awareness."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    
    # Sort customers by angle from depot for initial routing
    customer_angles = []
    for c in range(1, n_customers + 1):
        angle = math.atan2(coords[c][1] - coords[0][1], coords[c][0] - coords[0][0])
        customer_angles.append((angle, c))
    customer_angles.sort()
    
    # Group customers into routes based on angle sectors
    routes = []
    current_route = []
    current_demand = 0
    current_dist = 0.0
    
    for angle, c in customer_angles:
        if not current_route:
            # Check if we can start a new route with this customer
            d_to_cust = dist[0][c]
            d_back = dist[c][0]
            if d_to_cust + d_back <= battery_max:
                current_route = [c]
                current_demand = demands[c]
                current_dist = d_to_cust + d_back
            else:
                # Need a station - check if any station is reachable
                for st in stations:
                    d_to_st = dist[0][st]
                    d_st_to_cust = dist[st][c]
                    d_cust_back = dist[c][0]
                    if d_to_st + d_st_to_cust + d_cust_back <= battery_max + 1e-9:
                        current_route = [st, c]
                        current_demand = demands[c]
                        current_dist = d_to_st + d_st_to_cust + d_cust_back
                        break
        else:
            # Try to add to current route
            d_add = dist[current_route[-1]][c] + dist[c][0] - dist[current_route[-1]][0]
            if current_demand + demands[c] <= capacity and current_dist + d_add <= battery_max:
                current_route.append(c)
                current_demand += demands[c]
                current_dist += d_add
            else:
                # Try inserting a station before the new customer
                inserted = False
                for st in stations:
                    d_to_st = dist[current_route[-1]][st]
                    d_st_to_cust = dist[st][c]
                    d_cust_back = dist[c][0]
                    d_old_back = dist[current_route[-1]][0]
                    if current_demand + demands[c] <= capacity:
                        if d_to_st + d_st_to_cust + d_cust_back <= battery_max + 1e-9:
                            current_route.append(st)
                            current_route.append(c)
                            current_demand += demands[c]
                            current_dist += d_to_st + d_st_to_cust + d_cust_back - d_old_back
                            inserted = True
                            break
                if not inserted:
                    # Start new route
                    routes.append(current_route)
                    current_route = [c]
                    current_demand = demands[c]
                    d_to_cust = dist[0][c]
                    d_back = dist[c][0]
                    current_dist = d_to_cust + d_back
                    # Check if we need a station
                    if d_to_cust + d_back > battery_max:
                        for st in stations:
                            d_to_st = dist[0][st]
                            d_st_to_cust = dist[st][c]
                            if d_to_st + d_st_to_cust + d_back <= battery_max + 1e-9:
                                current_route = [st, c]
                                current_dist = d_to_st + d_st_to_cust + d_back
                                break
    
    if current_route:
        routes.append(current_route)
    
    # Now try to merge routes using savings
    if len(routes) > 1:
        route_of = {}
        for idx, r in enumerate(routes):
            for c in r:
                route_of[c] = idx
        
        route_demand = [0] * len(routes)
        for idx, r in enumerate(routes):
            for c in r:
                if c <= n_customers:
                    route_demand[idx] += demands[c]
        
        # Compute savings between adjacent customers in different routes
        savings = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Find best pair of customers to connect
                best_s = -1
                best_pair = None
                for ci in routes[i]:
                    if ci > n_customers:
                        continue
                    for cj in routes[j]:
                        if cj > n_customers:
                            continue
                        # Check if they're at the ends of their routes
                        if ci == routes[i][0] or ci == routes[i][-1]:
                            if cj == routes[j][0] or cj == routes[j][-1]:
                                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                                if s > best_s:
                                    best_s = s
                                    best_pair = (ci, cj)
                if best_pair and best_s > 0:
                    savings.append((best_s, i, j, best_pair))
        
        savings.sort(reverse=True)
        
        for s, ri, rj, (ci, cj) in savings:
            if ri == rj or route_demand[ri] + route_demand[rj] > capacity:
                continue
            
            # Try merging
            best_new_route = None
            best_dist = float('inf')
            
            for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
                merged = order[0] + order[1]
                feasible, total_dist = _fast_check_route(merged, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible and total_dist < best_dist:
                    best_dist = total_dist
                    best_new_route = merged
            
            if best_new_route is not None:
                routes[ri] = best_new_route
                routes[rj] = []
                route_demand[ri] += route_demand[rj]
                route_demand[rj] = 0
                for c in best_new_route:
                    route_of[c] = ri
        
        routes = [r for r in routes if r]
    
    # Ensure all customers are served
    all_served = set()
    for r in routes:
        all_served.update(r)
    
    for c in range(1, n_customers + 1):
        if c not in all_served:
            inserted = False
            for idx, r in enumerate(routes):
                current_demand = sum(demands[v] for v in r if v <= n_customers)
                if current_demand + demands[c] > capacity:
                    continue
                best_pos = -1
                best_cost = float('inf')
                for pos in range(len(r) + 1):
                    new_route = r[:pos] + [c] + r[pos:]
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        if pos == 0:
                            inc = dist[0][c] + dist[c][r[0]] - dist[0][r[0]]
                        elif pos == len(r):
                            inc = dist[r[-1]][c] + dist[c][0] - dist[r[-1]][0]
                        else:
                            inc = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = pos
                if best_pos >= 0:
                    routes[idx] = r[:best_pos] + [c] + r[best_pos:]
                    inserted = True
                    break
            if not inserted:
                # Try with a station
                for st in stations:
                    for idx, r in enumerate(routes):
                        current_demand = sum(demands[v] for v in r if v <= n_customers)
                        if current_demand + demands[c] > capacity:
                            continue
                        # Try inserting station + customer
                        for pos in range(len(r) + 1):
                            new_route = r[:pos] + [st, c] + r[pos:]
                            feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                routes[idx] = new_route
                                inserted = True
                                break
                        if inserted:
                            break
                    if inserted:
                        break
                if not inserted:
                    routes.append([c])
    
    return routes


def _two_opt_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements with quick feasibility pre-check."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # Quick check first
                    quick_ok, _ = _quick_feasibility(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if quick_ok:
                        feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            route = new_route
                            improved = True
                            break
            if improved:
                break
    
    return route


def _relocate_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply relocate (move single node) improvements."""
    if len(route) < 3:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                return route
            node = route[i]
            rest = route[:i] + route[i+1:]
            
            for pos in range(len(rest) + 1):
                if time.time() > deadline:
                    return route
                if pos == i or (pos > i and pos <= i + 1):
                    continue
                
                new_route = rest[:pos] + [node] + rest[pos:]
                
                # Quick distance check
                if pos == 0:
                    if len(rest) > 0:
                        d_old = dist[0][rest[0]] + dist[rest[-1]][0]
                        d_new = dist[0][node] + dist[node][rest[0]] + dist[rest[-1]][0]
                    else:
                        d_old = 0
                        d_new = dist[0][node] + dist[node][0]
                elif pos == len(rest):
                    if len(rest) > 0:
                        d_old = dist[rest[-1]][0]
                        d_new = dist[rest[-1]][node] + dist[node][0]
                    else:
                        d_old = 0
                        d_new = dist[0][node] + dist[node][0]
                else:
                    if len(rest) > 0:
                        d_old = dist[rest[pos-1]][rest[pos]]
                        d_new = dist[rest[pos-1]][node] + dist[node][rest[pos]]
                    else:
                        d_old = 0
                        d_new = dist[0][node] + dist[node][0]
                
                if d_new < d_old - 1e-9:
                    quick_ok, _ = _quick_feasibility(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if quick_ok:
                        feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            route = new_route
                            improved = True
                            break
            if improved:
                break
    
    return route


def _swap_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply swap (exchange two nodes) improvements."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 5
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                # Only swap if both are customers or both are stations
                if (route[i] <= n_customers) != (route[j] <= n_customers):
                    continue
                
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                quick_ok, _ = _quick_feasibility(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                if quick_ok:
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        # Check if it actually improves distance
                        old_dist = 0.0
                        prev = 0
                        for v in route:
                            old_dist += dist[prev][v]
                            prev = v
                        old_dist += dist[prev][0]
                        
                        new_dist = 0.0
                        prev = 0
                        for v in new_route:
                            new_dist += dist[prev][v]
                            prev = v
                        new_dist += dist[prev][0]
                        
                        if new_dist < old_dist - 1e-9:
                            route = new_route
                            improved = True
                            break
            if improved:
                break
    
    return route


def _solve_instance(instance, time_limit=45):
    """Solve a single instance with time limit."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _build_dist_matrix(coords)
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    routes = _construct_initial_routes(instance, dist)
    
    # Local search loop with time management
    while time.time() < deadline - 3:
        improved_any = False
        for i in range(len(routes)):
            if time.time() > deadline - 3:
                break
            old_route = routes[i][:]
            routes[i] = _two_opt_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 3)
            if time.time() > deadline - 3:
                break
            routes[i] = _relocate_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 3)
            if time.time() > deadline - 3:
                break
            routes[i] = _swap_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 3)
            if routes[i] != old_route:
                improved_any = True
        
        if not improved_any:
            break
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    
    # Use most of the available time (150s total, use ~140s with safety margin)
    per_instance_time = 140
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
