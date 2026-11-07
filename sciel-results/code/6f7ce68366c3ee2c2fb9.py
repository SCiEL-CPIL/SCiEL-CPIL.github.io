"""EV routing with time windows – LNS with DP-based reconstruction."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from collections import defaultdict


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, D, instance):
    """
    Simulate a single route. Returns (total_distance, valid, final_time, final_battery, load).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])

    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev][node]
        dist_total += d
        t += d
        batt -= d
        if batt < -1e-9:
            return dist_total, False, t, batt, load
        if node in stations_set:
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt, load
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt, load
            t += service[node]
        prev = node

    d = D[prev][0]
    dist_total += d
    t += d
    batt -= d
    if batt < -1e-9:
        return dist_total, False, t, batt, load
    if t > horizon + 1e-9:
        return dist_total, False, t, batt, load

    return dist_total, True, t, batt, load


def _total_distance(routes, D, instance):
    total = 0.0
    for route in routes:
        dist, valid, _, _, _ = _simulate(route, D, instance)
        if not valid:
            return float('inf')
        total += dist
    return total


def _is_valid_solution(routes, instance, D):
    served = set()
    for route in routes:
        dist, valid, _, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
        for node in route:
            if node in set(instance["stations"]):
                continue
            if node in served:
                return False
            served.add(node)
    n_customers = instance["n_customers"]
    expected = set(range(1, n_customers + 1))
    return served == expected


def _nearest_neighbor_construct(instance, D):
    """Construct initial routes using nearest neighbor heuristic."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)
    
    routes = []
    remaining = set(customers)
    
    while remaining:
        # Start a new route from depot
        route = []
        current = 0
        t = 0.0
        batt = instance["battery"]
        load = 0
        stations_set = set(instance["stations"])
        
        while remaining:
            # Find nearest feasible customer
            best_customer = None
            best_dist = float('inf')
            
            for c in remaining:
                d = D[current][c]
                new_t = t + d
                new_batt = batt - d
                
                # Check battery
                if new_batt < -1e-9:
                    continue
                
                # Check if we need to charge
                if c in stations_set:
                    pass
                else:
                    # Check time window
                    if new_t < instance["ready"][c]:
                        new_t = instance["ready"][c]
                    if new_t > instance["due"][c] + 1e-9:
                        continue
                    
                    # Check capacity
                    if load + instance["demands"][c] > instance["capacity"] + 1e-9:
                        continue
                    
                    # Check if we can return to depot after serving
                    d_return = D[c][0]
                    if new_t + instance["service"][c] + d_return > instance["horizon"] + 1e-9:
                        continue
                    if new_batt - d_return < -1e-9:
                        continue
                
                if d < best_dist:
                    best_dist = d
                    best_customer = c
            
            if best_customer is None:
                break
            
            # Add the customer
            d = D[current][best_customer]
            t += d
            batt -= d
            route.append(best_customer)
            
            if best_customer not in stations_set:
                if t < instance["ready"][best_customer]:
                    t = instance["ready"][best_customer]
                t += instance["service"][best_customer]
                load += instance["demands"][best_customer]
            else:
                recharge_time = (instance["battery"] - batt) * instance["recharge_rate"]
                t += recharge_time
                batt = instance["battery"]
            
            current = best_customer
            remaining.discard(best_customer)
        
        if route:
            # Check if we can return to depot
            d_return = D[current][0]
            if t + d_return <= instance["horizon"] + 1e-9 and batt - d_return >= -1e-9:
                routes.append(route)
            else:
                # Try to fix by adding stations or dropping the last customer
                # For simplicity, just drop the last customer and retry
                if route:
                    route.pop()
                    remaining.add(route[-1] if route else best_customer)
                    if route:
                        routes.append(route)
    
    return routes


def _savings_construct(instance, D):
    """Clarke-Wright savings heuristic."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations_set = set(instance["stations"])
    
    # Initialize: each customer is its own route
    routes = [[c] for c in customers if c not in stations_set]
    route_of = {c: i for i, route in enumerate(routes) for c in route}
    
    # Compute savings
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1, c2 = customers[i], customers[j]
            if c1 in stations_set or c2 in stations_set:
                continue
            s = D[0][c1] + D[0][c2] - D[c1][c2]
            savings.append((s, c1, c2))
    
    savings.sort(reverse=True)
    
    for s, c1, c2 in savings:
        if s <= 0:
            break
        
        r1 = route_of.get(c1)
        r2 = route_of.get(c2)
        
        if r1 is None or r2 is None or r1 == r2:
            continue
        
        route1 = routes[r1]
        route2 = routes[r2]
        
        # Check if c1 and c2 are at the ends of their routes
        can_merge = False
        new_route = None
        
        if route1[0] == c1 and route2[0] == c2:
            new_route = route2 + route1
            can_merge = True
        elif route1[0] == c1 and route2[-1] == c2:
            new_route = list(reversed(route2)) + route1
            can_merge = True
        elif route1[-1] == c1 and route2[0] == c2:
            new_route = route1 + route2
            can_merge = True
        elif route1[-1] == c1 and route2[-1] == c2:
            new_route = route1 + list(reversed(route2))
            can_merge = True
        
        if not can_merge:
            continue
        
        # Check feasibility of merged route
        load = sum(instance["demands"][c] for c in new_route if c not in stations_set)
        if load > instance["capacity"] + 1e-9:
            continue
        
        # Quick feasibility check
        t = 0.0
        batt = instance["battery"]
        prev = 0
        feasible = True
        for node in new_route:
            d = D[prev][node]
            t += d
            batt -= d
            if batt < -1e-9:
                feasible = False
                break
            if node not in stations_set:
                if t < instance["ready"][node]:
                    t = instance["ready"][node]
                if t > instance["due"][node] + 1e-9:
                    feasible = False
                    break
                t += instance["service"][node]
            prev = node
        
        if feasible:
            d_return = D[prev][0]
            t += d_return
            batt -= d_return
            if t > instance["horizon"] + 1e-9 or batt < -1e-9:
                feasible = False
        
        if feasible:
            # Merge routes
            routes[r1] = new_route
            routes[r2] = []
            for c in new_route:
                route_of[c] = r1
            # Clean up empty routes
            routes = [r for r in routes if r]
            # Update route_of
            for i, route in enumerate(routes):
                for c in route:
                    route_of[c] = i
    
    return routes


def _insert_station_if_needed(route, D, instance):
    """Insert stations into route if needed for battery constraints."""
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    
    new_route = []
    t = 0.0
    batt = battery_max
    prev = 0
    
    for node in route:
        d = D[prev][node]
        
        if batt - d < -1e-9:
            # Need to charge before going to node
            # Find the best station to visit
            best_station = None
            best_total_d = float('inf')
            
            for st in instance["stations"]:
                d1 = D[prev][st]
                d2 = D[st][node]
                if d1 <= batt + 1e-9 and d1 + d2 <= battery_max + 1e-9:
                    total_d = d1 + d2
                    if total_d < best_total_d:
                        best_total_d = total_d
                        best_station = st
            
            if best_station is None:
                return None
            
            # Visit the station
            d1 = D[prev][best_station]
            t += d1
            batt -= d1
            new_route.append(best_station)
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
            prev = best_station
        
        # Now go to node
        d = D[prev][node]
        t += d
        batt -= d
        new_route.append(node)
        
        if node not in stations_set:
            if t < instance["ready"][node]:
                t = instance["ready"][node]
            if t > instance["due"][node] + 1e-9:
                return None
            t += instance["service"][node]
            load = sum(instance["demands"][c] for c in new_route if c not in stations_set)
            if load > instance["capacity"] + 1e-9:
                return None
        else:
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
        
        prev = node
    
    # Return to depot
    d = D[prev][0]
    if batt - d < -1e-9:
        # Need to charge before returning
        best_station = None
        best_total_d = float('inf')
        
        for st in instance["stations"]:
            d1 = D[prev][st]
            d2 = D[st][0]
            if d1 <= batt + 1e-9 and d1 + d2 <= battery_max + 1e-9:
                total_d = d1 + d2
                if total_d < best_total_d:
                    best_total_d = total_d
                    best_station = st
        
        if best_station is None:
            return None
        
        d1 = D[prev][best_station]
        t += d1
        batt -= d1
        new_route.append(best_station)
        recharge_time = (battery_max - batt) * recharge_rate
        t += recharge_time
        batt = battery_max
        prev = best_station
        d = D[prev][0]
        t += d
        batt -= d
    else:
        t += d
        batt -= d
    
    if t > instance["horizon"] + 1e-9 or batt < -1e-9:
        return None
    
    return new_route


def _two_opt(route, D, instance, max_iters=20, time_limit=None):
    """2-opt improvement for a single route."""
    if time_limit is None:
        time_limit = time.time() + 3.0
    
    improved = True
    iters = 0
    current_dist, valid, _, _, _ = _simulate(route, D, instance)
    if not valid:
        return route
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > time_limit:
            break
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                dist, valid, _, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - 1e-9:
                    route = new_route
                    current_dist = dist
                    improved = True
    return route


def _or_opt(route, D, instance, max_iters=15, time_limit=None):
    """Or-opt: relocate 1 or 2 consecutive nodes."""
    if time_limit is None:
        time_limit = time.time() + 3.0
    
    improved = True
    iters = 0
    current_dist, valid, _, _, _ = _simulate(route, D, instance)
    if not valid:
        return route
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > time_limit:
            break
        n = len(route)
        for i in range(n):
            for length in [1, 2]:
                if i + length > n:
                    continue
                segment = route[i:i+length]
                rest = route[:i] + route[i+length:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + segment + rest[j:]
                    dist, valid, _, _, _ = _simulate(new_route, D, instance)
                    if valid and dist < current_dist - 1e-9:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route


def _inter_route_swap(routes, D, instance, max_iters=10, time_limit=None):
    """Inter-route improvement: swap customers between routes."""
    if time_limit is None:
        time_limit = time.time() + 3.0
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > time_limit:
            break
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        da, va, _, _, _ = _simulate(ra, D, instance)
                        db, vb, _, _, _ = _simulate(rb, D, instance)
                        if va and vb:
                            old_dist = _simulate(routes[a], D, instance)[0] + _simulate(routes[b], D, instance)[0]
                            new_dist = da + db
                            if new_dist < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True
    return routes


def _lns_reconstruct(routes, D, instance, num_customers_to_remove, time_limit=None):
    """LNS: remove a random subset of customers and reinsert them optimally."""
    if time_limit is None:
        time_limit = time.time() + 2.0
    
    # Collect all customers from routes
    all_customers = []
    for route in routes:
        for node in route:
            if node not in set(instance["stations"]):
                all_customers.append(node)
    
    if len(all_customers) <= num_customers_to_remove:
        return routes
    
    # Remove a random subset of customers
    to_remove = set(random.sample(all_customers, num_customers_to_remove))
    
    # Create new routes without these customers
    new_routes = []
    for route in routes:
        new_route = [node for node in route if node not in to_remove]
        if new_route:
            new_routes.append(new_route)
    
    # Try to reinsert each removed customer
    for c in to_remove:
        best_route_idx = -1
        best_pos = -1
        best_dist = float('inf')
        
        for r_idx, route in enumerate(new_routes):
            # Try inserting at each position
            for pos in range(len(route) + 1):
                test_route = route[:pos] + [c] + route[pos:]
                dist, valid, _, _, _ = _simulate(test_route, D, instance)
                if valid and dist < best_dist:
                    best_dist = dist
                    best_route_idx = r_idx
                    best_pos = pos
        
        # Also try creating a new route with just this customer
        test_route = [c]
        dist, valid, _, _, _ = _simulate(test_route, D, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_route_idx = -2  # Special marker for new route
            best_pos = 0
        
        if best_route_idx == -2:
            new_routes.append([c])
        elif best_route_idx >= 0:
            new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [c] + new_routes[best_route_idx][best_pos:]
        else:
            # Can't insert, add as a new route (may be infeasible, but we'll check later)
            new_routes.append([c])
    
    return new_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    
    # Try multiple construction heuristics and pick the best
    best_routes = None
    best_dist = float('inf')
    
    # Nearest neighbor with different random seeds
    for seed in [0, 1, 2]:
        random.seed(seed)
        routes = _nearest_neighbor_construct(instance, D)
        # Insert stations if needed
        full_routes = []
        for route in routes:
            full_route = _insert_station_if_needed(route, D, instance)
            if full_route is not None:
                full_routes.append(full_route)
        if full_routes and _is_valid_solution(full_routes, instance, D):
            dist = _total_distance(full_routes, D, instance)
            if dist < best_dist:
                best_dist = dist
                best_routes = full_routes
    
    # Savings heuristic
    routes = _savings_construct(instance, D)
    full_routes = []
    for route in routes:
        full_route = _insert_station_if_needed(route, D, instance)
        if full_route is not None:
            full_routes.append(full_route)
    if full_routes and _is_valid_solution(full_routes, instance, D):
        dist = _total_distance(full_routes, D, instance)
        if dist < best_dist:
            best_dist = dist
            best_routes = full_routes
    
    # Fallback: one customer per route
    if best_routes is None or not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = []
        for c in customers:
            route = _insert_station_if_needed([c], D, instance)
            if route is not None:
                best_routes.append(route)
        best_dist = _total_distance(best_routes, D, instance)
    
    # Local search improvements
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        time_limit = time.time() + (deadline - time.time()) / (max_passes + 2)
        
        # 2-opt on each route
        for i in range(len(best_routes)):
            new_route = _two_opt(best_routes[i], D, instance, max_iters=10, time_limit=time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
        
        # Or-opt
        for i in range(len(best_routes)):
            new_route = _or_opt(best_routes[i], D, instance, max_iters=10, time_limit=time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
        
        # Inter-route swap
        new_routes = _inter_route_swap(best_routes, D, instance, max_iters=5, time_limit=time_limit)
        if new_routes != best_routes:
            best_routes = new_routes
            improved = True
        
        # LNS reconstruction
        if len(best_routes) > 1:
            num_remove = max(2, len([c for r in best_routes for c in r if c not in set(instance["stations"])]) // 4)
            new_routes = _lns_reconstruct(best_routes, D, instance, num_remove, time_limit=time_limit)
            if new_routes and _is_valid_solution(new_routes, instance, D):
                new_dist = _total_distance(new_routes, D, instance)
                if new_dist < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = new_dist
                    improved = True
        
        if not improved:
            break
    
    # Final validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]
    
    return best_routes
# EVOLVE-BLOCK-END
