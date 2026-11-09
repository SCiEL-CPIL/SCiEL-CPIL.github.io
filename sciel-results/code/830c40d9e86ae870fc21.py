"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations


def _euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _route_distance(coords, route):
    """Total Euclidean distance including depot legs."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0  # depot
    for node in route:
        total += _euclidean(coords[prev], coords[node])
        prev = node
    total += _euclidean(coords[prev], coords[0])  # back to depot
    return total


def _check_route(coords, demands, ready, due, service, route, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Check if a route is feasible. Returns (feasible, total_dist)."""
    if not route:
        return True, 0.0
    
    total_demand = sum(demands[n] for n in route)
    if total_demand > capacity:
        return False, 0.0
    
    time = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0
    
    for node in route:
        d = _euclidean(coords[prev], coords[node])
        total_dist += d
        time += d / speed
        battery -= d
        
        if battery < -1e-9:
            return False, 0.0
        
        # At node
        if node in stations:
            # Charging station: add charging time, refill battery
            energy_needed = battery_max - battery
            if energy_needed > 1e-9:
                time += energy_needed * recharge_rate
            battery = battery_max
        else:
            # Customer: check time window
            if time < ready[node]:
                time = ready[node]
            if time > due[node] + 1e-9:
                return False, 0.0
            time += service[node]
        
        prev = node
    
    # Return to depot
    d = _euclidean(coords[prev], coords[0])
    total_dist += d
    time += d / speed
    battery -= d
    
    if battery < -1e-9:
        return False, 0.0
    if time > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist


def _greedy_insertion(coords, demands, ready, due, service, capacity, battery_max, speed, recharge_rate, horizon, stations, customers, max_routes=10):
    """Greedy insertion heuristic to build routes."""
    n = len(customers)
    if n == 0:
        return []
    
    # Sort customers by due time (earliest first) for initial ordering
    sorted_customers = sorted(customers, key=lambda c: due[c])
    
    routes = []
    route_demands = []
    route_batteries = []  # not needed, we check full route each time
    
    for cust in sorted_customers:
        best_route_idx = -1
        best_pos = -1
        best_dist = float('inf')
        best_feasible = False
        
        for r_idx, route in enumerate(routes):
            # Try inserting cust at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, dist = _check_route(
                    coords, demands, ready, due, service, new_route,
                    capacity, battery_max, speed, recharge_rate, horizon, stations
                )
                if feasible and dist < best_dist:
                    best_dist = dist
                    best_route_idx = r_idx
                    best_pos = pos
                    best_feasible = True
        
        if best_feasible:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
        else:
            # Start a new route
            new_route = [cust]
            feasible, _ = _check_route(
                coords, demands, ready, due, service, new_route,
                capacity, battery_max, speed, recharge_rate, horizon, stations
            )
            if feasible:
                routes.append(new_route)
            else:
                # Should not happen with single customer, but handle gracefully
                routes.append(new_route)
    
    return routes


def _local_search_2opt(coords, demands, ready, due, service, capacity, battery_max, speed, recharge_rate, horizon, stations, routes, max_iter=50):
    """2-opt local search within each route."""
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, new_dist = _check_route(
                        coords, demands, ready, due, service, new_route,
                        capacity, battery_max, speed, recharge_rate, horizon, stations
                    )
                    if feasible:
                        old_dist = _route_distance(coords, route)
                        if new_dist < old_dist - 1e-9:
                            routes[r_idx] = new_route
                            improved = True
    return routes


def _cross_route_2opt(coords, demands, ready, due, service, capacity, battery_max, speed, recharge_rate, horizon, stations, routes, max_iter=20):
    """Or-opt: move a segment from one route to another or within route."""
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            for seg_len in [1, 2]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    # Remove segment from this route
                    remaining = route[:i] + route[i+seg_len:]
                    if not remaining:
                        continue
                    # Try inserting into each other route
                    for r2_idx in range(len(routes)):
                        if r2_idx == r_idx:
                            continue
                        target = routes[r2_idx]
                        for pos in range(len(target) + 1):
                            new_target = target[:pos] + segment + target[pos:]
                            feasible1, _ = _check_route(
                                coords, demands, ready, due, service, remaining,
                                capacity, battery_max, speed, recharge_rate, horizon, stations
                            )
                            if not feasible1:
                                continue
                            feasible2, dist2 = _check_route(
                                coords, demands, ready, due, service, new_target,
                                capacity, battery_max, speed, recharge_rate, horizon, stations
                            )
                            if not feasible2:
                                continue
                            old_dist1 = _route_distance(coords, route)
                            old_dist2 = _route_distance(coords, target)
                            new_dist1 = _route_distance(coords, remaining)
                            if new_dist1 + dist2 < old_dist1 + old_dist2 - 1e-9:
                                routes[r_idx] = remaining
                                routes[r2_idx] = new_target
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    # Phase 1: Greedy insertion
    routes = _greedy_insertion(
        coords, demands, ready, due, service,
        capacity, battery_max, speed, recharge_rate, horizon, stations, customers
    )
    
    # Phase 2: Local search - 2-opt within routes
    routes = _local_search_2opt(
        coords, demands, ready, due, service,
        capacity, battery_max, speed, recharge_rate, horizon, stations, routes
    )
    
    # Phase 3: Cross-route moves
    routes = _cross_route_2opt(
        coords, demands, ready, due, service,
        capacity, battery_max, speed, recharge_rate, horizon, stations, routes
    )
    
    # Phase 4: Another round of 2-opt
    routes = _local_search_2opt(
        coords, demands, ready, due, service,
        capacity, battery_max, speed, recharge_rate, horizon, stations, routes
    )
    
    return routes
# EVOLVE-BLOCK-END
