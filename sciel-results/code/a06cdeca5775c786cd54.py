"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import permutations


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate_route(route, D, ready, due, service, demands,
                    battery, speed, recharge_rate, horizon, capacity, n_nodes):
    """Simulate a single route (list of node ids, depot implicit at ends).
    Returns (feasible, total_distance, total_time, end_battery)."""
    total_dist = 0.0
    t = 0.0
    bat = battery
    prev = 0  # depot
    for node in route:
        d = D[prev, node]
        total_dist += d
        t += d / speed
        bat -= d
        if bat < 0:
            return False, total_dist, t, bat
        if node == 0:
            prev = node
            continue
        if node in range(1, n_nodes) and demands[node] > 0:
            # customer
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        elif demands[node] == 0 and node > 0:
            # charging station
            bat = battery  # refill
            # recharge time: (battery - energy_before) * recharge_rate
            # energy before refill was bat (after travel), but we already subtracted d
            # Actually the spec says: station adds (battery - energy) * recharge_rate time and refills
            # "energy" here is the battery level upon arrival (after travel cost)
            # We need to compute recharge time based on arrival battery
            # Let me re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # I think "energy" means the battery level when arriving, and "battery" is max capacity
            # So recharge_time = (max_battery - arrival_battery) * recharge_rate
            # But I already set bat = battery above. Let me redo.
            pass
        prev = node
    # return to depot
    d = D[prev, 0]
    total_dist += d
    t += d / speed
    bat -= d
    if bat < 0:
        return False, total_dist, t, bat
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    return True, total_dist, t, bat


def _simulate_route_v2(route, D, ready, due, service, demands,
                       battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set):
    """Proper simulation with charging stations."""
    total_dist = 0.0
    t = 0.0
    bat = float(battery)
    prev = 0
    for node in route:
        d = D[prev, node]
        total_dist += d
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, total_dist, t, bat
        bat = max(bat, 0.0)
        if node in stations_set:
            # Charging station: recharge time = (max_battery - arrival_battery) * recharge_rate
            recharge_time = (float(battery) - bat) * recharge_rate
            t += recharge_time
            bat = float(battery)
        elif node >= 1 and node < n_nodes and demands[node] > 0:
            # Customer
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        prev = node
    # return to depot
    d = D[prev, 0]
    total_dist += d
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return False, total_dist, t, bat
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    return True, total_dist, t, bat


def _route_demand(route, demands):
    return sum(demands[n] for n in route if demands[n] > 0)


def _is_feasible(route, D, ready, due, service, demands,
                 battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set):
    if _route_demand(route, demands) > capacity + 1e-9:
        return False
    ok, _, _, _ = _simulate_route_v2(route, D, ready, due, service, demands,
                                     battery, speed, recharge_rate, horizon, n_nodes, stations_set)
    return ok


def _total_distance(routes, D):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total += D[prev, node]
            prev = node
        total += D[prev, 0]
    return total


def plan(instance):
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit
    
    coords = instance["coords"]
    n_nodes = len(coords)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    stations_set = set(stations)
    D = _dist_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    
    # --- Initial solution: angle-based clustering ---
    # Compute angle from depot for each customer
    depot_x, depot_y = coords[0][0], coords[0][1]
    angles = {}
    for c in customers:
        dx = coords[c][0] - depot_x
        dy = coords[c][1] - depot_y
        angles[c] = math.atan2(dy, dx)
    
    # Sort customers by angle
    sorted_customers = sorted(customers, key=lambda c: angles[c])
    
    # Determine number of routes: try to fit customers into routes respecting capacity
    # Start with a reasonable number of routes
    total_demand = sum(demands[c] for c in customers)
    min_routes = max(1, math.ceil(total_demand / capacity))
    # Also consider distance: each route should be manageable
    # Start with min_routes, maybe add a few more for time windows
    
    n_routes = min_routes + 2  # extra routes for flexibility
    
    # Distribute customers into routes by angle sectors
    routes = [[] for _ in range(n_routes)]
    route_demands = [0] * n_routes
    
    for c in sorted_customers:
        ang = angles[c]
        # Find the route whose sector contains this angle
        # Sector i covers [2*pi*i/n_routes, 2*pi*(i+1)/n_routes)
        # Normalize angle to [0, 2*pi)
        norm_ang = ang % (2 * math.pi)
        sector = int(norm_ang / (2 * math.pi) * n_routes) % n_routes
        # Try to insert into the sector's route if capacity allows
        if route_demands[sector] + demands[c] <= capacity:
            routes[sector].append(c)
            route_demands[sector] += demands[c]
        else:
            # Find the route with the least demand that can fit
            best_r = -1
            best_rem = -1
            for r in range(n_routes):
                rem = capacity - route_demands[r]
                if rem >= demands[c] and rem > best_rem:
                    best_rem = rem
                    best_r = r
            if best_r >= 0:
                routes[best_r].append(c)
                route_demands[best_r] += demands[c]
            else:
                # Add a new route
                routes.append([c])
                route_demands.append(demands[c])
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    # --- Local search: 2-opt and swap within routes ---
    def try_2opt(route, D, ready, due, service, demands,
                 battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set):
        """Try 2-opt moves within a single route. Return improved route or None."""
        if len(route) < 2:
            return None
        best_route = None
        best_dist = None
        current_dist = 0.0
        prev = 0
        for node in route:
            current_dist += D[prev, node]
            prev = node
        current_dist += D[prev, 0]
        
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                # Reverse segment [i..j]
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                # Check demand (same)
                # Check feasibility
                if not _is_feasible(new_route, D, ready, due, service, demands,
                                    battery, speed, recharge_rate, horizon, n_nodes, stations_set):
                    continue
                # Compute distance
                nd = 0.0
                prev = 0
                for node in new_route:
                    nd += D[prev, node]
                    prev = node
                nd += D[prev, 0]
                if best_dist is None or nd < best_dist - 1e-9:
                    best_dist = nd
                    best_route = new_route
        return best_route
    
    def try_swap(r1, r2, D, ready, due, service, demands,
                 battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set):
        """Try swapping a customer between two routes. Return (new_r1, new_r2) or None."""
        if not r1 or not r2:
            return None
        best_pair = None
        best_dist = None
        
        current_d1 = 0.0
        prev = 0
        for node in r1:
            current_d1 += D[prev, node]
            prev = node
        current_d1 += D[prev, 0]
        
        current_d2 = 0.0
        prev = 0
        for node in r2:
            current_d2 += D[prev, node]
            prev = node
        current_d2 += D[prev, 0]
        
        current_total = current_d1 + current_d2
        
        for i, c1 in enumerate(r1):
            for j, c2 in enumerate(r2):
                # Swap c1 and c2
                new_r1 = r1[:i] + [c2] + r1[i+1:]
                new_r2 = r2[:j] + [c1] + r2[j+1:]
                # Check demands
                if _route_demand(new_r1, demands) > capacity + 1e-9:
                    continue
                if _route_demand(new_r2, demands) > capacity + 1e-9:
                    continue
                if not _is_feasible(new_r1, D, ready, due, service, demands,
                                    battery, speed, recharge_rate, horizon, n_nodes, stations_set):
                    continue
                if not _is_feasible(new_r2, D, ready, due, service, demands,
                                    battery, speed, recharge_rate, horizon, n_nodes, stations_set):
                    continue
                nd1 = 0.0
                prev = 0
                for node in new_r1:
                    nd1 += D[prev, node]
                    prev = node
                nd1 += D[prev, 0]
                nd2 = 0.0
                prev = 0
                for node in new_r2:
                    nd2 += D[prev, node]
                    prev = node
                nd2 += D[prev, 0]
                nd_total = nd1 + nd2
                if best_dist is None or nd_total < best_dist - 1e-9:
                    best_dist = nd_total
                    best_pair = (new_r1, new_r2)
        return best_pair
    
    # Apply local search
    improved = True
    max_iters = 20
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline - 5.0:
            break
        
        # 2-opt within each route
        for r in range(len(routes)):
            if time.time() > deadline - 5.0:
                break
            new_route = try_2opt(routes[r], D, ready, due, service, demands,
                                  battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set)
            if new_route is not None:
                routes[r] = new_route
                improved = True
        
        # Swap between pairs of routes
        for i in range(len(routes)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, len(routes)):
                if time.time() > deadline - 5.0:
                    break
                pair = try_swap(routes[i], routes[j], D, ready, due, service, demands,
                                battery, speed, recharge_rate, horizon, n_nodes, stations_set)
                if pair is not None:
                    routes[i], routes[j] = pair
                    improved = True
    
    # Final feasibility check
    all_customers = set()
    for route in routes:
        for c in route:
            all_customers.add(c)
    if all_customers != set(customers):
        # Fallback: one customer per route
        return [[c] for c in customers]
    
    # Verify all routes are feasible
    for route in routes:
        if not _is_feasible(route, D, ready, due, service, demands,
                            battery, speed, recharge_rate, horizon, n_nodes, stations_set):
            # Fallback
            return [[c] for c in customers]
    
    return routes
# EVOLVE-BLOCK-END
