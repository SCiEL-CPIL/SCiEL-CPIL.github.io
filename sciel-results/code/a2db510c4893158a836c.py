"""EV routing with time windows, charging stations, and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def plan(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    depot = 0
    
    # Precompute distance matrix
    dist = cdist(coords, coords)
    
    # Node types: 0 = depot, 1 = customer, 2 = station
    node_type = np.zeros(n_nodes, dtype=np.int8)
    for i in range(1, n_nodes):
        if i <= n_customers:
            node_type[i] = 1  # customer
        elif i in stations:
            node_type[i] = 2
    
    customer_ids = list(range(1, n_customers + 1))
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_customer[1:n_customers+1] = True
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Initial solution: sort by due time, one customer per route
    sorted_customers = sorted(customer_ids, key=lambda v: due[v])
    routes = [[v] for v in sorted_customers]
    
    def simulate_route(route):
        """Simulate a route, return (total_distance, feasible, end_time, end_battery)"""
        if not route:
            return 0.0, True, 0.0, battery_max
        total_dist = 0.0
        t = 0.0
        batt = battery_max
        prev = depot
        
        for node in route:
            d = dist[prev, node]
            if d > batt:
                return total_dist + d, False, 0, 0
            batt -= d
            t += d / speed
            total_dist += d
            
            if node_type[node] == 1:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return total_dist, False, 0, 0
                t += service[node]
            elif node_type[node] == 2:
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            
            prev = node
        
        d = dist[prev, depot]
        if d > batt:
            return total_dist + d, False, 0, 0
        batt -= d
        t += d / speed
        total_dist += d
        
        if t > horizon:
            return total_dist, False, 0, 0
        
        return total_dist, True, t, batt
    
    def total_distance(routes_list):
        total = 0.0
        for route in routes_list:
            d, _, _, _ = simulate_route(route)
            total += d
        return total
    
    def is_valid_solution(routes_list):
        served = set()
        for route in routes_list:
            for node in route:
                if node_type[node] != 1:
                    return False
                if node in served:
                    return False
                served.add(node)
        if served != set(customer_ids):
            return False
        for route in routes_list:
            _, feasible, _, _ = simulate_route(route)
            if not feasible:
                return False
        return True
    
    if not is_valid_solution(routes):
        return [[v] for v in customer_ids]
    
    best_routes = [list(r) for r in routes]
    best_dist = total_distance(best_routes)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute route-level data for fast evaluation
    # For each route, store: list of nodes, cumulative distances, cumulative times, cumulative battery
    def precompute_route_data(route):
        """Precompute arrays for a route: dists, times, batteries, total_dist"""
        if not route:
            return None
        n = len(route)
        dists = np.zeros(n)
        times = np.zeros(n)
        batteries = np.zeros(n)
        t = 0.0
        batt = battery_max
        prev = depot
        total_d = 0.0
        for i, node in enumerate(route):
            d = dist[prev, node]
            dists[i] = d
            total_d += d
            batt -= d
            t += d / speed
            if node_type[node] == 1:
                t = max(t, ready[node])
                t += service[node]
            elif node_type[node] == 2:
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            times[i] = t
            batteries[i] = batt
            prev = node
        # Add return to depot
        d_ret = dist[prev, depot]
        total_d += d_ret
        t += d_ret / speed
        batt -= d_ret
        return {
            'dists': dists,
            'times': times,
            'batteries': batteries,
            'total_dist': total_d,
            'end_time': t,
            'end_battery': batt,
            'ret_dist': d_ret
        }
    
    # Fast feasibility check for a route
    def fast_simulate(route):
        """Fast simulation using precomputed logic. Returns (dist, feasible)"""
        if not route:
            return 0.0, True
        t = 0.0
        batt = battery_max
        prev = depot
        total_d = 0.0
        for node in route:
            d = dist[prev, node]
            if d > batt:
                return total_d + d, False
            batt -= d
            t += d / speed
            total_d += d
            if node_type[node] == 1:
                t = max(t, ready[node])
                if t > due[node]:
                    return total_d, False
                t += service[node]
            elif node_type[node] == 2:
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            prev = node
        d = dist[prev, depot]
        if d > batt:
            return total_d + d, False
        total_d += d
        t += d / speed
        if t > horizon:
            return total_d, False
        return total_d, True
    
    # Local search with 2-opt and inter-route moves
    def try_2opt_improve(route, i, j, current_dist):
        """Try reversing segment i..j. Returns (new_route, new_dist) if improved, else None."""
        if i >= j:
            return None
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        cap = sum(demands[node] for node in new_route)
        if cap > capacity:
            return None
        d, feasible = fast_simulate(new_route)
        if feasible and d < current_dist:
            return (new_route, d)
        return None
    
    # Run local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n_routes = len(routes)
        
        # Compute current distances
        route_dists = []
        total_d = 0.0
        for r in routes:
            d, f = fast_simulate(r)
            if not f:
                break
            route_dists.append(d)
            total_d += d
        else:
            # 2-opt within each route
            for r in range(n_routes):
                route = routes[r]
                L = len(route)
                if L < 2:
                    continue
                for i in range(L - 1):
                    if (time.time() - start_time) > time_limit:
                        break
                    for j in range(i + 1, L):
                        result = try_2opt_improve(route, i, j, route_dists[r])
                        if result is not None:
                            new_route, new_d = result
                            routes[r] = new_route
                            route_dists[r] = new_d
                            total_d += new_d - route_dists[r]
                            route_dists[r] = new_d
                            improved = True
        
        # Inter-route: try moving a customer from one route to another
        if (time.time() - start_time) < time_limit:
            for r1 in range(n_routes):
                if (time.time() - start_time) > time_limit:
                    break
                route1 = routes[r1]
                if len(route1) < 2:
                    continue
                for i1 in range(len(route1)):
                    node = route1[i1]
                    # Try removing from route1
                    new_r1 = route1[:i1] + route1[i1+1:]
                    d1, f1 = fast_simulate(new_r1)
                    if not f1:
                        continue
                    # Try inserting into each other route
                    for r2 in range(n_routes):
                        if r2 == r1:
                            continue
                        if (time.time() - start_time) > time_limit:
                            break
                        route2 = routes[r2]
                        # Try all insertion positions
                        for pos in range(len(route2) + 1):
                            new_r2 = route2[:pos] + [node] + route2[pos:]
                            # Check capacity
                            cap2 = sum(demands[n] for n in new_r2)
                            if cap2 > capacity:
                                continue
                            d2, f2 = fast_simulate(new_r2)
                            if not f2:
                                continue
                            # Check total improvement
                            new_total = d1 + d2
                            old_total = route_dists[r1] + route_dists[r2]
                            if new_total < old_total - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                route_dists[r1] = d1
                                route_dists[r2] = d2
                                total_d += new_total - old_total
                                improved = True
                                break  # Restart r2 loop with new route2
                        if improved:
                            break
    
    # Final validation
    if is_valid_solution(routes):
        return routes
    else:
        return [[v] for v in customer_ids]
# EVOLVE-BLOCK-END
