import numpy as np
import math
import time

def plan(instance):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for safety

    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Node 0 is depot, 1..n_customers are customers, rest are stations
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = np.linalg.norm(coords[i] - coords[j])
            dist[i, j] = d
            dist[j, i] = d

    def simulate_route(route):
        """
        Returns (is_feasible, total_distance)
        """
        if not route:
            return True, 0.0
            
        t = 0.0
        b = float(battery_cap)
        load = 0.0
        curr_pos = 0
        total_dist = 0.0
        
        for node in route:
            d = dist[curr_pos, node]
            b -= d
            if b < -1e-9:
                return False, 0.0
            t += d / speed
            total_dist += d
            curr_pos = node
            
            if node in stations:
                # Charge time: (battery_cap - b) * recharge_rate
                charge_time = (battery_cap - b) * recharge_rate
                t += charge_time
                b = battery_cap
            elif 1 <= node <= n_customers:
                # Customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False, 0.0
                t += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0
            else:
                # Depot in middle? Not allowed in route list.
                pass
        
        # Return to depot
        d = dist[curr_pos, 0]
        b -= d
        if b < -1e-9:
            return False, 0.0
        t += d / speed
        total_dist += d
        
        if t > horizon + 1e-9:
            return False, 0.0
            
        return True, total_dist

    def total_distance(routes):
        total = 0.0
        for r in routes:
            feasible, d = simulate_route(r)
            if not feasible:
                return float('inf')
            total += d
        return total

    # 1. Constructive Heuristic: Nearest Neighbor with station support
    def construct_initial():
        unvisited = set(range(1, n_customers + 1))
        routes = []
        
        while unvisited:
            route = []
            curr_pos = 0
            t = 0.0
            b = float(battery_cap)
            load = 0.0
            
            while unvisited:
                best_node = None
                best_d = float('inf')
                best_new_t = 0.0
                best_new_b = 0.0
                best_load = 0.0
                best_type = 'customer' # 'customer' or 'station'
                
                # Candidates: Unvisited customers
                for c in unvisited:
                    if load + demands[c] > capacity + 1e-9:
                        continue
                    d = dist[curr_pos, c]
                    new_b = b - d
                    if new_b < -1e-9:
                        continue
                    new_t = t + d / speed
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c] + 1e-9:
                        continue
                    new_t += service[c]
                    new_load = load + demands[c]
                    
                    # Prefer closer customers
                    if d < best_d:
                        best_d = d
                        best_node = c
                        best_new_t = new_t
                        best_new_b = new_b
                        best_load = new_load
                        best_type = 'customer'
                
                # Candidates: Stations
                # We consider stations if no customer is found, or if going to a station 
                # allows reaching a customer that is currently unreachable due to battery.
                # For simplicity, if no customer is directly reachable, we look for a station.
                if best_node is None:
                    for s in stations:
                        d = dist[curr_pos, s]
                        new_b = b - d
                        if new_b < -1e-9:
                            continue
                        new_t = t + d / speed
                        charge_time = (battery_cap - new_b) * recharge_rate
                        new_t += charge_time
                        new_b = battery_cap
                        
                        # Check if from this station we can reach at least one customer or depot
                        # To be safe, let's just pick the closest station if we are stuck.
                        # This might lead to unnecessary station visits if a customer was reachable 
                        # but we missed it due to the 'best_node is None' check logic.
                        # However, if best_node is None, it means NO customer was reachable.
                        # So a station is the only option to proceed (or return to depot).
                        
                        # Check if returning to depot is better (shorter distance)
                        d_depot = dist[curr_pos, 0]
                        if d < best_d:
                            best_d = d
                            best_node = s
                            best_new_t = new_t
                            best_new_b = new_b
                            best_load = load
                            best_type = 'station'
                
                if best_node is None:
                    # Stuck, go to depot
                    d = dist[curr_pos, 0]
                    new_b = b - d
                    if new_b < -1e-9:
                        # Infeasible
                        break
                    new_t = t + d / speed
                    if new_t > horizon + 1e-9:
                        # Infeasible
                        break
                    # End route
                    break
                else:
                    route.append(best_node)
                    curr_pos = best_node
                    t = best_new_t
                    b = best_new_b
                    load = best_load
                    if best_type == 'customer':
                        unvisited.remove(best_node)
            
            if route:
                # Check feasibility of the constructed route
                feasible, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Fallback: assign remaining unvisited to single routes
                    for c in list(unvisited):
                        routes.append([c])
                    unvisited.clear()
                    break
            
        return routes

    routes = construct_initial()
    
    # Verify initial solution
    if not routes or not all(simulate_route(r)[0] for r in routes):
        # Fallback to trivial solution
        return [[i] for i in range(1, n_customers + 1)]

    current_dist = total_distance(routes)
    best_routes = [r[:] for r in routes]
    best_dist = current_dist

    # 2. Local Search
    def try_2opt(routes, route_idx):
        route = routes[route_idx]
        if len(route) < 2:
            return False, 0.0
        
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                # Reverse segment i..j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                feasible, new_d = simulate_route(new_route)
                if feasible:
                    old_d = simulate_route(route)[1]
                    if new_d < old_d - 1e-9:
                        routes[route_idx] = new_route
                        return True, new_d - old_d
        return False, 0.0

    def try_or_opt(routes):
        improved = False
        best_improvement = 0.0
        best_move = None
        
        n_routes = len(routes)
        
        for i in range(n_routes):
            route_i = routes[i]
            for j in range(len(route_i)):
                for k in range(j, min(j+3, len(route_i))):
                    segment = route_i[j:k+1]
                    new_route_i = route_i[:j] + route_i[k+1:]
                    
                    for m in range(n_routes):
                        if m == i:
                            continue
                        route_m = routes[m]
                        for pos in range(len(route_m) + 1):
                            new_route_m = route_m[:pos] + segment + route_m[pos:]
                            
                            feas_i, d_i = simulate_route(new_route_i)
                            if not feas_i:
                                continue
                            feas_m, d_m = simulate_route(new_route_m)
                            if not feas_m:
                                continue
                            
                            old_d_i = simulate_route(route_i)[1]
                            old_d_m = simulate_route(route_m)[1]
                            new_total = d_i + d_m
                            old_total = old_d_i + old_d_m
                            
                            if new_total < old_total - 1e-9:
                                best_improvement = old_total - new_total
                                best_move = (i, j, k, m, pos, new_route_i, new_route_m)
                                improved = True
        
        if best_move:
            i, j, k, m, pos, new_route_i, new_route_m = best_move
            routes[i] = new_route_i
            routes[m] = new_route_m
            return True, best_improvement
            
        return False, 0.0

    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try 2-Opt on each route
        for i in range(len(routes)):
            imp, delta = try_2opt(routes, i)
            if imp:
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
            
        # Try Or-Opt
        imp, delta = try_or_opt(routes)
        if imp:
            improved = True
            no_improve_count = 0
            continue
            
        no_improve_count += 1
        
        current_dist = total_distance(routes)
        if current_dist < best_dist:
            best_dist = current_dist
            best_routes = [r[:] for r in routes]

    if total_distance(best_routes) < total_distance(routes):
        return best_routes
    return routes
