# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from math import sqrt

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0

    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)

    # Pre-compute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d

    def simulate_route(route, full_check=True):
        """
        Simulate a route. Returns (is_valid, total_dist).
        If full_check is False, only checks battery and capacity, not time windows.
        """
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        for node in route:
            d = dist[cur_node, node]
            cur_time += d / speed
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in stations:
                energy = battery - cur_bat
                cur_time += energy * recharge_rate
                cur_bat = battery
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist
                if full_check:
                    cur_time = max(cur_time, ready[node])
                    if cur_time > due[node] + 1e-9:
                        return False, total_dist
                cur_time += service[node]

            cur_node = node

        d = dist[cur_node, 0]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if full_check and cur_time > horizon + 1e-9:
            return False, total_dist

        return True, total_dist

    def compute_total_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            cur_node = 0
            for node in route:
                total += dist[cur_node, node]
                cur_node = node
            total += dist[cur_node, 0]
        return total

    def insert_best_position(route, node, full_check=True):
        """
        Find the best position to insert 'node' into 'route' maintaining validity.
        Returns (new_route, is_valid, total_dist)
        """
        best_cost = float('inf')
        best_new_route = None
        
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [node] + route[pos:]
            is_valid, total_d = simulate_route(new_route, full_check)
            if is_valid:
                if total_d < best_cost:
                    best_cost = total_d
                    best_new_route = new_route
        
        if best_new_route is not None:
            return best_new_route, True, best_cost
        return route, False, float('inf')

    def construct_swept_cluster_solution():
        """
        1. Sort customers by angle from depot.
        2. Sweep to create clusters based on capacity and battery feasibility.
        3. Insert remaining customers into existing clusters if possible.
        """
        angles = [math.atan2(coords[c, 1] - coords[0, 1], coords[c, 0] - coords[0, 0]) for c in customers]
        sorted_customers = [c for c, a in sorted(zip(customers, angles), key=lambda x: x[1])]
        
        routes = []
        route_loads = []
        unvisited = set(sorted_customers)
        
        current_route = []
        current_load = 0
        
        for c in sorted_customers:
            if c not in unvisited:
                continue
                
            if current_load + demands[c] <= capacity:
                new_route, is_valid, _ = insert_best_position(current_route, c)
                if is_valid:
                    current_route = new_route
                    current_load += demands[c]
                    unvisited.remove(c)
                    continue
            
            if current_route:
                routes.append(current_route)
                route_loads.append(current_load)
                current_route = []
                current_load = 0
            
            new_route, is_valid, _ = insert_best_position([], c)
            if is_valid:
                current_route = new_route
                current_load = demands[c]
                unvisited.remove(c)
        
        if current_route:
            routes.append(current_route)
            route_loads.append(current_load)
            
        for c in list(unvisited):
            best_r_idx = -1
            best_cost = float('inf')
            
            for r_idx, route in enumerate(routes):
                if route_loads[r_idx] + demands[c] > capacity:
                    continue
                new_route, is_valid, total_d = insert_best_position(route, c)
                if is_valid:
                    min_d = min(dist[n, c] for n in route) if route else dist[0, c]
                    if min_d < best_cost:
                        best_cost = min_d
                        best_r_idx = r_idx
            
            if best_r_idx != -1:
                routes[best_r_idx], _, _ = insert_best_position(routes[best_r_idx], c)
                route_loads[best_r_idx] += demands[c]
            else:
                routes.append([c])
                route_loads.append(demands[c])
                
            unvisited.remove(c)
            
        return routes

    def local_search(routes, time_limit):
        """
        Local search with 2-opt, or-opt, and inter-route moves.
        """
        current_total_dist = compute_total_distance(routes)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            
            # 1. Intra-route 2-opt
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid:
                            old_route_dist = 0.0
                            cn = 0
                            for node in route:
                                old_route_dist += dist[cn, node]
                                cn = node
                            old_route_dist += dist[cn, 0]
                            
                            if new_dist < old_route_dist - 1e-9:
                                routes[r_idx] = new_route
                                improved = True
                                current_total_dist += new_dist - old_route_dist
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # 2. Or-opt
            if not improved:
                for r_idx in range(len(routes)):
                    route = routes[r_idx]
                    n = len(route)
                    for length in [1, 2, 3]:
                        for i in range(n - length + 1):
                            segment = route[i:i+length]
                            new_route = route[:i] + route[i+length:]
                            for j in range(len(new_route) + 1):
                                if j == i:
                                    continue
                                candidate = new_route[:j] + segment + new_route[j:]
                                if candidate == route:
                                    continue
                                is_valid, new_dist = simulate_route(candidate)
                                if is_valid:
                                    old_route_dist = 0.0
                                    cn = 0
                                    for node in route:
                                        old_route_dist += dist[cn, node]
                                        cn = node
                                    old_route_dist += dist[cn, 0]
                                    
                                    if new_dist < old_route_dist - 1e-9:
                                        routes[r_idx] = candidate
                                        improved = True
                                        current_total_dist += new_dist - old_route_dist
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

            # 3. Inter-route swap
            if not improved and len(routes) > 1:
                for r1 in range(len(routes)):
                    for r2 in range(r1 + 1, len(routes)):
                        route1 = routes[r1]
                        route2 = routes[r2]
                        for i in range(len(route1)):
                            for j in range(len(route2)):
                                c1 = route1[i]
                                c2 = route2[j]
                                new_route1 = route1[:i] + [c2] + route1[i+1:]
                                new_route2 = route2[:j] + [c1] + route2[j+1:]
                                
                                v1, d1 = simulate_route(new_route1)
                                v2, d2 = simulate_route(new_route2)
                                
                                if v1 and v2:
                                    old_d1 = 0.0
                                    cn = 0
                                    for node in route1:
                                        old_d1 += dist[cn, node]
                                        cn = node
                                    old_d1 += dist[cn, 0]
                                    
                                    old_d2 = 0.0
                                    cn = 0
                                    for node in route2:
                                        old_d2 += dist[cn, node]
                                        cn = node
                                    old_d2 += dist[cn, 0]
                                    
                                    if d1 + d2 < old_d1 + old_d2 - 1e-9:
                                        routes[r1] = new_route1
                                        routes[r2] = new_route2
                                        improved = True
                                        current_total_dist += (d1 + d2) - (old_d1 + old_d2)
                                        break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            
            # 4. Relocate
            if not improved and len(routes) > 1:
                for r1 in range(len(routes)):
                    route1 = routes[r1]
                    for i in range(len(route1)):
                        c = route1[i]
                        new_route1 = route1[:i] + route1[i+1:]
                        for r2 in range(len(routes)):
                            if r2 == r1:
                                continue
                            route2 = routes[r2]
                            for j in range(len(route2) + 1):
                                new_route2 = route2[:j] + [c] + route2[j:]
                                v1, d1 = simulate_route(new_route1)
                                v2, d2 = simulate_route(new_route2)
                                if v1 and v2:
                                    old_d1 = 0.0
                                    cn = 0
                                    for node in route1:
                                        old_d1 += dist[cn, node]
                                        cn = node
                                    old_d1 += dist[cn, 0]
                                    
                                    old_d2 = 0.0
                                    cn = 0
                                    for node in route2:
                                        old_d2 += dist[cn, node]
                                        cn = node
                                    old_d2 += dist[cn, 0]
                                    
                                    if d1 + d2 < old_d1 + old_d2 - 1e-9:
                                        routes[r1] = new_route1
                                        routes[r2] = new_route2
                                        improved = True
                                        current_total_dist += (d1 + d2) - (old_d1 + old_d2)
                                        break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    continue

    # Main logic
    routes = construct_swept_cluster_solution()
    
    if time.time() < deadline - 2.0:
        local_search(routes, deadline - 2.0 - time.time())
    
    return routes
# EVOLVE-BLOCK-END
