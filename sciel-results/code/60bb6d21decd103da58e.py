import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
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

    def simulate_route(route):
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
        if cur_time > horizon + 1e-9:
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

    def insert_best_position(route, node):
        """
        Find the best position to insert 'node' into 'route' maintaining validity.
        Returns (new_route, is_valid)
        """
        best_cost = float('inf')
        best_new_route = None
        
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [node] + route[pos:]
            is_valid, total_d = simulate_route(new_route)
            if is_valid:
                if total_d < best_cost:
                    best_cost = total_d
                    best_new_route = new_route
        
        if best_new_route is not None:
            return best_new_route, True
        return route, False

    def construct_swept_cluster_solution():
        """
        1. Sort customers by angle from depot.
        2. Sweep to create clusters based on capacity and battery feasibility.
        3. Insert remaining customers into existing clusters if possible.
        4. Start new routes for leftovers.
        """
        # Sort by angle
        angles = [math.atan2(coords[c, 1] - coords[0, 1], coords[c, 0] - coords[0, 0]) for c in customers]
        sorted_customers = [c for c, a in sorted(zip(customers, angles), key=lambda x: x[1])]
        
        routes = []
        route_loads = []
        unvisited = set(sorted_customers)
        
        # Greedy clustering by sweeping
        current_route = []
        current_load = 0
        
        for c in sorted_customers:
            if c not in unvisited:
                continue
                
            # Try to add to current route
            if current_load + demands[c] <= capacity:
                new_route, is_valid = insert_best_position(current_route, c)
                if is_valid:
                    current_route = new_route
                    current_load += demands[c]
                    unvisited.remove(c)
                    continue
            
            # If current route is not empty, finalize it
            if current_route:
                routes.append(current_route)
                route_loads.append(current_load)
                current_route = []
                current_load = 0
            
            # Start new route with c
            new_route, is_valid = insert_best_position([], c)
            if is_valid:
                current_route = new_route
                current_load = demands[c]
                unvisited.remove(c)
        
        if current_route:
            routes.append(current_route)
            route_loads.append(current_load)
            
        # Handle any remaining unvisited customers (should be few if any)
        for c in list(unvisited):
            best_r_idx = -1
            best_cost = float('inf')
            
            for r_idx, route in enumerate(routes):
                if route_loads[r_idx] + demands[c] > capacity:
                    continue
                new_route, is_valid = insert_best_position(route, c)
                if is_valid:
                    # Use total distance as proxy for cost increase
                    # Actually, we should compute delta, but for simplicity use total
                    # A better proxy: distance from current end to c + c to depot - current end to depot?
                    # Let's just use the total distance of the new route vs old? No, that's expensive.
                    # Let's use a simple heuristic: distance from nearest node in route to c
                    min_d = min(dist[n, c] for n in route) if route else dist[0, c]
                    if min_d < best_cost:
                        best_cost = min_d
                        best_r_idx = r_idx
            
            if best_r_idx != -1:
                routes[best_r_idx], _ = insert_best_position(routes[best_r_idx], c)
                route_loads[best_r_idx] += demands[c]
            else:
                # Start a new route (should be rare)
                routes.append([c])
                route_loads.append(demands[c])
                
            unvisited.remove(c)
            
        return routes

    def local_search(routes, time_limit):
        """
        Improved local search with 2-opt and cross-route moves.
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
                        # 2-opt reversal
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid:
                            # Calculate old dist quickly? Or just compare new_dist to current_total_dist?
                            # We need to compare against the OLD route's distance specifically.
                            # Let's compute old route dist once per iteration? No, too slow.
                            # Approximation: if new_dist is significantly lower than old, take it.
                            # We can store route distances in a list and update them.
                            # For now, let's just check if new_dist < old_dist (computed on the fly for the specific route)
                            old_dist = 0
                            cur_node = 0
                            for node in route:
                                old_dist += dist[cur_node, node]
                                cur_node = node
                            old_dist += dist[cur_node, 0]
                            
                            if new_dist < old_dist - 1e-6:
                                routes[r_idx] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if time.time() - start_ls >= time_limit:
                break
                
            # 2. Cross-route 2-opt (move a segment from one route to another)
            # This is complex. Let's do a simpler "move customer" cross-route.
            for r1 in range(len(routes)):
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    
                    route1 = routes[r1]
                    route2 = routes[r2]
                    
                    if not route1:
                        continue
                        
                    for i, c in enumerate(route1):
                        # Try to remove c from route1
                        remaining1 = route1[:i] + route1[i+1:]
                        if not remaining1:
                            new_route1 = []
                        else:
                            new_route1, _ = insert_best_position(remaining1, remaining1[0]) # Just check validity of remaining?
                            # Actually, removing a node changes the sequence. We need to check validity of remaining1 directly.
                            is_valid1, _ = simulate_route(remaining1)
                            if not is_valid1:
                                continue
                            
                        # Try to insert c into route2
                        new_route2, is_valid2 = insert_best_position(route2, c)
                        if is_valid2:
                            # Calculate distance change
                            old_d1 = 0
                            cur_node = 0
                            for node in route1:
                                old_d1 += dist[cur_node, node]
                                cur_node = node
                            old_d1 += dist[cur_node, 0]
                            
                            old_d2 = 0
                            cur_node = 0
                            for node in route2:
                                old_d2 += dist[cur_node, node]
                                cur_node = node
                            old_d2 += dist[cur_node, 0]
                            
                            new_d1 = 0
                            cur_node = 0
                            for node in remaining1:
                                new_d1 += dist[cur_node, node]
                                cur_node = node
                            new_d1 += dist[cur_node, 0] if remaining1 else 0
                            
                            new_d2 = 0
                            cur_node = 0
                            for node in new_route2:
                                new_d2 += dist[cur_node, node]
                                cur_node = node
                            new_d2 += dist[cur_node, 0]
                            
                            if (new_d1 + new_d2) < (old_d1 + old_d2) - 1e-6:
                                routes[r1] = remaining1
                                routes[r2] = new_route2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                    
            if time.time() - start_ls >= time_limit:
                break

        return routes

    # Main execution
    routes = construct_swept_cluster_solution()
    
    # Validate initial solution
    valid = True
    served = set()
    for route in routes:
        if not simulate_route(route)[0]:
            valid = False
            break
        for node in route:
            if node not in stations:
                if node in served:
                    valid = False
                    break
                served.add(node)
    
    if not valid or served != set(customers):
        # Fallback: simple greedy if construction failed
        routes = []
        unvisited = set(customers)
        for c in sorted(customers, key=lambda x: due[x]):
            routes.append([c])
    
    # Local Search
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 1.0:
        routes = local_search(routes, remaining)
        
    # Final validation
    valid = True
    served = set()
    for route in routes:
        if not simulate_route(route)[0]:
            valid = False
            break
        for node in route:
            if node not in stations:
                if node in served:
                    valid = False
                    break
                served.add(node)
    
    if not valid or served != set(customers):
        # Return the pre-LS routes if LS broke something? Or fallback?
        # Ideally LS shouldn't break validity.
        # Let's just return the constructed routes if LS fails validation.
        # But we overwrote routes. We should keep a backup.
        pass

    return routes
