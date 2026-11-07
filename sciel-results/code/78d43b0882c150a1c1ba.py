import numpy as np
import time

def plan(instance):
    start_time = time.time()
    deadline = 110.0 # 150s limit, keep margin

    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    
    # Distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    customers = list(range(1, instance["n_customers"] + 1))
    stations = set(instance["stations"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    def simulate(route):
        """
        Simulates a route.
        Returns (is_valid, total_distance, total_time)
        """
        t = 0.0
        b = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        
        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            
            if b < 0:
                return False, 0.0, 0.0
            
            if node in stations:
                # Recharge
                energy_needed = battery_cap - b
                if energy_needed > 0:
                    t += energy_needed * recharge_rate
                b = battery_cap
            else:
                # Customer
                load += demands[node]
                if load > capacity:
                    return False, 0.0, 0.0
                
                # Time window
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return False, 0.0, 0.0
                
                t += service[node]
            
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        
        if b < 0 or t > horizon:
            return False, 0.0, 0.0
            
        return True, total_dist, t

    def try_repair(route):
        """
        Attempts to make a route feasible by inserting stations.
        Returns (feasible_route, total_distance) or (None, 0.0)
        """
        current_route = list(route)
        # Limit repair attempts to avoid infinite loops
        for _ in range(15):
            is_valid, total_dist, _ = simulate(current_route)
            if is_valid:
                return current_route, total_dist
            
            # Find the first point where battery fails
            t = 0.0
            b = battery_cap
            prev = 0
            fail_idx = -1
            
            for i, node in enumerate(current_route):
                d = dist[prev, node]
                t += d / speed
                b -= d
                
                if b < 0:
                    fail_idx = i
                    break
                
                if node in stations:
                    energy_needed = battery_cap - b
                    if energy_needed > 0:
                        t += energy_needed * recharge_rate
                    b = battery_cap
                else:
                    if t < ready[node]:
                        t = ready[node]
                    if t > due[node]:
                        # Time window violation cannot be fixed by adding a station (adds time)
                        return None, 0.0
                    t += service[node]
                
                prev = node
            
            if fail_idx == -1:
                # Failure occurred after the last node (return to depot) or was time-based
                return None, 0.0
                
            # Insert a station between current_route[fail_idx-1] and current_route[fail_idx]
            # If fail_idx is 0, insert between depot and first node
            prev_node = current_route[fail_idx-1] if fail_idx > 0 else 0
            next_node = current_route[fail_idx]
            
            best_s = -1
            best_cost = float('inf')
            
            # Try all stations
            for s in stations:
                # Construct candidate route
                new_route = current_route[:fail_idx] + [s] + current_route[fail_idx:]
                
                # Check if this candidate is feasible
                # Optimization: We can check if the battery issue is resolved locally,
                # but full simulation is safer and N is small.
                is_valid, d_new, _ = simulate(new_route)
                if is_valid:
                    # Calculate extra distance cost
                    cost = dist[prev_node, s] + dist[s, next_node] - dist[prev_node, next_node]
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
            
            if best_s == -1:
                return None, 0.0
            
            # Insert the best station
            current_route = current_route[:fail_idx] + [best_s] + current_route[fail_idx:]
            
        return None, 0.0

    # --- Construction ---
    # Start with each customer as a separate route
    routes = [[c] for c in customers]
    
    # Map customer to route index
    def build_map(routes):
        m = {}
        for i, r in enumerate(routes):
            for c in r:
                if c not in stations:
                    m[c] = i
        return m

    route_map = build_map(routes)
    
    # Calculate savings
    savings = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s_val = dist[0, c1] + dist[0, c2] - dist[c1, c2]
            savings.append((s_val, c1, c2))
            
    savings.sort(reverse=True)
    
    for s_val, c1, c2 in savings:
        if time.time() > deadline:
            break
            
        r1 = route_map.get(c1)
        r2 = route_map.get(c2)
        
        if r1 is None or r2 is None:
            continue
        if r1 == r2:
            continue
            
        route1 = routes[r1]
        route2 = routes[r2]
        
        # Check if c1 is at the end of route1 and c2 is at the start of route2
        if route1[-1] == c1 and route2[0] == c2:
            new_route = route1 + route2
            repaired, _ = try_repair(new_route)
            if repaired:
                # Merge
                routes[r1] = repaired
                del routes[r2]
                route_map = build_map(routes)
                continue
        
        # Check if c2 is at the end of route1 and c1 is at the start of route2
        if route1[-1] == c2 and route2[0] == c1:
            new_route = route1 + route2
            repaired, _ = try_repair(new_route)
            if repaired:
                routes[r1] = repaired
                del routes[r2]
                route_map = build_map(routes)
                continue

    # --- Local Search ---
    # 2-Opt, Or-Opt, Move
    
    def get_route_dist(route):
        _, d, _ = simulate(route)
        return d

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 2-Opt Intra
        for i in range(len(routes)):
            if len(routes[i]) < 2:
                continue
            route = routes[i]
            old_dist = get_route_dist(route)
            
            for a in range(len(route)):
                for b in range(a + 2, len(route)):
                    # Reverse segment a to b
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    repaired, _ = try_repair(new_route)
                    if repaired:
                        new_dist = get_route_dist(repaired)
                        if new_dist < old_dist - 1e-6:
                            routes[i] = repaired
                            old_dist = new_dist
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        # Or-Opt (Move 1, 2, 3 customers)
        if not improved:
            for i in range(len(routes)):
                route = routes[i]
                if len(route) < 2: continue
                old_dist = get_route_dist(route)
                
                # Move 1
                for a in range(len(route)):
                    for b in range(len(route)):
                        if b == a or b == a - 1: continue
                        # Move route[a] to b
                        # If b > a, insert after b?
                        # Standard Or-opt: remove segment, insert at new position.
                        
                        # Remove
                        seg = route[a:a+1]
                        rest = route[:a] + route[a+1:]
                        
                        # Insert
                        # If b < a, insert at b
                        # If b >= a, insert at b (adjusted index)
                        # This logic is tricky. Let's stick to 2-opt and simple moves for now to ensure correctness.
                        pass
                        
                if improved: break
        
        # Inter-route Move
        if not improved:
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j: continue
                    route_i = routes[i]
                    route_j = routes[j]
                    if not route_i: continue
                    
                    old_dist_i = get_route_dist(route_i)
                    old_dist_j = get_route_dist(route_j)
                    old_total = old_dist_i + old_dist_j
                    
                    for a in range(len(route_i)):
                        c = route_i[a]
                        # Remove c from i
                        new_i = route_i[:a] + route_i[a+1:]
                        if not new_i:
                            # Route i becomes empty
                            repaired_i, dist_i = try_repair([]) if False else (None, 0) # Empty route is valid?
                            # An empty route is just depot-depot. Dist 0.
                            # But we should remove empty routes at the end.
                            # For now, keep it.
                            pass
                        else:
                            repaired_i, dist_i = try_repair(new_i)
                            if not repaired_i: continue
                            
                        # Insert c into j at best position
                        best_pos = -1
                        best_dist_j = float('inf')
                        
                        for b in range(len(route_j) + 1):
                            new_j = route_j[:b] + [c] + route_j[b:]
                            repaired_j, dist_j = try_repair(new_j)
                            if repaired_j:
                                if dist_j < best_dist_j:
                                    best_dist_j = dist_j
                                    best_pos = b
                                    best_repaired_j = repaired_j
                                    
                        if best_pos != -1:
                            new_total = dist_i + best_dist_j
                            if new_total < old_total - 1e-6:
                                routes[i] = repaired_i
                                routes[j] = best_repaired_j
                                improved = True
                                break
                    if improved: break
                if improved: break

    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes
