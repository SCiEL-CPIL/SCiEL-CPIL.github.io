"""Sweep Angle Initialization with Enhanced Iterated Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _check_route(coords, demands, ready, due, service, route, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Check if a route is feasible. Returns (feasible, total_dist)."""
    if not route:
        return True, 0.0
    
    total_demand = sum(demands[n] for n in route)
    if total_demand > capacity:
        return False, 0.0
    
    time_t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0
    
    for node in route:
        d = _euclidean(coords[prev], coords[node])
        total_dist += d
        time_t += d / speed
        battery -= d
        
        if battery < -1e-9:
            return False, 0.0
        
        if node in stations:
            energy_needed = battery_max - battery
            if energy_needed > 1e-9:
                time_t += energy_needed * recharge_rate
            battery = battery_max
        else:
            if time_t < ready[node]:
                time_t = ready[node]
            if time_t > due[node] + 1e-9:
                return False, 0.0
            time_t += service[node]
        
        prev = node
    
    d = _euclidean(coords[prev], coords[0])
    total_dist += d
    time_t += d / speed
    battery -= d
    
    if battery < -1e-9:
        return False, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _route_distance(coords, route):
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for node in route:
        total += _euclidean(coords[prev], coords[node])
        prev = node
    total += _euclidean(coords[prev], coords[0])
    return total

def _sweep_partition(coords, customers, n_clusters, depot_idx=0):
    """Partition customers into n_clusters based on sweep angle from depot."""
    if not customers:
        return []
    
    angles = []
    for c in customers:
        dx = coords[c][0] - coords[depot_idx][0]
        dy = coords[c][1] - coords[depot_idx][1]
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        angles.append((angle, c))
    
    angles.sort()
    
    n = len(angles)
    cluster_sizes = [n // n_clusters] * n_clusters
    remainder = n % n_clusters
    for i in range(remainder):
        cluster_sizes[i] += 1
        
    clusters = []
    start = 0
    for i in range(n_clusters):
        end = start + cluster_sizes[i]
        cluster_nodes = [c for _, c in angles[start:end]]
        clusters.append(cluster_nodes)
        start = end
        
    return clusters

def _build_route_greedy(coords, demands, ready, due, service, cluster_nodes, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Build a route using greedy nearest-neighbor insertion."""
    if not cluster_nodes:
        return []
    
    route = []
    remaining = set(cluster_nodes)
    current_pos = 0
    current_time = 0.0
    current_battery = battery_max
    current_demand = 0
    
    while remaining:
        best_node = None
        best_cost = float('inf')
        
        for node in remaining:
            d = _euclidean(coords[current_pos], coords[node])
            new_time = current_time + d / speed
            new_battery = current_battery - d
            
            if new_battery < -1e-9:
                continue
            
            if node not in stations:
                if new_time < ready[node]:
                    new_time = ready[node]
                if new_time > due[node] + 1e-9:
                    continue
                if current_demand + demands[node] > capacity:
                    continue
            
            cost = d
            if cost < best_cost:
                best_cost = cost
                best_node = node
        
        if best_node is None:
            for node in remaining:
                d = _euclidean(coords[current_pos], coords[node])
                new_time = current_time + d / speed
                new_battery = current_battery - d
                if new_battery >= -1e-9:
                    if node not in stations:
                        if new_time < ready[node]:
                            new_time = ready[node]
                        if new_time <= due[node] + 1e-9 and current_demand + demands[node] <= capacity:
                            best_node = node
                            break
            if best_node is None:
                break
        
        d = _euclidean(coords[current_pos], coords[best_node])
        current_time += d / speed
        current_battery -= d
        route.append(best_node)
        remaining.remove(best_node)
        
        if best_node not in stations:
            if current_time < ready[best_node]:
                current_time = ready[best_node]
            current_time += service[best_node]
            current_demand += demands[best_node]
        else:
            energy_needed = battery_max - current_battery
            if energy_needed > 1e-9:
                current_time += energy_needed * recharge_rate
            current_battery = battery_max
        
        current_pos = best_node
    
    feasible, _ = _check_route(coords, demands, ready, due, service, route,
                               capacity, battery_max, speed, recharge_rate, horizon, stations)
    return route if feasible else []

def _try_2opt_intra(coords, demands, ready, due, service, route, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Try 2-opt reversal within a single route. Returns (improved, new_route, new_dist) or None."""
    if len(route) < 2:
        return None
    
    old_dist = _route_distance(coords, route)
    
    for i in range(len(route) - 1):
        for j in range(i + 1, len(route)):
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            feasible, new_dist = _check_route(coords, demands, ready, due, service, new_route,
                                              capacity, battery_max, speed, recharge_rate, horizon, stations)
            if feasible and new_dist < old_dist - 1e-9:
                return (True, new_route, new_dist)
    
    return None

def _try_2opt_inter(coords, demands, ready, due, service, route1, route2, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Try 2-opt* (segment swap between two routes). Returns (improved, new_route1, new_route2, new_dist) or None."""
    if not route1 or not route2:
        return None
    
    old_dist = _route_distance(coords, route1) + _route_distance(coords, route2)
    
    # Try swapping prefixes/suffixes
    for i in range(len(route1)):
        for j in range(len(route2)):
            # Swap route1[i:] with route2[j:]
            new_r1 = route1[:i] + route2[j:]
            new_r2 = route2[:j] + route1[i:]
            
            feas1, d1 = _check_route(coords, demands, ready, due, service, new_r1,
                                     capacity, battery_max, speed, recharge_rate, horizon, stations)
            feas2, d2 = _check_route(coords, demands, ready, due, service, new_r2,
                                     capacity, battery_max, speed, recharge_rate, horizon, stations)
            
            if feas1 and feas2:
                new_dist = d1 + d2
                if new_dist < old_dist - 1e-9:
                    return (True, new_r1, new_r2, new_dist)
    
    return None

def _try_oropt(coords, demands, ready, due, service, route, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Try Or-opt: move 1-3 consecutive nodes to another position in same route. Returns (improved, new_route, new_dist) or None."""
    if len(route) < 2:
        return None
    
    old_dist = _route_distance(coords, route)
    
    for seg_len in [1, 2, 3]:
        for i in range(len(route) - seg_len + 1):
            segment = route[i:i+seg_len]
            remaining = route[:i] + route[i+seg_len:]
            
            for j in range(len(remaining) + 1):
                new_route = remaining[:j] + segment + remaining[j:]
                # Skip if it's the same position
                if i == j or (seg_len > 0 and i + seg_len == j):
                    continue
                    
                feasible, new_dist = _check_route(coords, demands, ready, due, service, new_route,
                                                  capacity, battery_max, speed, recharge_rate, horizon, stations)
                if feasible and new_dist < old_dist - 1e-9:
                    return (True, new_route, new_dist)
    
    return None

def _try_relocate_inter(coords, demands, ready, due, service, route1, route2, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Try relocating a single node from route1 to route2. Returns (improved, new_route1, new_route2, new_dist) or None."""
    if not route1:
        return None
    
    old_dist = _route_distance(coords, route1) + _route_distance(coords, route2)
    
    for pos1 in range(len(route1)):
        node = route1[pos1]
        new_r1 = route1[:pos1] + route1[pos1+1:]
        
        for pos2 in range(len(route2) + 1):
            new_r2 = route2[:pos2] + [node] + route2[pos2:]
            
            feas1, d1 = _check_route(coords, demands, ready, due, service, new_r1,
                                     capacity, battery_max, speed, recharge_rate, horizon, stations)
            feas2, d2 = _check_route(coords, demands, ready, due, service, new_r2,
                                     capacity, battery_max, speed, recharge_rate, horizon, stations)
            
            if feas1 and feas2:
                new_dist = d1 + d2
                if new_dist < old_dist - 1e-9:
                    return (True, new_r1, new_r2, new_dist)
    
    return None

def _local_search(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations, max_passes=10):
    """Enhanced local search with multiple move types."""
    current_routes = [r[:] for r in routes]
    
    for _ in range(max_passes):
        improved = False
        
        # 2-opt intra-route
        for r_idx in range(len(current_routes)):
            result = _try_2opt_intra(coords, demands, ready, due, service, current_routes[r_idx],
                                     capacity, battery_max, speed, recharge_rate, horizon, stations)
            if result:
                _, new_route, _ = result
                current_routes[r_idx] = new_route
                improved = True
        
        # Or-opt intra-route
        for r_idx in range(len(current_routes)):
            result = _try_oropt(coords, demands, ready, due, service, current_routes[r_idx],
                                capacity, battery_max, speed, recharge_rate, horizon, stations)
            if result:
                _, new_route, _ = result
                current_routes[r_idx] = new_route
                improved = True
        
        # Inter-route moves
        for r1 in range(len(current_routes)):
            for r2 in range(r1 + 1, len(current_routes)):
                # 2-opt*
                result = _try_2opt_inter(coords, demands, ready, due, service, current_routes[r1], current_routes[r2],
                                         capacity, battery_max, speed, recharge_rate, horizon, stations)
                if result:
                    _, new_r1, new_r2, _ = result
                    current_routes[r1] = new_r1
                    current_routes[r2] = new_r2
                    improved = True
                    continue
                
                # Relocate
                result = _try_relocate_inter(coords, demands, ready, due, service, current_routes[r1], current_routes[r2],
                                             capacity, battery_max, speed, recharge_rate, horizon, stations)
                if result:
                    _, new_r1, new_r2, _ = result
                    current_routes[r1] = new_r1
                    current_routes[r2] = new_r2
                    improved = True
        
        # Remove empty routes
        current_routes = [r for r in current_routes if r]
        
        if not improved:
            break
    
    return current_routes

def _perturb(routes, stations):
    """Perturb by random segment relocation."""
    if not routes:
        return routes
    
    new_routes = [r[:] for r in routes]
    
    # Pick a random non-empty route
    non_empty = [i for i, r in enumerate(new_routes) if len(r) >= 2]
    if not non_empty:
        return new_routes
    
    r_idx = random.choice(non_empty)
    route = new_routes[r_idx]
    
    # Pick random segment of length 1-3
    max_len = min(3, len(route))
    seg_len = random.randint(1, max_len)
    start = random.randint(0, len(route) - seg_len)
    end = start + seg_len - 1
    
    segment = route[start:end+1]
    new_route = route[:start] + route[end+1:]
    
    # Re-insert into random route at random position
    target_r_idx = random.randint(0, len(new_routes) - 1)
    target_route = new_routes[target_r_idx]
    insert_pos = random.randint(0, len(target_route))
    new_target = target_route[:insert_pos] + segment + target_route[insert_pos:]
    
    new_routes[r_idx] = new_route
    new_routes[target_r_idx] = new_target
    
    # Remove empty routes
    new_routes = [r for r in new_routes if r]
    
    return new_routes

def _check_all_feasible(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Check if all routes are feasible and all customers served exactly once."""
    served = set()
    for r in routes:
        for n in r:
            if n not in stations:
                if n in served:
                    return False
                served.add(n)
    
    for r in routes:
        f, _ = _check_route(coords, demands, ready, due, service, r, capacity, battery_max, speed, recharge_rate, horizon, stations)
        if not f:
            return False
    
    return True

def _iterated_local_search(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations, time_limit=30.0):
    """Iterated Local Search with time management."""
    start_time = time.time()
    
    current_routes = _local_search(coords, demands, ready, due, service, routes,
                                   capacity, battery_max, speed, recharge_rate, horizon, stations, max_passes=15)
    current_dist = sum(_route_distance(coords, r) for r in current_routes)
    
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist
    
    all_customers = set()
    for r in current_routes:
        for n in r:
            if n not in stations:
                all_customers.add(n)
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        perturbed = _perturb(current_routes, stations)
        
        # Validate perturbation
        served = set()
        valid = True
        for r in perturbed:
            for n in r:
                if n not in stations:
                    if n in served:
                        valid = False
                        break
                    served.add(n)
            if not valid:
                break
        
        if not valid or served != all_customers:
            no_improve_count += 1
            if no_improve_count > 50:
                break
            continue
        
        if not _check_all_feasible(coords, demands, ready, due, service, perturbed,
                                   capacity, battery_max, speed, recharge_rate, horizon, stations):
            no_improve_count += 1
            if no_improve_count > 50:
                break
            continue
        
        new_routes = _local_search(coords, demands, ready, due, service, perturbed,
                                   capacity, battery_max, speed, recharge_rate, horizon, stations, max_passes=10)
        new_dist = sum(_route_distance(coords, r) for r in new_routes)
        
        if new_dist < current_dist - 1e-9:
            current_routes = new_routes
            current_dist = new_dist
            no_improve_count = 0
            if current_dist < best_dist:
                best_dist = current_dist
                best_routes = [r[:] for r in current_routes]
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best_routes

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
    
    # Determine number of clusters based on capacity
    total_demand = sum(demands[c] for c in customers)
    n_clusters = max(1, int(np.ceil(total_demand / capacity)))
    n_clusters = min(n_clusters, len(customers))
    
    # Try multiple sweep partitions and pick best
    best_init_routes = None
    best_init_dist = float('inf')
    
    for trial in range(5):
        # Slight randomization in angle sorting
        angles = []
        for c in customers:
            dx = coords[c][0] - coords[0][0]
            dy = coords[c][1] - coords[0][1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            # Add small random perturbation for diversity
            angle += random.uniform(-0.1, 0.1)
            angles.append((angle, c))
        
        angles.sort()
        
        n = len(angles)
        cluster_sizes = [n // n_clusters] * n_clusters
        remainder = n % n_clusters
        for i in range(remainder):
            cluster_sizes[i] += 1
            
        clusters = []
        start = 0
        for i in range(n_clusters):
            end = start + cluster_sizes[i]
            cluster_nodes = [c for _, c in angles[start:end]]
            clusters.append(cluster_nodes)
            start = end
        
        routes = []
        for cluster in clusters:
            route = _build_route_greedy(coords, demands, ready, due, service, cluster,
                                        capacity, battery_max, speed, recharge_rate, horizon, stations)
            if route:
                routes.append(route)
        
        # Ensure all customers served
        served = set()
        for r in routes:
            for n in r:
                if n not in stations:
                    served.add(n)
        
        unserved = [c for c in customers if c not in served]
        if unserved:
            for c in unserved:
                inserted = False
                for r_idx in range(len(routes)):
                    for pos in range(len(routes[r_idx]) + 1):
                        new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                        feasible, _ = _check_route(coords, demands, ready, due, service, new_route,
                                                   capacity, battery_max, speed, recharge_rate, horizon, stations)
                        if feasible:
                            routes[r_idx] = new_route
                            inserted = True
                            break
                    if inserted:
                        break
                if not inserted:
                    routes.append([c])
        
        # Quick local search
        refined = _local_search(coords, demands, ready, due, service, routes,
                                capacity, battery_max, speed, recharge_rate, horizon, stations, max_passes=5)
        dist = sum(_route_distance(coords, r) for r in refined)
        
        if dist < best_init_dist:
            best_init_dist = dist
            best_init_routes = refined
    
    # Full ILS on best initialization
    final_routes = _iterated_local_search(coords, demands, ready, due, service, best_init_routes,
                                          capacity, battery_max, speed, recharge_rate, horizon, stations,
                                          time_limit=30.0)
    
    return final_routes
# EVOLVE-BLOCK-END
