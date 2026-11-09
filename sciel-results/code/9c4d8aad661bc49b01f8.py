"""Cluster-based EV routing with Simulated Annealing refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def _euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


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
        
        if node in stations:
            energy_needed = battery_max - battery
            if energy_needed > 1e-9:
                time += energy_needed * recharge_rate
            battery = battery_max
        else:
            if time < ready[node]:
                time = ready[node]
            if time > due[node] + 1e-9:
                return False, 0.0
            time += service[node]
        
        prev = node
    
    d = _euclidean(coords[prev], coords[0])
    total_dist += d
    time += d / speed
    battery -= d
    
    if battery < -1e-9:
        return False, 0.0
    if time > horizon + 1e-9:
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


def _kmeans_cluster(points, k, n_iter=50, seed=42):
    """Simple k-means clustering."""
    rng = np.random.RandomState(seed)
    n = len(points)
    if k >= n:
        return [list(range(n))]
    
    # Initialize centroids randomly
    centroids = points[rng.choice(n, k, replace=False)].copy()
    
    for _ in range(n_iter):
        # Assign points to nearest centroid
        dists = np.array([[np.sum((points[i] - centroids[j])**2) for j in range(k)] for i in range(n)])
        assignments = np.argmin(dists, axis=1)
        
        # Update centroids
        new_centroids = centroids.copy()
        for j in range(k):
            mask = assignments == j
            if np.sum(mask) > 0:
                new_centroids[j] = np.mean(points[mask], axis=0)
            else:
                # Reinitialize empty cluster
                new_centroids[j] = points[rng.randint(n)]
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Group points by assignment
    clusters = [[] for _ in range(k)]
    for i, a in enumerate(assignments):
        clusters[a].append(i)
    
    return [c for c in clusters if c]


def _build_route_nn(coords, demands, ready, due, service, cluster_nodes, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Build a route using nearest-neighbor with time window awareness."""
    if not cluster_nodes:
        return []
    
    remaining = list(cluster_nodes)
    route = []
    current_pos = 0
    current_time = 0.0
    current_battery = battery_max
    current_demand = 0
    
    while remaining:
        best_node = None
        best_dist = float('inf')
        
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
            
            if d < best_dist:
                best_dist = d
                best_node = node
        
        if best_node is None:
            # Fallback: pick any feasible node
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
        
        # Move to best_node
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
    
    # Check if route is feasible
    feasible, _ = _check_route(coords, demands, ready, due, service, route,
                               capacity, battery_max, speed, recharge_rate, horizon, stations)
    return route if feasible else []


def _simulated_annealing(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations, n_iter=2000, t_start=1.0, t_end=0.01):
    """Simulated Annealing to optimize routes."""
    if not routes:
        return routes
    
    current_routes = [r[:] for r in routes]
    current_dist = sum(_route_distance(coords, r) for r in current_routes)
    
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max(n_iter, 1))
    
    all_customers = set()
    for r in current_routes:
        for n in r:
            if n not in stations:
                all_customers.add(n)
    
    for _ in range(n_iter):
        t *= cooling
        
        # Choose a random move
        move_type = random.choice(['2opt', 'relocate', 'swap', 'cross_relocate'])
        new_routes = [r[:] for r in current_routes]
        
        if move_type == '2opt':
            if len(current_routes) > 0:
                r_idx = random.randint(0, len(current_routes) - 1)
                route = current_routes[r_idx]
                if len(route) >= 2:
                    i = random.randint(0, len(route) - 2)
                    j = random.randint(i + 1, len(route) - 1)
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_routes[r_idx] = new_route
        
        elif move_type == 'relocate':
            if len(current_routes) >= 1:
                r_idx = random.randint(0, len(current_routes) - 1)
                route = current_routes[r_idx]
                if len(route) >= 1:
                    pos = random.randint(0, len(route) - 1)
                    node = route[pos]
                    new_source = route[:pos] + route[pos+1:]
                    
                    if len(current_routes) >= 2:
                        r2_idx = random.choice([i for i in range(len(current_routes)) if i != r_idx])
                        target = current_routes[r2_idx]
                        new_target = target[:random.randint(0, len(target))] + [node] + target[random.randint(0, len(target)):]
                        
                        if new_source:
                            new_routes[r_idx] = new_source
                            new_routes[r2_idx] = new_target
                        else:
                            new_routes[r2_idx] = new_target
                            new_routes.pop(r_idx)
                    
                    else:
                        new_routes[r_idx] = new_source
        
        elif move_type == 'swap':
            if len(current_routes) >= 2:
                r1, r2 = random.sample(range(len(current_routes)), 2)
                if len(current_routes[r1]) >= 1 and len(current_routes[r2]) >= 1:
                    p1 = random.randint(0, len(current_routes[r1]) - 1)
                    p2 = random.randint(0, len(current_routes[r2]) - 1)
                    n1 = current_routes[r1][p1]
                    n2 = current_routes[r2][p2]
                    new_routes[r1][p1] = n2
                    new_routes[r2][p2] = n1
        
        elif move_type == 'cross_relocate':
            if len(current_routes) >= 2:
                r1, r2 = random.sample(range(len(current_routes)), 2)
                if len(current_routes[r1]) >= 1 and len(current_routes[r2]) >= 1:
                    p1 = random.randint(0, len(current_routes[r1]) - 1)
                    p2 = random.randint(0, len(current_routes[r2]) - 1)
                    node = current_routes[r1][p1]
                    new_source = current_routes[r1][:p1] + current_routes[r1][p1+1:]
                    target = current_routes[r2]
                    new_target = target[:random.randint(0, len(target))] + [node] + target[random.randint(0, len(target)):]
                    
                    if new_source:
                        new_routes[r1] = new_source
                        new_routes[r2] = new_target
                    else:
                        new_routes[r2] = new_target
                        new_routes.pop(r1)
        
        # Validate new routes
        valid = True
        new_dist = 0.0
        served = set()
        
        for r in new_routes:
            feasible, dist = _check_route(coords, demands, ready, due, service, r,
                                          capacity, battery_max, speed, recharge_rate, horizon, stations)
            if not feasible:
                valid = False
                break
            new_dist += dist
            for n in r:
                if n not in stations:
                    if n in served:
                        valid = False
                        break
                    served.add(n)
        
        if valid and served == all_customers:
            delta = new_dist - current_dist
            if delta < 0 or random.random() < np.exp(-delta / max(t, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_routes = [r[:] for r in current_routes]
    
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
    
    # Cluster customers
    customer_coords = np.array([coords[c] for c in customers])
    clusters = _kmeans_cluster(customer_coords, n_clusters, n_iter=50, seed=42)
    
    # Build routes for each cluster
    routes = []
    for cluster in clusters:
        cluster_nodes = [customers[i] for i in cluster]
        route = _build_route_nn(coords, demands, ready, due, service, cluster_nodes,
                                capacity, battery_max, speed, recharge_rate, horizon, stations)
        if route:
            routes.append(route)
    
    # Ensure all customers are served
    served = set()
    for r in routes:
        for n in r:
            if n not in stations:
                served.add(n)
    
    unserved = [c for c in customers if c not in served]
    if unserved:
        # Add unserved customers to existing routes or create new ones
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
    
    # Simulated Annealing refinement
    routes = _simulated_annealing(coords, demands, ready, due, service, routes,
                                  capacity, battery_max, speed, recharge_rate, horizon, stations,
                                  n_iter=3000, t_start=1.0, t_end=0.01)
    
    return routes
# EVOLVE-BLOCK-END
