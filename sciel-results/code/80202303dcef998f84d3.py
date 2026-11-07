"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

def _dist(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.sqrt(dx*dx + dy*dy)

def _build_dist_matrix(instance):
    coords = instance["coords"]
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _dist(coords, i, j)
            D[i][j] = d
            D[j][i] = d
    return D

def _check_route(route, instance, D):
    """Check if a route is feasible. Returns (is_feasible, total_dist)."""
    n_customers = instance["n_customers"]
    demands = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Check demand
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0
    
    # Simulate the route
    cur_node = 0
    cur_time = 0.0
    cur_battery = float(battery_cap)
    total_dist = 0.0
    
    for v in route:
        d = D[cur_node][v]
        total_dist += d
        cur_time += d / speed
        cur_battery -= d
        
        if cur_battery < -1e-9:
            return False, 0.0
        
        if v in stations:
            # Recharge
            energy_needed = battery_cap - cur_battery
            cur_time += energy_needed * recharge_rate
            cur_battery = float(battery_cap)
        else:
            # Customer
            if v > n_customers:
                # This shouldn't happen if route only has customers
                pass
            # Time window
            if cur_time < ready[v]:
                cur_time = float(ready[v])
            if cur_time > due[v] + 1e-9:
                return False, 0.0
            cur_time += service[v]
        
        cur_node = v
    
    # Return to depot
    d = D[cur_node][0]
    total_dist += d
    cur_time += d / speed
    cur_battery -= d
    
    if cur_battery < -1e-9:
        return False, 0.0
    if cur_time > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _route_distance(route, instance, D):
    """Compute total distance of a route including depot legs."""
    if not route:
        return 0.0
    total = D[0][route[0]]
    for i in range(len(route) - 1):
        total += D[route[i]][route[i+1]]
    total += D[route[-1]][0]
    return total

def _nearest_neighbor_cluster(customers, instance, D, max_nodes=5):
    """Create a cluster using nearest neighbor heuristic."""
    if not customers:
        return []
    
    # Start from the customer with earliest due time
    start = min(customers, key=lambda v: instance["due"][v])
    cluster = [start]
    remaining = set(customers) - {start}
    
    while remaining and len(cluster) < max_nodes:
        last = cluster[-1]
        # Find nearest neighbor in remaining
        nearest = min(remaining, key=lambda v: D[last][v])
        cluster.append(nearest)
        remaining.remove(nearest)
    
    return cluster

def _try_insert(customer, route, instance, D):
    """Try to insert customer into route at best position. Returns (pos, feasible, new_dist)."""
    n_customers = instance["n_customers"]
    demands = instance["demand"]
    capacity = instance["capacity"]
    
    # Check capacity
    current_demand = sum(demands[v] for v in route if v <= n_customers)
    if current_demand + demands[customer] > capacity:
        return -1, False, float('inf')
    
    best_pos = -1
    best_delta = float('inf')
    
    for pos in range(len(route) + 1):
        # Try inserting at position pos
        new_route = route[:pos] + [customer] + route[pos:]
        feasible, dist = _check_route(new_route, instance, D)
        if feasible:
            old_dist = _route_distance(route, instance, D) if route else 0.0
            delta = dist - old_dist
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
    
    if best_pos == -1:
        return -1, False, float('inf')
    
    return best_pos, True, best_delta

def _local_search(routes, instance, D, max_iterations=50):
    """Perform 2-opt and relocate moves to improve routes."""
    n_customers = instance["n_customers"]
    
    for _ in range(max_iterations):
        improved = False
        
        # Try relocating customers between routes
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                customer = routes[i][j]
                # Remove from route i
                new_route_i = routes[i][:j] + routes[i][j+1:]
                feasible_i, _ = _check_route(new_route_i, instance, D)
                if not feasible_i:
                    continue
                
                # Try inserting into other routes
                for k in range(len(routes)):
                    if k == i:
                        continue
                    pos, feasible, delta = _try_insert(customer, routes[k], instance, D)
                    if feasible:
                        # Check if this improves total distance
                        old_dist_i = _route_distance(routes[i], instance, D)
                        old_dist_k = _route_distance(routes[k], instance, D)
                        new_dist_i = _route_distance(new_route_i, instance, D)
                        new_route_k = routes[k][:pos] + [customer] + routes[k][pos:]
                        new_dist_k = _route_distance(new_route_k, instance, D)
                        
                        if new_dist_i + new_dist_k < old_dist_i + old_dist_k - 1e-9:
                            routes[i] = new_route_i
                            routes[k] = new_route_k
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return routes

def _cluster_and_solve(instance, deadline):
    """Cluster customers and solve each cluster independently."""
    D = _build_dist_matrix(instance)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Sort by due time for better clustering
    customers.sort(key=lambda v: instance["due"][v])
    
    # Create clusters using nearest neighbor
    routes = []
    remaining = list(customers)
    
    while remaining:
        cluster = _nearest_neighbor_cluster(remaining, instance, D, max_nodes=4)
        if not cluster:
            break
        
        # Sort cluster by due time
        cluster.sort(key=lambda v: instance["due"][v])
        
        # Check if the cluster route is feasible
        feasible, dist = _check_route(cluster, instance, D)
        if feasible:
            routes.append(cluster)
            remaining = [c for c in remaining if c not in set(cluster)]
        else:
            # Try smaller cluster
            if len(cluster) > 1:
                cluster = cluster[:len(cluster)//2 + 1]
                cluster.sort(key=lambda v: instance["due"][v])
                feasible, dist = _check_route(cluster, instance, D)
                if feasible:
                    routes.append(cluster)
                    remaining = [c for c in remaining if c not in set(cluster)]
                else:
                    # Single customer
                    single = [cluster[0]]
                    feasible, dist = _check_route(single, instance, D)
                    if feasible:
                        routes.append(single)
                        remaining = [c for c in remaining if c != cluster[0]]
                    else:
                        # This shouldn't happen, but handle it
                        remaining.remove(cluster[0])
            else:
                # Single customer
                single = [cluster[0]]
                feasible, dist = _check_route(single, instance, D)
                if feasible:
                    routes.append(single)
                remaining = [c for c in remaining if c != cluster[0]]
    
    # Local search
    if time.time() < deadline - 5:
        routes = _local_search(routes, instance, D, max_iterations=20)
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 120  # Leave margin for safety
    
    # Build distance matrix
    D = _build_dist_matrix(instance)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Try clustering approach
    routes = _cluster_and_solve(instance, deadline)
    
    # Verify all customers are served
    served = set()
    for route in routes:
        for v in route:
            if v <= n_customers:
                served.add(v)
    
    # Add any missing customers as single routes
    for c in customers:
        if c not in served:
            routes.append([c])
    
    # Final verification
    valid_routes = []
    for route in routes:
        feasible, dist = _check_route(route, instance, D)
        if feasible:
            valid_routes.append(route)
    
    return valid_routes
# EVOLVE-BLOCK-END
