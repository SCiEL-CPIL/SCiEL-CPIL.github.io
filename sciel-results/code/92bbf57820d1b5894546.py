"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt, inf

def _dist(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return sqrt(dx*dx + dy*dy)

def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = sqrt(dx*dx + dy*dy)
            D[i, j] = d
            D[j, i] = d
    return D

def _simulate_sequence(seq, instance, D):
    """
    Simulates a route defined by a sequence of customer IDs.
    Handles charging automatically by finding the best station if battery is low.
    Returns (is_feasible, total_distance).
    """
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    horizon = instance["horizon"]
    recharge_rate = instance["recharge_rate"]
    speed = instance["speed"]
    
    # Check capacity
    total_demand = sum(instance["demands"][c] for c in seq)
    if total_demand > capacity:
        return False, inf

    time = 0.0
    battery = float(battery_max)
    curr = 0
    total_dist = 0.0
    
    for c in seq:
        d_direct = D[curr, c]
        
        # Check if we can go directly
        if battery >= d_direct:
            # Check time window
            arrival_time = time + d_direct / speed
            if arrival_time < instance["ready"][c]:
                arrival_time = instance["ready"][c]
            if arrival_time > instance["due"][c] + 1e-9:
                return False, inf
            
            # Move
            time = arrival_time + instance["service"][c]
            battery -= d_direct
            total_dist += d_direct
            curr = c
        else:
            # Need to charge. Find best station.
            best_station = None
            best_cost = inf # We want to minimize distance, but must be feasible
            
            for s in stations:
                d_to_s = D[curr, s]
                if battery < d_to_s:
                    continue
                
                # After charging, battery is max.
                # Can we reach c from s?
                d_s_to_c = D[s, c]
                if battery_max < d_s_to_c:
                    continue
                
                # Check time
                # Time to reach s
                t_to_s = time + d_to_s / speed
                # Charging time
                energy_needed = battery_max - (battery - d_to_s)
                charge_time = energy_needed * recharge_rate
                t_after_charge = t_to_s + charge_time
                # Travel to c
                t_to_c = t_after_charge + d_s_to_c / speed
                
                if t_to_c < instance["ready"][c]:
                    t_to_c = instance["ready"][c]
                
                if t_to_c > instance["due"][c] + 1e-9:
                    continue
                
                # Feasible. Cost is distance: d_to_s + d_s_to_c
                cost = d_to_s + d_s_to_c
                if cost < best_cost:
                    best_cost = cost
                    best_station = s
                    best_t_to_c = t_to_c
                    best_d_to_s = d_to_s
                    best_d_s_to_c = d_s_to_c
                    best_charge_time = charge_time
                    best_t_to_s = t_to_s

            if best_station is None:
                return False, inf
            
            # Execute move via station
            time = best_t_to_c + instance["service"][c]
            battery = battery_max - best_d_s_to_c
            total_dist += best_cost
            curr = c

    # Return to depot
    d_back = D[curr, 0]
    if battery >= d_back:
        arrival_time = time + d_back / speed
        if arrival_time > horizon + 1e-9:
            return False, inf
        total_dist += d_back
    else:
        # Need to charge before returning
        best_station = None
        best_cost = inf
        for s in stations:
            d_to_s = D[curr, s]
            if battery < d_to_s:
                continue
            d_s_to_0 = D[s, 0]
            if battery_max < d_s_to_0:
                continue
            
            t_to_s = time + d_to_s / speed
            energy_needed = battery_max - (battery - d_to_s)
            charge_time = energy_needed * recharge_rate
            t_after_charge = t_to_s + charge_time
            t_to_0 = t_after_charge + d_s_to_0 / speed
            
            if t_to_0 > horizon + 1e-9:
                continue
            
            cost = d_to_s + d_s_to_0
            if cost < best_cost:
                best_cost = cost
                best_station = s
                best_t_to_0 = t_to_0
                best_d_to_s = d_to_s
                best_d_s_to_0 = d_s_to_0
                best_charge_time = charge_time
                best_t_to_s = t_to_s

        if best_station is None:
            return False, inf
        
        total_dist += best_cost
        # time = best_t_to_0 # Not needed for distance score, but good for completeness
    
    return True, total_dist

def _kmeans_cluster(points, k, n_iter=10):
    """Simple K-Means clustering."""
    n = len(points)
    if k >= n:
        return [[i] for i in range(n)]
    
    # Initialize centroids randomly
    indices = np.random.choice(n, k, replace=False)
    centroids = np.array([points[i] for i in indices])
    
    for _ in range(n_iter):
        # Assign points to nearest centroid
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.linalg.norm(centroids - points[i], axis=1)
            labels[i] = np.argmin(dists)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if np.sum(mask) > 0:
                new_centroids[j] = np.mean(points[mask], axis=0)
            else:
                # Keep old centroid
                new_centroids[j] = centroids[j]
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Group indices
    clusters = [[] for _ in range(k)]
    for i in range(n):
        clusters[labels[i]].append(i)
        
    return clusters

def plan(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    horizon = instance["horizon"]
    recharge_rate = instance["recharge_rate"]
    speed = instance["speed"]
    
    D = _build_dist_matrix(coords)
    customers = list(range(1, n_customers + 1))
    
    # Determine number of routes
    # Average demand
    avg_demand = np.mean(instance["demands"][1:n_customers+1])
    max_customers_per_route = int(capacity / avg_demand)
    # Ensure at least 1 customer per route, and not too many routes
    k_routes = max(1, int(np.ceil(n_customers / max_customers_per_route)))
    k_routes = min(k_routes, n_customers)
    
    # Cluster customers
    # Use customer coords
    cust_coords = np.array([coords[i] for i in customers])
    clusters = _kmeans_cluster(cust_coords, k_routes)
    
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        
        # Initial sequence: Sort by angle from depot or just nearest neighbor
        # Let's try Nearest Neighbor starting from depot
        seq = []
        remaining = set(cluster)
        curr = 0
        battery = float(battery_max)
        time = 0.0
        
        while remaining:
            best_next = None
            best_dist = inf
            
            for c in remaining:
                d = D[curr, c]
                # Rough check: if battery is enough, consider it
                # We'll rely on _simulate_sequence for final feasibility
                # For NN, just pick closest
                if d < best_dist:
                    best_dist = d
                    best_next = c
            
            if best_next is None:
                break
                
            seq.append(best_next)
            remaining.remove(best_next)
            
            # Update state roughly for NN (not strictly accurate for battery, but guides order)
            # We don't update battery here to keep it simple, NN is just for initial order
            curr = best_next
        
        # Refine sequence with 2-opt
        # We will use _simulate_sequence to evaluate
        current_seq = seq
        current_feasible, current_dist = _simulate_sequence(current_seq, instance, D)
        
        if not current_feasible:
            # If initial NN is infeasible, try other heuristics or just return empty/fix later
            # For now, try sorting by due time
            current_seq = sorted(cluster, key=lambda x: instance["due"][x])
            current_feasible, current_dist = _simulate_sequence(current_seq, instance, D)
            
            if not current_feasible:
                # Last resort: try to split or just keep and hope?
                # If infeasible, we might need to move customers to other routes.
                # For simplicity in this version, we'll try a few random permutations if small
                if len(cluster) <= 8:
                    import random
                    for _ in range(100):
                        random.shuffle(current_seq)
                        f, d = _simulate_sequence(current_seq, instance, D)
                        if f:
                            current_feasible = True
                            current_dist = d
                            break
        
        if current_feasible:
            # 2-opt improvement
            improved = True
            while improved:
                improved = False
                n = len(current_seq)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        new_seq = current_seq[:i] + current_seq[i:j+1][::-1] + current_seq[j+1:]
                        f, d = _simulate_sequence(new_seq, instance, D)
                        if f and d < current_dist - 1e-9:
                            current_seq = new_seq
                            current_dist = d
                            improved = True
            routes.append(current_seq)
        else:
            # If we can't form a feasible route for this cluster, 
            # we must handle it. 
            # Strategy: Add customers one by one to existing feasible routes or create singletons.
            # This is a fallback.
            for c in cluster:
                routes.append([c])
    
    return routes
# EVOLVE-BLOCK-END
