import math
import time
import random
import numpy as np
from scipy.spatial.distance import cdist

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    # Safety margin: The evaluator kills at 150s. We use 145s to be safe.
    DEADLINE = 145.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix
    dist = cdist(coords, coords)

    customers = list(range(1, n_cust + 1))

    # Helper to check route validity and get cost
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, 0.0
            
            # Time window check
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, 0.0
            t += service[node]
            
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, 0.0
                
            d_total += d
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, 0.0
        if t > horizon + _EPS:
            return False, 0.0, 0.0
        d_total += d
        return True, d_total, t

    def evaluate_routes(routes):
        total_d = 0.0
        for r in routes:
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Strategy: Cluster-First, Route-Second ---
    
    # 1. Determine number of clusters (K)
    # Heuristic: Estimate average distance and capacity constraints.
    # Total demand / capacity gives a lower bound on number of routes.
    total_demand = sum(demands[c] for c in customers)
    min_routes_by_capacity = max(1, math.ceil(total_demand / capacity))
    
    # Also consider spatial spread.
    # Let's try K values from min_routes to min_routes + 2
    # We will evaluate multiple K values and pick the best.
    
    best_overall_routes = None
    best_overall_dist = float('inf')
    
    # We have 145s total. Let's allocate time to try different K values.
    # K is likely small (2-5 routes for 25 customers with cap 100, avg demand ~15 -> 25*15/100 = 3.75 -> 4 routes).
    
    possible_ks = []
    for k in range(min_routes_by_capacity, min_routes_by_capacity + 3):
        if k <= n_cust:
            possible_ks.append(k)
            
    if not possible_ks:
        possible_ks = [1, 2, 3]

    # Sort customers by coordinates for K-Means initialization or just use random init
    cust_coords = coords[1:n_cust+1]
    
    rng = np.random.default_rng(42)
    
    # Time allocation per K
    time_per_k = (DEADLINE - 5.0) / len(possible_ks)
    
    for K in possible_ks:
        if time.time() > DEADLINE - 5.0:
            break
            
        # 2. Cluster customers using K-Means
        # Simple K-Means implementation
        # Init: Pick K random customers as centroids
        centroids = cust_coords[rng.choice(n_cust, K, replace=False)]
        
        labels = np.zeros(n_cust, dtype=int)
        
        for _ in range(20): # Max iterations
            # Assign
            distances = cdist(cust_coords, centroids)
            new_labels = np.argmin(distances, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Update
            for k in range(K):
                mask = labels == k
                if np.sum(mask) > 0:
                    centroids[k] = cust_coords[mask].mean(axis=0)
                else:
                    # If a cluster is empty, reassign a random point
                    centroids[k] = cust_coords[rng.integers(0, n_cust)]
                    
        # Form clusters
        clusters = [[] for _ in range(K)]
        for i, label in enumerate(labels):
            clusters[label].append(i + 1) # i is 0-indexed in cust_coords, so +1 is node id
            
        # Filter empty clusters (should not happen with K <= N and good init, but just in case)
        clusters = [c for c in clusters if len(c) > 0]
        
        # 3. Solve each cluster independently
        # For each cluster, we need to find a valid tour.
        # Since clusters are small (25/K ~ 5-12 nodes), we can use a robust local search.
        
        cluster_routes = []
        valid_cluster_set = True
        
        # We need to solve them such that they are valid.
        # If a cluster is too large or infeasible, we might need to split it or accept it's invalid.
        # Given the problem structure, clusters from K-Means are spatially compact.
        # We will use a local search to optimize each cluster.
        
        # Allocate time for local search
        # Total time left for this K: time_per_k
        # We can distribute time equally or by cluster size.
        
        for cluster in clusters:
            if time.time() > DEADLINE - 5.0:
                valid_cluster_set = False
                break
                
            # Initial solution for cluster: Sort by angle (sweep) or nearest neighbor
            # Let's use a simple nearest neighbor construction for the cluster
            # Start at depot
            unvisited = set(cluster)
            route = []
            current = 0
            t = 0.0
            b = battery_cap
            load = 0.0
            
            while unvisited:
                # Find nearest valid customer
                best_next = -1
                best_d = float('inf')
                for c in unvisited:
                    d = dist[current, c]
                    # Quick feasibility check for time/battery
                    # This is an approximation, full check is in check_route
                    if d < best_d:
                        # Check if battery allows going there and back (roughly)
                        # b - d >= 0
                        if b - d >= 0:
                            best_d = d
                            best_next = c
                            
                if best_next == -1:
                    # No valid next step found in this greedy pass.
                    # Try to find any valid step with higher distance
                    for c in unvisited:
                        d = dist[current, c]
                        if b - d >= 0:
                            best_next = c
                            break
                            
                if best_next == -1:
                    # Infeasible cluster configuration with this greedy start
                    # We might need to handle this by splitting or re-clustering.
                    # For now, mark as invalid and break.
                    valid_cluster_set = False
                    break
                
                route.append(best_next)
                unvisited.remove(best_next)
                current = best_next
                # Update t, b, load roughly for greedy? No, just construct.
                # The local search will fix the order.
                
            if not valid_cluster_set:
                break
                
            # Now optimize 'route' using local search
            # We have a time budget for this cluster
            # Let's say we have 2 seconds per cluster on average?
            # Better: calculate remaining time and distribute.
            
            # For simplicity, run a fixed number of iterations or time-based loop
            # Let's use a time-based loop with a cap.
            
            # Local Search: 2-Opt and Relocate
            # We need to ensure validity.
            
            # Initial check
            is_valid, curr_d, _ = check_route(route)
            if not is_valid:
                # If greedy failed, try reversing or shuffling?
                # Or just accept it might be invalid and this K fails.
                valid_cluster_set = False
                break
                
            # Local Search
            # Time limit for this cluster: (DEADLINE - current_time - 5) / remaining_clusters
            # Let's just use a fixed small time or iteration count to avoid overhead.
            # Given small N, 1000 iterations of 2-opt is fast.
            
            # We'll use a simple loop
            improved = True
            while improved and time.time() < DEADLINE - 5.0:
                improved = False
                # 2-Opt
                n = len(route)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        # Try reversing segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_d, _ = check_route(new_route)
                        if is_valid and new_d < curr_d - _EPS:
                            route = new_route
                            curr_d = new_d
                            improved = True
                            # Restart 2-opt after improvement (standard)
                            break
                    if improved:
                        break
                
                if not improved:
                    # Try Relocate
                    for i in range(len(route)):
                        c = route[i]
                        rest = route[:i] + route[i+1:]
                        for j in range(len(rest) + 1):
                            new_route = rest[:j] + [c] + rest[j:]
                            is_valid, new_d, _ = check_route(new_route)
                            if is_valid and new_d < curr_d - _EPS:
                                route = new_route
                                curr_d = new_d
                                improved = True
                                break
                        if improved:
                            break
                            
            cluster_routes.append(route)

        if valid_cluster_set:
            # Check total validity
            is_valid, total_d, _ = evaluate_routes(cluster_routes)
            if is_valid:
                if total_d < best_overall_dist:
                    best_overall_dist = total_d
                    best_overall_routes = [list(r) for r in cluster_routes]

    # Fallback if no valid solution found
    if best_overall_routes is None:
        # One customer per route
        best_overall_routes = [[c] for c in customers]
        # Verify
        is_valid, _, _ = evaluate_routes(best_overall_routes)
        if not is_valid:
            # This should not happen, but if it does, return empty? No, must return list.
            # If even single customers are invalid, the instance is infeasible.
            # But problem implies feasible.
            pass

    return best_overall_routes
