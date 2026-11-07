# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def plan(instance):
    start_time = time.time()
    DEADLINE = 115.0
    
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
    
    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    
    def is_feasible(route):
        """Check if a route is feasible. Returns (valid, total_distance)."""
        if not route:
            return True, 0.0
            
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        total_demand = 0.0
        
        for node in route:
            d = dist[curr_node][node]
            total_dist += d
            
            if curr_battery < d:
                return False, 0.0
            
            curr_battery -= d
            curr_time += d / speed
            
            if node in stations:
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
            else:
                total_demand += demands[node]
                if total_demand > capacity:
                    return False, 0.0
                
                arrival_time = curr_time
                service_start = max(arrival_time, ready[node])
                if service_start > due[node]:
                    return False, 0.0
                
                curr_time = service_start + service[node]
                
            curr_node = node
            
        d = dist[curr_node][0]
        total_dist += d
        
        if curr_battery < d:
            return False, 0.0
        
        curr_battery -= d
        curr_time += d / speed
        
        if curr_time > horizon:
            return False, 0.0
            
        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        curr = 0
        total = 0.0
        for node in route:
            total += dist[curr][node]
            curr = node
        total += dist[curr][0]
        return total

    def two_opt(route):
        """Improve route using 2-opt. Returns improved route."""
        if len(route) < 3:
            return route
            
        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return best_route
            
        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for i in range(n - 1):
                if time.time() - start_time > DEADLINE:
                    return best_route
                for j in range(i + 1, n):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    valid, new_dist = is_feasible(new_route)
                    if valid and new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def or_opt(route):
        """OR-opt: move 1, 2, or 3 consecutive nodes to another position."""
        if len(route) < 3:
            return route
            
        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return best_route
            
        n = len(best_route)
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                return best_route
            for seg_len in [1, 2, 3]:
                if n < seg_len + 1:
                    continue
                for i in range(n - seg_len + 1):
                    if time.time() - start_time > DEADLINE:
                        return best_route
                    segment = best_route[i:i+seg_len]
                    # Remove segment
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    # Try inserting at all positions in remaining
                    for j in range(len(remaining) + 1):
                        if j == i or j == i + 1:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        valid, new_dist = is_feasible(new_route)
                        if valid and new_dist < best_dist:
                            best_route = new_route
                            best_dist = new_dist
                            n = len(best_route)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    def relocate_between_routes(routes):
        """Try to relocate a customer from one route to another."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                return routes
            for i in range(len(routes)):
                if len(routes[i]) <= 1:
                    continue
                for pos in range(len(routes[i])):
                    if time.time() - start_time > DEADLINE:
                        return routes
                    c = routes[i][pos]
                    # Try removing from route i
                    route_i_without = routes[i][:pos] + routes[i][pos+1:]
                    # Try inserting into other routes
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        # Check capacity
                        demand_j = sum(demands[node] for node in routes[j])
                        if demand_j + demands[c] > capacity:
                            continue
                        for p in range(len(routes[j]) + 1):
                            new_route_j = routes[j][:p] + [c] + routes[j][p:]
                            valid_j, dist_j = is_feasible(new_route_j)
                            if not valid_j:
                                continue
                            valid_i, dist_i = is_feasible(route_i_without)
                            if not valid_i:
                                continue
                            new_total = dist_j + dist_i
                            old_total = route_distance(routes[i]) + route_distance(routes[j])
                            if new_total < old_total - 1e-9:
                                routes[i] = route_i_without
                                routes[j] = new_route_j
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # Remove empty routes
            routes[:] = [r for r in routes if r]
        return routes

    # Savings-based construction
    # Start with each customer as its own route
    routes = [[c] for c in customers]
    
    # Compute savings for each pair (i, j): savings = dist[0][i] + dist[0][j] - dist[i][j]
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, i, j))
    
    # Sort by savings descending
    savings.sort(key=lambda x: -x[0])
    
    # Union-Find for tracking which route each customer belongs to
    route_of_customer = {c: i for i, c in enumerate(customers)}
    route_list = list(range(len(customers)))  # Active route indices
    active_routes = set(range(len(customers)))
    
    # For each pair, try to merge their routes
    for s, i, j in savings:
        if time.time() - start_time > DEADLINE:
            break
        if s <= 0:
            continue
            
        ci = customers[i]
        cj = customers[j]
        
        ri = route_of_customer.get(ci)
        rj = route_of_customer.get(cj)
        
        if ri is None or rj is None:
            continue
        if ri == rj:
            continue
        if ri not in active_routes or rj not in active_routes:
            continue
            
        # Get the two routes
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Check combined capacity
        demand_i = sum(demands[node] for node in route_i)
        demand_j = sum(demands[node] for node in route_j)
        if demand_i + demand_j > capacity:
            continue
            
        # Try all ways to concatenate the two routes
        # route_i ends at last element, route_j ends at last element
        # Possible merges:
        # 1. route_i + route_j (i's end connects to j's start)
        # 2. route_i + reversed(route_j) (i's end connects to j's end)
        # 3. reversed(route_i) + route_j (i's start connects to j's start)
        # 4. reversed(route_i) + reversed(route_j) (i's start connects to j's end)
        
        candidates = []
        ri_rev = route_i[::-1]
        rj_rev = route_j[::-1]
        
        candidates.append(route_i + route_j)
        candidates.append(route_i + rj_rev)
        candidates.append(ri_rev + route_j)
        candidates.append(ri_rev + rj_rev)
        
        best_candidate = None
        best_dist = float('inf')
        
        for cand in candidates:
            valid, d = is_feasible(cand)
            if valid and d < best_dist:
                best_dist = d
                best_candidate = cand
        
        if best_candidate is not None:
            # Merge: keep route_i, remove route_j
            routes[ri] = best_candidate
            active_routes.discard(rj)
            for node in route_j:
                route_of_customer[node] = ri
            
    # Remove empty/inactive routes
    final_routes = [routes[i] for i in active_routes if routes[i]]
    
    # Local search: 2-opt + OR-opt on each route
    for r_idx in range(len(final_routes)):
        if time.time() - start_time > DEADLINE:
            break
        final_routes[r_idx] = two_opt(final_routes[r_idx])
        if time.time() - start_time > DEADLINE:
            break
        final_routes[r_idx] = or_opt(final_routes[r_idx])
    
    # Inter-route relocation
    final_routes = relocate_between_routes(final_routes)
    
    # Final 2-opt pass
    for r_idx in range(len(final_routes)):
        if time.time() - start_time > DEADLINE:
            break
        final_routes[r_idx] = two_opt(final_routes[r_idx])
    
    return final_routes
# EVOLVE-BLOCK-END
