"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _dist(i, j, coords):
    return np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)

def _route_distance(route, coords):
    if not route:
        return 0.0
    total = _dist(0, route[0], coords)
    for k in range(len(route) - 1):
        total += _dist(route[k], route[k+1], coords)
    total += _dist(route[-1], 0, coords)
    return total

def _check_route_feasibility(route, instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False, 0, 0
    
    battery = battery_max
    time = 0.0
    prev = 0
    for v in route:
        d = _dist(prev, v, coords)
        battery -= d
        if battery < -1e-9:
            return False, 0, 0
        time += d / speed
        if v not in stations:
            time = max(time, ready[v])
            if time > due[v] + 1e-9:
                return False, 0, 0
            time += service[v]
        else:
            energy = battery_max - battery
            time += energy * recharge_rate
            battery = battery_max
        prev = v
    
    d = _dist(prev, 0, coords)
    battery -= d
    if battery < -1e-9:
        return False, 0, 0
    time += d / speed
    if time > horizon + 1e-9:
        return False, 0, 0
    
    return True, battery, time

def _insert_customer(route, v, instance, coords):
    """Find best position to insert v into route, return (new_route, cost_increase) or None."""
    best_pos = None
    best_cost = float('inf')
    base_dist = _route_distance(route, coords)
    
    for pos in range(len(route) + 1):
        new_route = route[:pos] + [v] + route[pos:]
        feasible, _, _ = _check_route_feasibility(new_route, instance)
        if feasible:
            new_dist = _route_distance(new_route, coords)
            cost = new_dist
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                
    if best_pos is not None:
        return route[:best_pos] + [v] + route[best_pos:], best_cost
    return None, float('inf')

def _construct_initial(instance):
    coords = np.array(instance["coords"], dtype=float)
    customers = list(range(1, instance["n_customers"] + 1))
    random.shuffle(customers)
    
    routes = []
    unserved = set(customers)
    
    while unserved:
        # Start new route with first available customer
        v = customers[0]
        if v not in unserved:
            continue
        route = [v]
        unserved.remove(v)
        
        # Try to add other customers
        changed = True
        while changed:
            changed = False
            best_v = None
            best_pos = None
            best_cost = float('inf')
            
            for u in list(unserved):
                # Try inserting u into route
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [u] + route[pos:]
                    feasible, _, _ = _check_route_feasibility(new_route, instance)
                    if feasible:
                        new_dist = _route_distance(new_route, coords)
                        if new_dist < best_cost:
                            best_cost = new_dist
                            best_v = u
                            best_pos = pos
            
            if best_v is not None:
                route = route[:best_pos] + [best_v] + route[best_pos:]
                unserved.remove(best_v)
                changed = True
        
        routes.append(route)
        
    return routes

def _two_opt(route, instance, coords):
    improved = True
    iters = 0
    max_iter = 20
    while improved and iters < max_iter:
        improved = False
        iters += 1
        n = len(route)
        if n < 2:
            break
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if new_route == route:
                    continue
                feasible, _, _ = _check_route_feasibility(new_route, instance)
                if feasible:
                    if _route_distance(new_route, coords) < _route_distance(route, coords) - 1e-9:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    return route

def _or_opt(route, instance, coords):
    improved = True
    iters = 0
    max_iter = 20
    while improved and iters < max_iter:
        improved = False
        iters += 1
        n = len(route)
        if n < 2:
            break
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                seg = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + seg + rest[j:]
                    if new_route == route:
                        continue
                    feasible, _, _ = _check_route_feasibility(new_route, instance)
                    if feasible:
                        if _route_distance(new_route, coords) < _route_distance(route, coords) - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    random.seed(42)
    
    # Initial solution
    routes = _construct_initial(instance)
    
    # Local search
    for i in range(len(routes)):
        routes[i] = _two_opt(routes[i], instance, coords)
        routes[i] = _or_opt(routes[i], instance, coords)
        
    current_routes = [r[:] for r in routes]
    current_cost = sum(_route_distance(r, coords) for r in current_routes)
    
    # ALNS
    T = 100.0
    cooling = 0.99
    max_iter = 500
    destroy_ratio = 0.3
    
    for it in range(max_iter):
        # Destroy
        k = max(1, int(len([v for r in current_routes for v in r]) * destroy_ratio))
        all_customers = [v for r in current_routes for v in r]
        random.shuffle(all_customers)
        to_remove = all_customers[:k]
        
        new_routes = []
        for r in current_routes:
            new_r = [v for v in r if v not in to_remove]
            if new_r:
                new_routes.append(new_r)
        
        # Repair
        for v in to_remove:
            best_route_idx = -1
            best_cost = float('inf')
            best_new_route = None
            
            for i, r in enumerate(new_routes):
                new_r, cost = _insert_customer(r, v, instance, coords)
                if new_r is not None:
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = i
                        best_new_route = new_r
            
            if best_route_idx >= 0:
                new_routes[best_route_idx] = best_new_route
            else:
                # Create new route
                new_routes.append([v])
        
        # Local search on changed routes (simplified: all)
        for i in range(len(new_routes)):
            new_routes[i] = _two_opt(new_routes[i], instance, coords)
            # new_routes[i] = _or_opt(new_routes[i], instance, coords) # Maybe too slow for every iter
            
        new_cost = sum(_route_distance(r, coords) for r in new_routes)
        
        # Accept
        if new_cost < current_cost:
            current_routes = new_routes
            current_cost = new_cost
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_routes = new_routes
                current_cost = new_cost
                
        T *= cooling
        if T < 1e-6:
            break
            
    # Final local search
    for i in range(len(current_routes)):
        current_routes[i] = _two_opt(current_routes[i], instance, coords)
        current_routes[i] = _or_opt(current_routes[i], instance, coords)
        
    # Verify
    all_nodes = []
    for r in current_routes:
        all_nodes.extend(r)
    customer_nodes = [v for v in all_nodes if v not in set(instance["stations"])]
    if sorted(customer_nodes) != list(range(1, instance["n_customers"] + 1)):
        # Fallback
        routes = _construct_initial(instance)
        for i in range(len(routes)):
            routes[i] = _two_opt(routes[i], instance, coords)
        return routes
        
    return current_routes
# EVOLVE-BLOCK-END
