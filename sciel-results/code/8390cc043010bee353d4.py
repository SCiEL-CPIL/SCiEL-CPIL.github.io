"""EV Routing: Cluster-First Cheapest Insertion with 2-Opt/Or-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

DEADLINE = 110.0  # Safety margin before 120s limit

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _validate_route(route, inst):
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    cap = inst["capacity"]
    batt_max = inst["battery"]
    speed = inst["speed"]
    r_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    stations = set(inst["stations"])
    
    # Check capacity
    total_demand = sum(demands[i] for i in route)
    if total_demand > cap:
        return False
        
    # Check time and battery
    t = 0.0
    batt = batt_max
    prev = 0  # depot
    
    for i in route:
        d = _dist(coords[prev], coords[i])
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return False
        
        # If station, recharge
        if i in stations:
            wait = (batt_max - batt) * r_rate
            t += wait
            batt = batt_max
        else:
            # Customer
            if t < ready[i]:
                t = ready[i]
            if t > due[i] + 1e-9:
                return False
            t += service[i]
        
        prev = i
    
    # Return to depot
    d = _dist(coords[prev], coords[0])
    t += d / speed
    batt -= d
    if batt < -1e-9:
        return False
    if t > horizon + 1e-9:
        return False
        
    return True

def _route_distance(route, inst):
    coords = inst["coords"]
    if not route:
        return 0.0
    d = _dist(coords[0], coords[route[0]])
    for i in range(len(route) - 1):
        d += _dist(coords[route[i]], coords[route[i+1]])
    d += _dist(coords[route[-1]], coords[0])
    return d

def _insert_customer(route, cust, inst):
    """Try to insert customer into route. Returns (new_route, cost) or (None, inf) if infeasible."""
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    cap = inst["capacity"]
    batt_max = inst["battery"]
    speed = inst["speed"]
    r_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    stations = set(inst["stations"])
    
    current_demand = sum(demands[i] for i in route)
    if current_demand + demands[cust] > cap:
        return None, float('inf')
    
    best_cost = float('inf')
    best_route = None
    
    # Try inserting at every position (including before first, between, after last)
    # We simulate the route with the customer inserted at position k
    # To be fast, we only check validity of the whole route if insertion is promising?
    # No, battery/time windows make it hard to pre-filter. We must validate.
    # However, we can skip validation if we just added a customer and the previous route was valid?
    # No, inserting in the middle changes travel times for subsequent nodes.
    
    n = len(route)
    for k in range(n + 1):
        new_route = route[:k] + [cust] + route[k:]
        if _validate_route(new_route, inst):
            c = _route_distance(new_route, inst)
            if c < best_cost:
                best_cost = c
                best_route = new_route
                
    return best_route, best_cost

def _two_opt(route, inst):
    """Improve route using 2-opt."""
    coords = inst["coords"]
    improved = True
    while improved:
        improved = False
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i+1 to j
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                if _validate_route(new_route, inst):
                    old_d = _route_distance(route, inst)
                    new_d = _route_distance(new_route, inst)
                    if new_d < old_d - 1e-6:
                        route = new_route
                        improved = True
    return route

def plan(instance):
    start_time = time.time()
    random.seed(42)
    
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    cap = instance["capacity"]
    batt_max = instance["battery"]
    speed = instance["speed"]
    r_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]
    
    customers = list(range(1, n_cust + 1))
    
    # Pre-compute distances from depot
    # Sort customers by due time to prioritize tight windows
    customers.sort(key=lambda x: due[x])
    
    routes = []
    unassigned = set(customers)
    
    # Cheapest Insertion Heuristic
    # We maintain a list of current routes.
    # At each step, pick the customer with the highest "penalty" (e.g., tightest window or furthest from existing routes)
    # Or simply iterate through unassigned customers and insert into the route that minimizes distance increase.
    
    # To ensure we don't get stuck with infeasible routes, we start with empty routes.
    # Actually, standard Cheapest Insertion starts with one customer per route or builds up.
    # Given 25 customers, we can try to pack them.
    
    # Strategy:
    # 1. Start with no routes.
    # 2. While unassigned:
    #    - For each unassigned customer, find the best (route, position) to insert.
    #      - If no route can accept it, create a new route [customer].
    #    - Pick the customer with the minimum insertion cost (or maximum penalty).
    #    - Insert it.
    
    # Let's use a simpler greedy:
    # Sort customers by due time.
    # For each customer, try to insert into existing routes (cheapest insertion).
    # If cannot insert into any, start new route.
    
    # This might not be optimal but is fast.
    
    # Better:
    # 1. Cluster customers by proximity? No, time windows matter.
    # 2. Just use Cheapest Insertion on the sorted list.
    
    # Let's refine:
    # We will maintain `routes` as a list of lists.
    # We iterate through customers in order of increasing due time.
    # For each customer, we find the feasible insertion with minimum distance increase.
    # If none, start a new route.
    
    # This is O(N^3) roughly, which is fine for N=25.
    
    # To handle battery/stations, we must validate.
    
    # Optimization: Pre-compute distances between all nodes?
    # N_total = 25 + 4 + 1 = 30. 30x30 matrix is tiny.
    dist_mat = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            dist_mat[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    
    # Use dist_mat for speed in validation?
    # Validation logic is complex due to time/battery.
    # But distance calculation is just sum of dist_mat entries.
    
    def _validate_route_fast(route):
        # Same logic but using dist_mat
        total_demand = sum(demands[i] for i in route)
        if total_demand > cap:
            return False
            
        t = 0.0
        batt = batt_max
        prev = 0
        
        for i in route:
            d = dist_mat[prev, i]
            t += d / speed
            batt -= d
            if batt < -1e-9:
                return False
                
            if i in stations:
                wait = (batt_max - batt) * r_rate
                t += wait
                batt = batt_max
            else:
                if t < ready[i]:
                    t = ready[i]
                if t > due[i] + 1e-9:
                    return False
                t += service[i]
                
            prev = i
            
        d = dist_mat[prev, 0]
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return False
        if t > horizon + 1e-9:
            return False
        return True

    def _route_dist_fast(route):
        if not route:
            return 0.0
        d = dist_mat[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_mat[route[i], route[i+1]]
        d += dist_mat[route[-1], 0]
        return d

    # Re-insertion logic using fast functions
    def _insert_customer_fast(route, cust):
        current_demand = sum(demands[i] for i in route)
        if current_demand + demands[cust] > cap:
            return None, float('inf')
            
        best_cost = float('inf')
        best_route = None
        n = len(route)
        for k in range(n + 1):
            new_route = route[:k] + [cust] + route[k:]
            if _validate_route_fast(new_route):
                c = _route_dist_fast(new_route)
                if c < best_cost:
                    best_cost = c
                    best_route = new_route
        return best_route, best_cost

    # Main loop
    for cust in customers:
        if time.time() - start_time > DEADLINE:
            break
            
        best_global_cost = float('inf')
        best_global_route_idx = -1
        best_global_new_route = None
        
        # Try inserting into existing routes
        for idx, r in enumerate(routes):
            new_r, cost = _insert_customer_fast(r, cust)
            if new_r is not None and cost < best_global_cost:
                best_global_cost = cost
                best_global_route_idx = idx
                best_global_new_route = new_r
                
        # Check if new route is better
        new_route_single = [cust]
        if _validate_route_fast(new_route_single):
            cost_single = _route_dist_fast(new_route_single)
            # Compare with best existing
            # If no existing route could take it, or single is cheaper
            if best_global_route_idx == -1 or cost_single < best_global_cost:
                best_global_cost = cost_single
                best_global_route_idx = -1
                best_global_new_route = new_route_single
        
        if best_global_route_idx == -1:
            routes.append(best_global_new_route)
        else:
            routes[best_global_route_idx] = best_global_new_route
            
    # Local Search: 2-Opt on each route
    if time.time() - start_time < DEADLINE - 10:
        for idx in range(len(routes)):
            if time.time() - start_time > DEADLINE:
                break
            routes[idx] = _two_opt(routes[idx], instance)
            
    return routes

# EVOLVE-BLOCK-END
