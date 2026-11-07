# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import random

def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            v = math.hypot(xi - xj, yi - yj)
            d[i, j] = v
            d[j, i] = v
    return d

def _check_route(route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed=1.0):
    """Return (feasible, total_distance, final_time, final_battery)."""
    if not route:
        return True, 0.0, 0.0, battery_cap
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    depot = 0

    t = 0.0
    bat = battery_cap
    load = 0.0
    total_dist = 0.0
    prev = depot

    for node in route:
        d = dist[prev][node]
        bat -= d
        t += d / speed
        total_dist += d
        if bat < -1e-9:
            return False, total_dist, t, bat
        if node in stations:
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
        else:
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, t, bat
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        prev = node

    # return to depot
    d = dist[prev][depot]
    bat -= d
    t += d / speed
    total_dist += d
    if bat < -1e-9:
        return False, total_dist, t, bat
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    return True, total_dist, t, bat

def _savings_construction(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = 1.0

    dist = _build_dist(coords)
    customers = list(range(1, n_customers + 1))
    
    # Initial routes: each customer is a route
    routes = [[c] for c in customers]
    route_index = {c: i for i, c in enumerate(customers)}
    # route_info[i] = [start, end, load]
    route_info = [[c, c, demands[c]] for c in customers]
    
    # Calculate savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            if s > 0:
                savings.append((s, i, j))
    
    savings.sort(key=lambda x: -x[0])
    
    # Merge routes
    for s, i, j in savings:
        if route_info[i][0] == -1 or route_info[j][0] == -1:
            continue # One of the routes is already merged (deleted)
        
        # Check capacity
        if route_info[i][2] + route_info[j][2] > capacity:
            continue
            
        # Try to merge route i and route j
        # We can merge if:
        # 1. end of i == ci and start of j == cj -> i + j
        # 2. start of i == ci and end of j == cj -> j + i (reverse i? No, routes are sequences)
        # Wait, in Clarke-Wright, routes are sequences. 
        # Let's maintain the sequence explicitly or just head/tail?
        # For simplicity and correctness with time windows later, 
        # standard CW merges chains. 
        # Let's keep the list.
        
        route_i = routes[i]
        route_j = routes[j]
        
        # Check if they can be connected
        # Case 1: end of i is ci and start of j is cj
        # Case 2: end of j is cj and start of i is ci
        # Note: Since we started with single nodes, and only merge, 
        # the "free" ends are the head and tail of the current chain.
        
        # Let's define:
        # head[i] = first node in route i
        # tail[i] = last node in route i
        # We need to track these.
        # I will refactor to use explicit head/tail arrays for speed.
        
        # Actually, let's just use the list and check first/last.
        if route_i[-1] == ci and route_j[0] == cj:
             # Merge i + j
             # Check feasibility of new combined route roughly? 
             # CW doesn't check TWs during construction usually, 
             # but we can do a quick check or rely on repair.
             # Given the constraints, let's do a full check on the merged route.
             new_route = route_i + route_j
             # Quick check: just distance and capacity (already checked)
             # Time windows are handled by the local search/repair.
             # However, if we create a route that is obviously infeasible, we waste time.
             # Let's do a fast feasibility check (ignoring charging optimization for a moment? No, must be exact).
             # Let's just do the full check.
             feasible, _, _, _ = _check_route(new_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)
             if feasible:
                 routes[i] = new_route
                 route_info[i][1] = route_j[1] # tail becomes tail of j
                 route_info[i][2] += route_info[j][2]
                 route_info[j][0] = -1
                 route_info[j][1] = -1
                 route_info[j][2] = 0
                 # Update route_index? Not needed if we only use i,j indices from savings list.
                 # But we need to know which indices are active.
                 # The loop iterates over fixed pairs. If i or j is dead, we skip.
                 # This works.
        elif route_i[0] == ci and route_j[-1] == cj:
             # Merge j + i
             new_route = route_j + route_i
             feasible, _, _, _ = _check_route(new_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)
             if feasible:
                 routes[i] = new_route
                 route_info[i][0] = route_j[0] # head becomes head of j
                 route_info[i][2] += route_info[j][2]
                 route_info[j][0] = -1
                 route_info[j][1] = -1
                 route_info[j][2] = 0
        # Other combinations (end i == cj, etc) are not possible with this savings definition 
        # where s(i,j) implies connecting i and j directly. 
        # Wait, savings s(i,j) = d(0,i)+d(0,j)-d(i,j). 
        # This connects node i and node j.
        # If route i ends at i and route j starts at j, we connect i->j.
        # If route i starts at i and route j ends at j, we connect j->i.
        # This covers the cases above.
        
    # Clean up
    final_routes = [r for r in routes if r]
    return final_routes

def _fast_relocate_check(route, i, j, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed):
    """
    Check if moving node i to position j in route is feasible.
    This is a simplified check. For full accuracy, we might need to simulate.
    Given the small size (25 customers), full simulation is O(N).
    Relocate is O(N^2) moves * O(N) check = O(N^3) per pass.
    With N=25, 25^3 = 15625 ops per pass. Very fast.
    So we can just use _check_route.
    """
    if i == j:
        return False, 0.0
    new_route = route[:i] + route[i+1:]
    # Adjust j if i < j
    if i < j:
        j -= 1
    new_route.insert(j, route[i])
    return _check_route(new_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)

def _optimize_routes(routes, instance, time_limit):
    coords = instance["coords"]
    dist = _build_dist(coords)
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = 1.0
    
    # Simulated Annealing
    # State: list of routes
    # Move: Relocate a customer to a different position in the same or different route.
    # Cost: Total distance.
    
    current_routes = [r[:] for r in routes]
    current_cost = 0.0
    for r in current_routes:
        if r:
            _, d, _, _ = _check_route(r, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)
            current_cost += d
    
    best_routes = [r[:] for r in current_routes]
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.9995
    start_time = time.time()
    
    n_customers = instance["n_customers"]
    all_customers = list(range(1, n_customers + 1))
    
    while T > 0.01 and (time.time() - start_time) < time_limit:
        # Pick a random customer
        c = random.choice(all_customers)
        
        # Find current route index and position
        curr_r_idx = -1
        curr_pos = -1
        for idx, r in enumerate(current_routes):
            if c in r:
                curr_r_idx = idx
                curr_pos = r.index(c)
                break
        
        if curr_r_idx == -1:
            continue
            
        # Pick a random target route and position
        # To allow moving between routes, pick a random route from all active routes
        active_routes = [idx for idx, r in enumerate(current_routes) if r]
        if not active_routes:
            break
            
        target_r_idx = random.choice(active_routes)
        target_pos = random.randint(0, len(current_routes[target_r_idx]))
        
        # Perform move
        # 1. Remove c from current route
        old_route = current_routes[curr_r_idx]
        new_curr_route = old_route[:curr_pos] + old_route[curr_pos+1:]
        
        # 2. Insert c into target route
        old_target_route = current_routes[target_r_idx]
        # If moving within same route, adjust position if removing before inserting
        if curr_r_idx == target_r_idx:
            if curr_pos < target_pos:
                target_pos -= 1
            if target_pos == curr_pos:
                # No change
                continue
            new_target_route = old_target_route[:target_pos] + [c] + old_target_route[target_pos:]
        else:
            new_target_route = old_target_route[:target_pos] + [c] + old_target_route[target_pos:]
        
        # Check feasibility
        feasible_curr, d_curr, _, _ = _check_route(new_curr_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)
        feasible_tgt, d_tgt, _, _ = _check_route(new_target_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)
        
        if not (feasible_curr and feasible_tgt):
            continue
            
        delta = (d_curr + d_tgt) - (
            (_check_route(old_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)[1]) + 
            (_check_route(old_target_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)[1])
        )
        
        # Actually, calculating old costs every time is expensive.
        # Let's maintain current_cost.
        # old_cost_curr = _check_route(old_route...)[1]
        # old_cost_tgt = _check_route(old_target_route...)[1]
        # This is 4 full simulations per move.
        # With 25 customers, it's fast enough.
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_routes[curr_r_idx] = new_curr_route
            current_routes[target_r_idx] = new_target_route
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = [r[:] for r in current_routes]
                
        T *= cooling
        
    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # 1. Construction
    routes = _savings_construction(instance)
    
    # 2. Optimization
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5.0:
        routes = _optimize_routes(routes, instance, remaining_time - 5.0)
        
    # 3. Verification
    coords = instance["coords"]
    dist = _build_dist(coords)
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    all_ok = True
    for route in routes:
        feasible, _, _, _ = _check_route(route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed=1.0)
        if not feasible:
            all_ok = False
            break
            
    if not all_ok:
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
        
    return routes
# EVOLVE-BLOCK-END
