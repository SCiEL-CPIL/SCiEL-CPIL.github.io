"""Rewritten program with stronger algorithm for EV routing with time windows."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _evaluate_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """
    Evaluate a single route.
    Returns (distance, penalty, is_feasible).
    """
    if not route:
        return 0.0, 0.0, True
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    penalty = 0.0
    is_feasible = True
    
    if total_demand > capacity:
        is_feasible = False
        penalty += (total_demand - capacity) * 1000.0
        
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            is_feasible = False
            penalty += abs(battery) * 1000.0
            
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v]:
                is_feasible = False
                penalty += (time_t - due[v]) * 100.0
            time_t += service[v]
        else:
            # Charging station
            if battery < battery_max:
                energy_needed = battery_max - battery
                time_t += energy_needed * recharge_rate
                battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        is_feasible = False
        penalty += abs(battery) * 1000.0
    if time_t > horizon:
        is_feasible = False
        penalty += (time_t - horizon) * 100.0
        
    return total_dist, penalty, is_feasible


def _evaluate_solution(routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """
    Evaluate the entire solution (list of routes).
    Returns (total_distance, total_penalty, is_feasible).
    """
    total_dist = 0.0
    total_penalty = 0.0
    is_feasible = True
    
    for route in routes:
        d, p, f = _evaluate_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        total_dist += d
        total_penalty += p
        if not f:
            is_feasible = False
            
    return total_dist, total_penalty, is_feasible


def _nearest_neighbor_route(start, unvisited, dist, ready, due, service, battery_max, battery, time_t, speed, recharge_rate, instance):
    """
    Build a route using nearest neighbor heuristic from a starting customer.
    Returns (route, final_battery, final_time, total_dist).
    """
    if not unvisited:
        return [], battery, time_t, 0.0
    
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    
    route = []
    current = start
    current_battery = battery
    current_time = time_t
    total_dist = 0.0
    remaining = list(unvisited)
    
    while remaining:
        best_next = None
        best_dist = float('inf')
        
        for c in remaining:
            d = dist[current][c]
            if d < best_dist:
                best_dist = d
                best_next = c
        
        if best_next is None:
            break
        
        # Check if we can go to best_next
        d = dist[current][best_next]
        new_battery = current_battery - d
        new_time = current_time + d / speed
        
        if new_battery < 0:
            # Need to charge first - find nearest station
            # For simplicity, skip this customer for now
            continue
        
        if best_next <= n_customers:
            if new_time < ready[best_next]:
                new_time = ready[best_next]
            if new_time > due[best_next]:
                # Time window violation, skip
                continue
            new_time += service[best_next]
        
        # Accept this customer
        route.append(best_next)
        total_dist += d
        current_battery = new_battery
        current_time = new_time
        current = best_next
        remaining.remove(best_next)
        
        # If battery is low, go to nearest station
        if current_battery < battery_max * 0.3:
            # Find nearest station
            best_station = None
            best_sd = float('inf')
            for s in stations:
                sd = dist[current][s]
                if sd < best_sd:
                    best_sd = sd
                    best_station = s
            
            if best_station is not None:
                d_s = dist[current][best_station]
                if current_battery - d_s >= 0:
                    new_battery = current_battery - d_s
                    new_time = current_time + d_s / speed
                    energy_needed = battery_max - new_battery
                    new_time += energy_needed * recharge_rate
                    new_battery = battery_max
                    total_dist += d_s
                    current_battery = new_battery
                    current_time = new_time
                    current = best_station
                    # Don't add station to route (it's implicit in the path)
                    # Actually, we need to track it for distance calculation
                    # For now, just update state
        
    return route, current_battery, current_time, total_dist


def _construct_savings_routes(instance, dist):
    """
    Construct initial routes using Clarke-Wright savings heuristic.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    
    customers = list(range(1, n_customers + 1))
    
    # Initial: each customer in its own route
    routes = [[c] for c in customers]
    route_demand = {c: demands[c] for c in customers}
    route_dist = {c: dist[0][c] + dist[c][0] for c in customers}
    
    # Calculate savings
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s = dist[0][c1] + dist[0][c2] - dist[c1][c2]
            savings.append((s, c1, c2))
    
    savings.sort(reverse=True)
    
    # Union-Find for tracking which route each customer belongs to
    parent = {c: c for c in customers}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Apply savings
    for s, c1, c2 in savings:
        if s <= 0:
            continue
        
        r1, r2 = find(c1), find(c2)
        if r1 == r2:
            continue
        
        # Check capacity
        if route_demand[r1] + route_demand[r2] > capacity:
            continue
        
        # Merge routes
        # Find the routes containing c1 and c2
        route1 = None
        route2 = None
        for route in routes:
            if c1 in route:
                route1 = route
            elif c2 in route:
                route2 = route
        
        if route1 is None or route2 is None:
            continue
        
        # Merge: append one to the other
        # Choose order to minimize distance
        d12 = dist[route1[-1]][route2[0]]
        d21 = dist[route2[-1]][route1[0]]
        
        if d12 <= d21:
            new_route = route1 + route2
        else:
            new_route = route2 + route1
        
        # Update
        routes.remove(route1)
        routes.remove(route2)
        routes.append(new_route)
        
        # Update data structures
        new_r = find(new_route[0])
        route_demand[new_r] = route_demand[r1] + route_demand[r2]
        route_dist[new_r] = route_dist[r1] + route_dist[r2] - s
        
        for c in new_route:
            parent[c] = new_r
        
        union(r1, r2)
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes


def _solve_instance(instance, time_limit=45):
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    
    # Initial Solution using savings heuristic
    routes = _construct_savings_routes(instance, dist)
    
    # Ensure all customers are covered
    covered = set()
    for r in routes:
        covered.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in covered]
    for c in missing:
        routes.append([c])
    
    # Evaluate initial solution
    init_dist, init_pen, init_feas = _evaluate_solution(routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
    
    # Parallel Tempering Setup
    n_replicas = 4
    temperatures = [200.0, 100.0, 50.0, 10.0]
    replicas = [list(routes) for _ in range(n_replicas)]
    replica_scores = []
    replica_penalties = []
    replica_feasible = []
    
    for i in range(n_replicas):
        d, p, f = _evaluate_solution(replicas[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        score = d + p
        replica_scores.append(score)
        replica_penalties.append(p)
        replica_feasible.append(f)
        
    best_routes = None
    best_score = float('inf')
    best_feasible_score = float('inf')
    
    # Find initial best feasible
    for i in range(n_replicas):
        if replica_feasible[i]:
            if replica_scores[i] < best_feasible_score:
                best_feasible_score = replica_scores[i]
                best_routes = [r[:] for r in replicas[i]]
                
    if best_routes is None:
        # If no feasible solution found, pick the one with lowest penalty
        min_pen_idx = min(range(n_replicas), key=lambda i: replica_penalties[i])
        best_routes = [r[:] for r in replicas[min_pen_idx]]
        best_feasible_score = replica_scores[min_pen_idx]

    # SA Loop
    step = 0
    while time.time() < deadline - 2:
        # Pick a random replica to move
        r_idx = random.randint(0, n_replicas - 1)
        old_routes = replicas[r_idx]
        old_score = replica_scores[r_idx]
        
        # Generate a move
        move_type = random.random()
        
        new_routes = [r[:] for r in old_routes]
        
        # Helper to find route index of a customer
        def get_route_index(routes, c):
            for idx, r in enumerate(routes):
                if c in r:
                    return idx
            return -1
            
        # Helper to get position of customer in route
        def get_pos(route, c):
            return route.index(c)
            
        if move_type < 0.35:
            # Relocate: Move a customer from one route to another
            all_customers = []
            for r in new_routes:
                all_customers.extend(r)
            
            if not all_customers:
                continue
                
            c = random.choice(all_customers)
            src_r_idx = get_route_index(new_routes, c)
            if src_r_idx == -1:
                continue
                
            src_route = new_routes[src_r_idx]
            pos_c = get_pos(src_route, c)
            
            # Remove from source
            src_route.pop(pos_c)
            
            # Pick destination route
            dst_r_idx = random.randint(0, len(new_routes) - 1)
            if dst_r_idx == src_r_idx:
                # Move within same route
                if len(src_route) < 1:
                    src_route.insert(pos_c, c)
                    continue
                dst_route = src_route
                ins_pos = random.randint(0, len(dst_route))
                dst_route.insert(ins_pos, c)
            else:
                dst_route = new_routes[dst_r_idx]
                ins_pos = random.randint(0, len(dst_route))
                dst_route.insert(ins_pos, c)
                
            # Remove empty routes
            new_routes = [r for r in new_routes if r]
            
        elif move_type < 0.65:
            # Swap: Exchange two customers from different routes
            if len(new_routes) < 2:
                continue
            r1, r2 = random.sample(range(len(new_routes)), 2)
            route1 = new_routes[r1]
            route2 = new_routes[r2]
            
            if not route1 or not route2:
                continue
                
            c1 = random.choice(route1)
            c2 = random.choice(route2)
            
            pos1 = route1.index(c1)
            pos2 = route2.index(c2)
            
            route1[pos1] = c2
            route2[pos2] = c1
            
        elif move_type < 0.85:
            # 2-Opt within a route
            if not new_routes:
                continue
            r_idx_move = random.randint(0, len(new_routes) - 1)
            route = new_routes[r_idx_move]
            if len(route) < 2:
                continue
            
            i = random.randint(0, len(route) - 2)
            j = random.randint(i + 1, len(route) - 1)
            
            route[i:j+1] = route[i:j+1][::-1]
            
        else:
            # Or-opt: Move a segment of 1-3 customers
            if not new_routes:
                continue
            r_idx_move = random.randint(0, len(new_routes) - 1)
            route = new_routes[r_idx_move]
            if len(route) < 3:
                continue
            
            seg_len = random.choice([1, 2, 3])
            if len(route) < seg_len:
                seg_len = len(route)
            
            i = random.randint(0, len(route) - seg_len)
            segment = route[i:i+seg_len]
            
            # Remove segment
            del route[i:i+seg_len]
            
            # Insert at random position
            if not route:
                route.extend(segment)
            else:
                ins_pos = random.randint(0, len(route))
                for k, c in enumerate(segment):
                    route.insert(ins_pos + k, c)
                
        # Evaluate new solution
        new_dist, new_pen, new_feas = _evaluate_solution(new_routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        new_score = new_dist + new_pen
        
        # Acceptance Criteria
        delta = new_score - old_score
        T = temperatures[r_idx]
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            replicas[r_idx] = new_routes
            replica_scores[r_idx] = new_score
            replica_penalties[r_idx] = new_pen
            replica_feasible[r_idx] = new_feas
            
            # Update best feasible
            if new_feas:
                if new_score < best_feasible_score:
                    best_feasible_score = new_score
                    best_routes = [r[:] for r in new_routes]
            
            # Parallel Tempering Swap
            if r_idx > 0:
                swap_idx = r_idx - 1
                if random.random() < 0.15:
                    delta_swap = replica_scores[r_idx] - replica_scores[swap_idx]
                    if delta_swap < 0:
                        # Always accept if hotter has better score
                        replicas[r_idx], replicas[swap_idx] = replicas[swap_idx], replicas[r_idx]
                        replica_scores[r_idx], replica_scores[swap_idx] = replica_scores[swap_idx], replica_scores[r_idx]
                        replica_penalties[r_idx], replica_penalties[swap_idx] = replica_penalties[swap_idx], replica_penalties[r_idx]
                        replica_feasible[r_idx], replica_feasible[swap_idx] = replica_feasible[swap_idx], replica_feasible[r_idx]
                    else:
                        prob = math.exp(-delta_swap / temperatures[r_idx])
                        if random.random() < prob:
                            replicas[r_idx], replicas[swap_idx] = replicas[swap_idx], replicas[r_idx]
                            replica_scores[r_idx], replica_scores[swap_idx] = replica_scores[swap_idx], replica_scores[r_idx]
                            replica_penalties[r_idx], replica_penalties[swap_idx] = replica_penalties[swap_idx], replica_penalties[r_idx]
                            replica_feasible[r_idx], replica_feasible[swap_idx] = replica_feasible[swap_idx], replica_feasible[r_idx]
            
        step += 1
        
    return best_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> ~45s per instance
    per_instance_time = 45
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
