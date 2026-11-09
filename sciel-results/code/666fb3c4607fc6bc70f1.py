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


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
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
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _construct_initial_routes(instance, dist, time_limit=5):
    """Construct initial routes using Nearest Neighbor + Insertion."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Sort customers by due date (earliest first) to help with time windows
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda c: due[c])
    
    routes = []
    remaining = set(customers)
    
    # We will build routes one by one.
    # Start with the first customer (earliest due date)
    while remaining:
        if time.time() > deadline:
            break
            
        # Start a new route with the first available customer
        start_cust = list(remaining)[0]
        remaining.remove(start_cust)
        
        current_route = [start_cust]
        current_demand = demands[start_cust]
        current_dist = dist[0][start_cust] + dist[start_cust][0] # Approximation, will be checked properly
        
        # Try to insert remaining customers into this route
        # We do this in a greedy manner: find the best insertion point for the next best customer
        # To keep it fast, we just try to insert customers in order of due date if they fit.
        
        changed = True
        while changed and remaining:
            changed = False
            # Try to find a customer that can be inserted
            # Iterate through remaining customers
            for c in list(remaining):
                if demands[c] + current_demand > capacity:
                    continue
                
                # Try all positions in current_route
                best_pos = -1
                best_cost = float('inf')
                
                # We need to check feasibility of the new route
                # To speed up, we can estimate cost first
                for pos in range(len(current_route) + 1):
                    new_route = current_route[:pos] + [c] + current_route[pos:]
                    feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        # Cost is increase in distance
                        if d < best_cost:
                            best_cost = d
                            best_pos = pos
                
                if best_pos != -1:
                    # Insert
                    current_route = current_route[:best_pos] + [c] + current_route[best_pos:]
                    current_demand += demands[c]
                    remaining.remove(c)
                    changed = True
                    break # Restart loop to try next customer
            
        routes.append(current_route)
        
        # If no more customers can be added, start a new route
        if not remaining:
            break
            
        # If the current route is "full" or no customer could be added, start a new one
        # The loop above breaks if no customer could be added in a full pass.
        # If `changed` is False, it means no customer could be added to the current route.
        # So we just proceed to the next iteration of the outer while loop.
        
    return routes


def _local_search(routes, dist, instance, time_limit=20):
    """Perform local search with Relocate, Swap, Or-Opt, and 2-Opt."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    # Precompute route distances and feasibility
    route_states = []
    for r in routes:
        feasible, d, t = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        route_states.append({
            'route': r,
            'feasible': feasible,
            'dist': d if feasible else float('inf'),
            'time': t
        })
    
    total_dist = sum(s['dist'] for s in route_states)
    
    # Helper to update route state
    def update_state(idx, new_route):
        feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        route_states[idx]['route'] = new_route
        route_states[idx]['feasible'] = feasible
        route_states[idx]['dist'] = d if feasible else float('inf')
        route_states[idx]['time'] = t
        return feasible, d, t
    
    # Remove empty routes
    active_indices = [i for i, s in enumerate(route_states) if s['route']]
    
    while time.time() < deadline - 1:
        improved = False
        
        # 1. Relocate: Move one customer from route i to route j
        for i in active_indices:
            if len(route_states[i]['route']) < 2:
                continue
            for pos in range(len(route_states[i]['route'])):
                c = route_states[i]['route'][pos]
                if c > n_customers: # Don't move stations in this simple relocate
                    continue
                
                # Remove from i
                new_route_i = route_states[i]['route'][:pos] + route_states[i]['route'][pos+1:]
                if not new_route_i:
                    continue # If empty, skip for now, handle separately if needed
                feas_i, d_i, t_i = _check_route(new_route_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if not feas_i:
                    continue
                
                # Try to insert into j
                for j in active_indices:
                    if i == j:
                        continue
                    rj = route_states[j]['route']
                    # Try all positions in j
                    for pos_j in range(len(rj) + 1):
                        new_route_j = rj[:pos_j] + [c] + rj[pos_j:]
                        feas_j, d_j, t_j = _check_route(new_route_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if not feas_j:
                            continue
                        
                        old_total = route_states[i]['dist'] + route_states[j]['dist']
                        new_total = d_i + d_j
                        
                        if new_total < old_total - 1e-9:
                            # Apply move
                            route_states[i]['route'] = new_route_i
                            route_states[i]['dist'] = d_i
                            route_states[j]['route'] = new_route_j
                            route_states[j]['dist'] = d_j
                            total_dist = new_total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            # Clean up empty routes
            active_indices = [i for i, s in enumerate(route_states) if s['route']]
            continue
        
        # 2. Or-Opt: Move 1-3 consecutive nodes within a route
        for i in active_indices:
            r = route_states[i]['route']
            n = len(r)
            if n < 2:
                continue
            
            for L in [1, 2, 3]:
                if n < L + 1:
                    continue
                for start in range(n - L):
                    moving = r[start:start+L]
                    # Check if moving contains stations? If so, it's complex. 
                    # For simplicity, only move customers or single stations.
                    # Actually, let's just try it.
                    rest = r[:start] + r[start+L:]
                    
                    # Try inserting at other positions
                    # Skip positions that are equivalent (before/after)
                    valid_positions = []
                    for pos in range(len(rest) + 1):
                        # Avoid inserting at the same place
                        if pos == start or (start > 0 and pos == start + L) or (start == 0 and pos == L):
                            # This logic is tricky. Let's just try all and check cost.
                            pass
                        
                        new_route = rest[:pos] + moving + rest[pos:]
                        if new_route == r:
                            continue
                        
                        feas, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if feas and d < route_states[i]['dist'] - 1e-9:
                            route_states[i]['route'] = new_route
                            route_states[i]['dist'] = d
                            total_dist = sum(s['dist'] for s in route_states)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 3. 2-Opt within a route
        for i in active_indices:
            r = route_states[i]['route']
            n = len(r)
            if n < 2:
                continue
            
            found_improve = False
            for a in range(n - 1):
                if found_improve:
                    break
                for b in range(a + 1, n):
                    # Reverse r[a:b+1]
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    if new_r == r:
                        continue
                    
                    feas, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feas and d < route_states[i]['dist'] - 1e-9:
                        route_states[i]['route'] = new_r
                        route_states[i]['dist'] = d
                        total_dist = sum(s['dist'] for s in route_states)
                        improved = True
                        found_improve = True
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # 4. 2-Opt between routes (Connect)
        # Try to connect route i and j by reversing a segment
        # This is expensive, so we limit attempts
        if len(active_indices) >= 2:
            # Pick random pair
            if random.random() < 0.5:
                i, j = random.sample(active_indices, 2)
                ri = route_states[i]['route']
                rj = route_states[j]['route']
                
                # Try reversing rj and connecting
                # New route i: ri + reversed(rj)
                # This is effectively a swap of the tail.
                # Let's try a simpler "Cross" move:
                # Take suffix of ri and suffix of rj and swap them?
                # Or just try to merge and split?
                
                # Let's try 2-Opt style:
                # Break edge (a, b) in i and (c, d) in j.
                # Connect a->c and b->d (with reversals if needed).
                
                # This is complex to implement efficiently without precomputation.
                # We'll skip this for now to keep the code robust and fast.
                pass

        if not improved:
            # If no improvement in a full pass, we could add noise or break.
            # For now, break to avoid infinite loop if stuck.
            # But we have time, so maybe restart local search with perturbation?
            # Let's just break for now.
            break
            
    # Return routes
    final_routes = [s['route'] for s in route_states if s['route']]
    return final_routes


def _solve_instance(instance, time_limit=25):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # 1. Construct initial routes
    construct_time = min(5, time_limit * 0.2)
    routes = _construct_initial_routes(instance, dist, construct_time)
    
    # Ensure all customers are served
    all_served = set()
    for r in routes:
        all_served.update(r)
    n_customers = instance["n_customers"]
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
        
    # 2. Local Search
    ls_time = deadline - time.time() - 2 # Leave 2s for verification
    if ls_time > 1:
        routes = _local_search(routes, dist, instance, ls_time)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> 40s per instance
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
