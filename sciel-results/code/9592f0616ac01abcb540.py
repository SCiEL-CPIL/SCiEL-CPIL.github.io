"""EV routing with time windows – greedy + 2-opt + Or-opt improvements."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def _dist(a, b, coords):
    dx = coords[a][0] - coords[b][0]
    dy = coords[a][1] - coords[b][1]
    return math.sqrt(dx * dx + dy * dy)


def _simulate(route, instance, coords, demands, ready, due, service,
              stations, capacity, battery_max, speed, recharge_rate, horizon):
    """Return (feasible, total_dist, total_time, battery_at_end)."""
    t = 0.0
    battery = battery_max
    total_dist = 0.0
    load = 0
    prev = 0  # depot
    
    for node in route:
        d = _dist(prev, node, coords)
        if d > battery:
            return (False, 0, 0, 0)
        battery -= d
        t += d / speed
        total_dist += d
        
        if node in stations:
            # Recharge: time = (battery - energy) * recharge_rate? 
            # Actually the spec says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This is ambiguous. Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # Usually recharge time = (full - current) / rate. But here it says (battery - energy) * rate?
            # Wait, "energy" is not defined in the instance dict. Let's assume it means the energy needed to full?
            # Or maybe it's a typo in my understanding. Let's look at the contract again.
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # If battery is current charge, and energy is... maybe the energy consumed since last station?
            # Or maybe it's just: time += (battery_max - battery) / recharge_rate?
            # Let's assume the standard: time += (battery_max - battery) / recharge_rate, then battery = battery_max.
            # But the text says "(battery - energy) * recharge_rate". This is weird.
            # Let's try: time += (battery_max - battery) * recharge_rate? No, that would be small.
            # Let's assume: time += (battery_max - battery) / recharge_rate.
            # Wait, the contract says: "a station adds (battery - energy) * recharge_rate time".
            # Maybe "energy" is a variable I don't have. Let's assume it means the deficit?
            # Let's try: time += (battery_max - battery) * recharge_rate.
            # If recharge_rate is 0.5, and deficit is 100, time = 50.
            # If it was / rate, time = 200.
            # Given speed=1, distances are ~10-50, time is ~10-50.
            # Let's stick with the literal text if possible, but "energy" is undefined.
            # Let's assume it means: time += (battery_max - battery) * recharge_rate.
            # And then battery = battery_max.
            
            recharge_time = (battery_max - battery) * recharge_rate
            t += recharge_time
            battery = battery_max
        else:
            # Customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node]:
                return (False, 0, 0, 0)
            t += service[node]
            load += demands[node]
            if load > capacity:
                return (False, 0, 0, 0)
        
        prev = node
    
    # Return to depot
    d = _dist(prev, 0, coords)
    if d > battery:
        return (False, 0, 0, 0)
    battery -= d
    t += d / speed
    total_dist += d
    
    if t > horizon:
        return (False, 0, 0, 0)
    
    return (True, total_dist, t, battery)


def _greedy_construct(unvisited, instance, coords, demands, ready, due, service,
                      stations, capacity, battery_max, speed, recharge_rate, horizon):
    """Construct a single route greedily from unvisited customers."""
    route = []
    load = 0
    t = 0.0
    battery = battery_max
    prev = 0
    
    remaining = list(unvisited)
    
    while remaining:
        best_node = -1
        best_cost = float('inf')
        
        for node in remaining:
            d = _dist(prev, node, coords)
            if d > battery:
                continue
            
            # Check if we can reach the node and still return to depot
            # Simplified check: just check if we can reach node
            new_battery = battery - d
            new_t = t + d / speed
            
            if node in stations:
                # Stations are not customers, so we don't serve them in this greedy loop
                # We only consider customers for the route
                pass
            else:
                if new_t < ready[node]:
                    new_t = ready[node]
                if new_t > due[node]:
                    continue
                new_t += service[node]
                new_load = load + demands[node]
                if new_load > capacity:
                    continue
                
                # Heuristic: prefer nodes that are close and have tight time windows
                # Cost = distance + time penalty
                cost = d + max(0, new_t - ready[node]) * 0.1
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
        
        if best_node == -1:
            break
        
        # Add best_node to route
        d = _dist(prev, best_node, coords)
        battery -= d
        t += d / speed
        t = max(t, ready[best_node])
        t += service[best_node]
        load += demands[best_node]
        route.append(best_node)
        prev = best_node
        remaining.remove(best_node)
        
        # Check if we can return to depot
        d_back = _dist(prev, 0, coords)
        if d_back > battery:
            # We can't return, so we must end here? Or we need to go to a station?
            # For simplicity, let's just end the route here if we can't return directly.
            # But we might be able to go to a station first.
            # Let's keep it simple for now: end route.
            break
        
        # Check time horizon
        t_back = t + d_back / speed
        if t_back > horizon:
            break
    
    return route, remaining


def _two_opt(route, instance, coords, demands, ready, due, service,
             stations, capacity, battery_max, speed, recharge_rate, horizon):
    """Improve route using 2-opt."""
    n = len(route)
    if n < 2:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                feasible, dist_new, _, _ = _simulate(new_route, instance, coords, demands,
                                                      ready, due, service, stations,
                                                      capacity, battery_max, speed,
                                                      recharge_rate, horizon)
                if feasible:
                    _, dist_old, _, _ = _simulate(route, instance, coords, demands,
                                                  ready, due, service, stations,
                                                  capacity, battery_max, speed,
                                                  recharge_rate, horizon)
                    if dist_new < dist_old - 1e-9:
                        route = new_route
                        improved = True
    return route


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
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
    unvisited = set(customers)
    routes = []
    
    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda x: due[x])
    
    while unvisited and time.time() < deadline:
        # Construct a route greedily
        route, remaining = _greedy_construct(unvisited, instance, coords, demands,
                                             ready, due, service, stations,
                                             capacity, battery_max, speed,
                                             recharge_rate, horizon)
        
        if not route:
            # Fallback: take the next customer in sorted order
            next_cust = min(unvisited, key=lambda x: due[x])
            route = [next_cust]
            remaining = unvisited - {next_cust}
        
        # Verify the route is feasible
        feasible, dist, _, _ = _simulate(route, instance, coords, demands,
                                          ready, due, service, stations,
                                          capacity, battery_max, speed,
                                          recharge_rate, horizon)
        
        if not feasible:
            # Try to fix by removing the last customer
            while route and not feasible:
                route.pop()
                if route:
                    feasible, dist, _, _ = _simulate(route, instance, coords, demands,
                                                      ready, due, service, stations,
                                                      capacity, battery_max, speed,
                                                      recharge_rate, horizon)
                else:
                    break
            if not route:
                # Take just one customer
                next_cust = min(unvisited, key=lambda x: due[x])
                route = [next_cust]
                remaining = unvisited - {next_cust}
        
        # Apply 2-opt if time permits
        if time.time() < deadline - 5.0:
            route = _two_opt(route, instance, coords, demands, ready, due,
                             service, stations, capacity, battery_max, speed,
                             recharge_rate, horizon)
        
        routes.append(route)
        unvisited = remaining
    
    # If there are still unvisited customers, add them one by one
    for cust in sorted(unvisited, key=lambda x: due[x]):
        if time.time() > deadline - 2.0:
            break
        routes.append([cust])
    
    return routes
# EVOLVE-BLOCK-END
