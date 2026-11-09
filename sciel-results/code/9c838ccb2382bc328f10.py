"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _dist_matrix(instance):
    """Vectorized Euclidean distance matrix."""
    coords = np.array(instance["coords"], dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    return D


def _route_time_and_battery(route, instance, dist):
    """
    Simulate a route (list of customer/station node ids, no depot).
    Returns (total_time, total_distance, feasible)
    """
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    # Check capacity
    total_demand = 0
    for node in route:
        if node not in stations:
            total_demand += demands[node]
    if total_demand > capacity:
        return (horizon + 1, 0.0, False)

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for node in route:
        d = float(dist[prev, node])
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return (horizon + 1, total_dist + d, False)

        if node in stations:
            energy_needed = battery_max - battery
            t += energy_needed * recharge_rate
            battery = battery_max
        else:
            if t < ready[node]:
                t = float(ready[node])
            if t > due[node] + 1e-9:
                return (horizon + 1, total_dist + d, False)
            t += float(service[node])

        total_dist += d
        prev = node

    # Return to depot
    d = float(dist[prev, 0])
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return (horizon + 1, total_dist + d, False)
    if t > horizon + 1e-9:
        return (horizon + 1, total_dist + d, False)

    total_dist += d
    return (t, total_dist, True)


def _route_demand(route, instance):
    stations = set(instance["stations"])
    return sum(instance["demands"][node] for node in route if node not in stations)


def _check_route_feasibility(route, instance, dist):
    _, _, feasible = _route_time_and_battery(route, instance, dist)
    return feasible


def _insert_customer(route, customer, instance, dist):
    """
    Try to insert a customer into the route at the best position.
    Returns (new_route, feasible) or (None, False) if no feasible insertion.
    """
    n = len(route)
    best_pos = -1
    best_cost = float('inf')
    stations = set(instance["stations"])
    current_demand = sum(instance["demands"][node] for node in route if node not in stations)
    
    if current_demand + instance["demands"][customer] > instance["capacity"]:
        return (None, False)
    
    for pos in range(n + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        _, total_dist_new, feasible = _route_time_and_battery(new_route, instance, dist)
        if feasible:
            if pos == 0:
                add_dist = float(dist[0, customer]) + float(dist[customer, route[0]]) - float(dist[0, route[0]])
            elif pos == n:
                add_dist = float(dist[route[-1], customer]) + float(dist[customer, 0]) - float(dist[route[-1], 0])
            else:
                add_dist = float(dist[route[pos-1], customer]) + float(dist[customer, route[pos]]) - float(dist[route[pos-1], route[pos]])
            if add_dist < best_cost:
                best_cost = add_dist
                best_pos = pos
    
    if best_pos == -1:
        return (None, False)
    
    new_route = route[:best_pos] + [customer] + route[best_pos:]
    return (new_route, True)


def _2opt(route, instance, dist, deadline=None):
    """
    Apply 2-opt local search to improve a route.
    Returns improved route.
    """
    n = len(route)
    if n < 2:
        return route
    
    best_route = route[:]
    _, best_dist, feasible = _route_time_and_battery(best_route, instance, dist)
    if not feasible:
        return best_route
    
    improved = True
    while improved:
        improved = False
        if deadline and time.time() > deadline:
            break
        for i in range(n - 1):
            if deadline and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline and time.time() > deadline:
                    break
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                _, new_dist, feasible = _route_time_and_battery(new_route, instance, dist)
                if feasible and new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
    
    return best_route


def _or_opt(route, instance, dist, deadline=None):
    """
    Apply Or-opt (move 1-3 consecutive nodes) local search.
    Returns improved route.
    """
    n = len(route)
    if n < 3:
        return route
    
    best_route = route[:]
    _, best_dist, feasible = _route_time_and_battery(best_route, instance, dist)
    if not feasible:
        return best_route
    
    improved = True
    while improved:
        improved = False
        if deadline and time.time() > deadline:
            break
        for seg_len in [1, 2, 3]:
            if deadline and time.time() > deadline:
                break
            for i in range(n - seg_len + 1):
                if deadline and time.time() > deadline:
                    break
                segment = best_route[i:i+seg_len]
                remaining = best_route[:i] + best_route[i+seg_len:]
                m = len(remaining)
                for pos in range(m + 1):
                    if pos == i or (pos > i and pos <= i + seg_len):
                        continue
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    _, new_dist, feasible = _route_time_and_battery(new_route, instance, dist)
                    if feasible and new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_route


def _inter_route_relocate(routes, instance, dist, deadline=None):
    """
    Try to relocate a customer from one route to another.
    Returns (improved_routes, total_dist).
    """
    n_routes = len(routes)
    if n_routes < 2:
        return routes, float('inf')
    
    # Compute total distance
    total_dist = 0.0
    for r in routes:
        _, d, _ = _route_time_and_battery(r, instance, dist)
        total_dist += d
    
    best_routes = [r[:] for r in routes]
    best_total = total_dist
    improved = True
    
    while improved:
        improved = False
        if deadline and time.time() > deadline:
            break
        for i in range(n_routes):
            if deadline and time.time() > deadline:
                break
            route_i = best_routes[i]
            for pos_i in range(len(route_i)):
                if deadline and time.time() > deadline:
                    break
                cust = route_i[pos_i]
                # Try removing from route i
                new_route_i = route_i[:pos_i] + route_i[pos_i+1:]
                _, dist_i, feas_i = _route_time_and_battery(new_route_i, instance, dist)
                if not feas_i:
                    continue
                
                for j in range(n_routes):
                    if j == i:
                        continue
                    if deadline and time.time() > deadline:
                        break
                    route_j = best_routes[j]
                    new_route_j, feas = _insert_customer(route_j, cust, instance, dist)
                    if not feas:
                        continue
                    _, dist_j, _ = _route_time_and_battery(new_route_j, instance, dist)
                    new_total = dist_i + dist_j + (total_dist - float(dist[0, route_i[0]]) - float(dist[route_i[-1], 0]) if len(route_i) > 0 else 0)
                    # Recalculate properly
                    new_total = 0.0
                    for k in range(n_routes):
                        if k == i:
                            new_total += dist_i
                        elif k == j:
                            new_total += dist_j
                        else:
                            _, d_k, _ = _route_time_and_battery(best_routes[k], instance, dist)
                            new_total += d_k
                    
                    if new_total < best_total - 1e-9:
                        best_routes[i] = new_route_i
                        best_routes[j] = new_route_j
                        best_total = new_total
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_routes, best_total


def _solve_instance(instance, deadline):
    """Solve a single instance, returning list of routes."""
    dist = _dist_matrix(instance)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]
    
    # Greedy construction: build routes by inserting customers
    routes = []
    unvisited = set(customers)
    
    while unvisited:
        # Start a new route with the customer with earliest due time
        first = min(unvisited, key=lambda v: instance["due"][v])
        unvisited.remove(first)
        route = [first]
        
        # Try to insert remaining customers
        changed = True
        while changed:
            changed = False
            remaining = list(unvisited)
            remaining.sort(key=lambda v: instance["due"][v])
            
            for cust in remaining[:]:
                if time.time() > deadline - 10:
                    break
                new_route, feasible = _insert_customer(route, cust, instance, dist)
                if feasible:
                    route = new_route
                    unvisited.remove(cust)
                    changed = True
        
        routes.append(route)
    
    # Local search: 2-opt + Or-opt + inter-route relocate, time-budgeted
    local_deadline = deadline - 5  # safety margin
    
    # First pass: intra-route optimization
    for i in range(len(routes)):
        if time.time() > local_deadline:
            break
        routes[i] = _2opt(routes[i], instance, dist, local_deadline)
    
    for i in range(len(routes)):
        if time.time() > local_deadline:
            break
        routes[i] = _or_opt(routes[i], instance, dist, local_deadline)
    
    # Inter-route relocate
    if time.time() < local_deadline:
        routes, _ = _inter_route_relocate(routes, instance, dist, local_deadline)
    
    # Second pass: re-optimize after inter-route changes
    for i in range(len(routes)):
        if time.time() > local_deadline:
            break
        routes[i] = _2opt(routes[i], instance, dist, local_deadline)
        routes[i] = _or_opt(routes[i], instance, dist, local_deadline)
    
    # Remove empty routes
    routes = [r for r in routes if len(r) > 0]
    
    return routes


def plan(instance):
    """
    Plan routes for the given instance.
    Returns a list of routes, each a list of node ids.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    routes = _solve_instance(instance, deadline)
    
    # Verify all customers are served exactly once
    all_served = []
    for route in routes:
        for node in route:
            if node > 0 and node <= instance["n_customers"]:
                all_served.append(node)
    
    expected = set(range(1, instance["n_customers"] + 1))
    served_set = set(all_served)
    
    if served_set != expected or len(all_served) != len(expected):
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END
