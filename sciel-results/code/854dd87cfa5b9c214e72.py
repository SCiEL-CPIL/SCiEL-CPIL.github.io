"""EV routing with time windows and battery constraints using LNS."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

# Global deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 135.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - time.time()


def _build_dist_matrix(coords):
    n = len(coords)
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist


def _simulate_route(route, instance, dist, battery, capacity, speed, recharge_rate, horizon):
    """
    Simulate a route. Returns (total_distance, valid, total_time).
    route: list of node ids (customers and stations), no depot.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations_set = set(instance["stations"])
    n_customers = instance["n_customers"]

    total_dist = 0.0
    curr_batt = battery
    curr_time = 0.0
    curr_load = 0.0
    prev_node = 0  # depot

    for node in route:
        d = dist[prev_node][node]
        curr_time += d / speed
        curr_batt -= d

        if curr_batt < -1e-9:
            return total_dist, False, curr_time

        if node in stations_set:
            # Charging station: add recharge time, refill battery
            energy_needed = battery - curr_batt
            curr_time += energy_needed * recharge_rate
            curr_batt = battery
        else:
            # Customer
            curr_time = max(curr_time, ready[node])
            if curr_time > due[node] + 1e-9:
                return total_dist, False, curr_time
            curr_load += demands[node]
            if curr_load > capacity + 1e-9:
                return total_dist, False, curr_time
            curr_time += service[node]

        total_dist += d
        prev_node = node

    # Return to depot
    d = dist[prev_node][0]
    curr_time += d / speed
    curr_batt -= d
    if curr_batt < -1e-9:
        return total_dist, False, curr_time
    if curr_time > horizon + 1e-9:
        return total_dist, False, curr_time

    total_dist += d
    return total_dist, True, curr_time


def _construct_greedy_route(instance, dist, unvisited, battery, capacity, speed, recharge_rate, horizon, max_nodes=15):
    """
    Greedily build a feasible route starting from depot, visiting nodes from unvisited.
    Returns (route, visited_set) or (None, None) if no feasible node can be added.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations_set = set(instance["stations"])

    route = []
    visited = set()
    curr_batt = battery
    curr_time = 0.0
    curr_load = 0.0
    prev_node = 0

    for _ in range(max_nodes):
        best_node = None
        best_score = float('inf')

        candidates = list(unvisited - visited)
        # Also consider adding a station if battery is low
        if curr_batt < battery * 0.4:
            for s in instance["stations"]:
                if s not in visited:
                    candidates.append(s)

        for node in candidates:
            d = dist[prev_node][node]
            new_time = curr_time + d / speed
            new_batt = curr_batt - d

            if new_batt < -1e-9:
                continue

            if node in stations_set:
                new_time += (battery - new_batt) * recharge_rate
                new_batt = battery
                new_load = curr_load
            else:
                new_time = max(new_time, ready[node])
                if new_time > due[node] + 1e-9:
                    continue
                new_load = curr_load + demands[node]
                if new_load > capacity + 1e-9:
                    continue
                new_time += service[node]

            # Check if we can return to depot
            d_back = dist[node][0]
            final_batt = new_batt - d_back
            final_time = new_time + d_back / speed
            if final_batt < -1e-9 or final_time > horizon + 1e-9:
                continue

            score = d
            if score < best_score:
                best_score = score
                best_node = node

        if best_node is None:
            break

        d = dist[prev_node][best_node]
        curr_time += d / speed
        curr_batt -= d

        if best_node in stations_set:
            curr_time += (battery - curr_batt) * recharge_rate
            curr_batt = battery
        else:
            curr_time = max(curr_time, ready[best_node])
            curr_time += service[best_node]
            curr_load += demands[best_node]
            visited.add(best_node)

        route.append(best_node)
        prev_node = best_node

    if not route:
        return None, None
    return route, visited


def _insert_customer_optimal(route, customer, instance, dist, battery, capacity, speed, recharge_rate, horizon):
    """
    Try to insert a customer into a route at the best feasible position.
    Returns (new_route, cost) or (None, inf) if infeasible.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations_set = set(instance["stations"])

    if not route:
        # Check if single customer route is feasible
        d1 = dist[0][customer]
        d2 = dist[customer][0]
        t1 = d1 / speed
        if t1 > due[customer] + 1e-9:
            return None, float('inf')
        t2 = t1 + service[customer] + d2 / speed
        b1 = battery - d1
        b2 = b1 - d2
        if b2 < -1e-9 or t2 > horizon + 1e-9:
            return None, float('inf')
        if demands[customer] > capacity + 1e-9:
            return None, float('inf')
        return [customer], d1 + d2

    best_pos = -1
    best_cost = float('inf')
    curr_load = sum(demands[n] for n in route if n not in stations_set)
    
    if curr_load + demands[customer] > capacity + 1e-9:
        return None, float('inf')

    for pos in range(len(route) + 1):
        # Create new route with customer inserted at pos
        if pos == 0:
            new_route = [customer] + route
        elif pos == len(route):
            new_route = route + [customer]
        else:
            new_route = route[:pos] + [customer] + route[pos:]
        
        # Quick check: total demand is already checked
        d_new, valid, t_new = _simulate_route(new_route, instance, dist, battery, capacity, speed, recharge_rate, horizon)
        
        if valid:
            # Calculate cost increase
            if pos == 0:
                old_cost = dist[0][route[0]]
                new_cost = dist[0][customer] + dist[customer][route[0]]
            elif pos == len(route):
                old_cost = dist[route[-1]][0]
                new_cost = dist[route[-1]][customer] + dist[customer][0]
            else:
                old_cost = dist[route[pos-1]][route[pos]]
                new_cost = dist[route[pos-1]][customer] + dist[customer][route[pos]]
            
            cost_inc = new_cost - old_cost
            if cost_inc < best_cost:
                best_cost = cost_inc
                best_pos = pos
                best_new_route = new_route

    if best_pos == -1:
        return None, float('inf')
    return best_new_route, best_cost


def _local_search_lns(instance, dist, routes, battery, capacity, speed, recharge_rate, horizon, max_iter=50):
    """
    Large Neighborhood Search: remove a subset of customers and re-insert them.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations_set = set(instance["stations"])
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))

    # Calculate initial total distance
    total_dist = 0.0
    for r in routes:
        d, valid, _ = _simulate_route(r, instance, dist, battery, capacity, speed, recharge_rate, horizon)
        if valid:
            total_dist += d
        else:
            total_dist += 10000.0  # Penalty for invalid

    best_routes = [list(r) for r in routes]
    best_total = total_dist

    for iter_num in range(max_iter):
        if _remaining_time() < 10.0:
            break

        # Select a random subset of customers to remove (LNS removal)
        # Remove 20-40% of customers
        removal_fraction = random.uniform(0.2, 0.4)
        n_remove = max(2, int(len(customers) * removal_fraction))
        removed_customers = random.sample(customers, n_remove)
        
        # Remove these customers from routes
        new_routes = []
        for r in routes:
            new_r = [n for n in r if n not in removed_customers]
            if new_r:
                new_routes.append(new_r)
        
        # Ensure all removed customers are unassigned
        unassigned = set(removed_customers)
        
        # Re-insert customers using best insertion
        # Sort by due date (earliest first) to prioritize tight windows
        unassigned_sorted = sorted(unassigned, key=lambda c: due[c])
        
        success = True
        for c in unassigned_sorted:
            best_cost = float('inf')
            best_route_idx = -1
            best_new_route = None
            
            for ri, r in enumerate(new_routes):
                new_r, cost = _insert_customer_optimal(r, c, instance, dist, battery, capacity, speed, recharge_rate, horizon)
                if new_r is not None and cost < best_cost:
                    best_cost = cost
                    best_route_idx = ri
                    best_new_route = new_r
            
            if best_route_idx == -1:
                # Try creating a new route for this customer
                new_r, cost = _insert_customer_optimal([], c, instance, dist, battery, capacity, speed, recharge_rate, horizon)
                if new_r is not None:
                    new_routes.append(new_r)
                    best_route_idx = len(new_routes) - 1
                    best_new_route = new_r
                else:
                    # Cannot insert, revert
                    success = False
                    break
            
            if success:
                new_routes[best_route_idx] = best_new_route

        if not success:
            continue

        # Evaluate new solution
        new_total = 0.0
        valid_all = True
        for r in new_routes:
            d, valid, _ = _simulate_route(r, instance, dist, battery, capacity, speed, recharge_rate, horizon)
            if valid:
                new_total += d
            else:
                valid_all = False
                break
        
        if not valid_all:
            continue

        # Check all customers are served
        served = set()
        for r in new_routes:
            for n in r:
                if n not in stations_set:
                    served.add(n)
        
        if served != set(customers):
            continue

        if new_total < best_total - 1e-6:
            best_total = new_total
            best_routes = [list(r) for r in new_routes]
            routes = [list(r) for r in new_routes]

    return best_routes


def _two_opt_improve(route, instance, dist, battery, capacity, speed, recharge_rate, horizon):
    """Apply 2-opt improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 5
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if _remaining_time() < 5.0:
                return route
            for j in range(i + 1, n):
                if _remaining_time() < 5.0:
                    return route
                # Reverse segment i..j
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                # Calculate distance difference
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    # Potential improvement, check feasibility
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d_val, valid, _ = _simulate_route(new_route, instance, dist, battery, capacity, speed, recharge_rate, horizon)
                    if valid:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def plan(instance):
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 135.0

    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    dist = _build_dist_matrix(coords)
    customers = list(range(1, n_customers + 1))

    # Phase 1: Construct initial solution using greedy route building
    unvisited = set(customers)
    routes = []
    max_nodes_per_route = 10

    while unvisited:
        if _remaining_time() < 15.0:
            break
        route, visited = _construct_greedy_route(
            instance, dist, unvisited, battery, capacity, speed, recharge_rate, horizon, max_nodes_per_route
        )
        if route is None:
            # Force add a single customer
            c = min(unvisited, key=lambda c: due[c])
            routes.append([c])
            unvisited.discard(c)
        else:
            routes.append(route)
            unvisited -= visited

    # Ensure all customers are covered
    covered = set()
    for r in routes:
        for n in r:
            if n not in set(stations):
                covered.add(n)
    for c in customers:
        if c not in covered:
            routes.append([c])

    # Phase 2: 2-opt improvement on each route
    if _remaining_time() > 20.0:
        for i in range(len(routes)):
            routes[i] = _two_opt_improve(routes[i], instance, dist, battery, capacity, speed, recharge_rate, horizon)

    # Phase 3: LNS improvement
    if _remaining_time() > 30.0:
        routes = _local_search_lns(instance, dist, routes, battery, capacity, speed, recharge_rate, horizon, max_iter=100)

    # Phase 4: Final 2-opt pass
    if _remaining_time() > 15.0:
        for i in range(len(routes)):
            routes[i] = _two_opt_improve(routes[i], instance, dist, battery, capacity, speed, recharge_rate, horizon)

    # Filter out empty routes
    routes = [r for r in routes if r]

    return routes
# EVOLVE-BLOCK-END
