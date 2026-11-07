# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """
        Simulate a route (list of node ids). 
        Returns (total_distance, feasible) 
        """
        if not route:
            return 0.0, True

        t = 0.0
        battery = battery_cap
        total_dist = 0.0
        current = 0  # depot

        for node in route:
            d = dist[current, node]
            t += d / speed
            battery -= d
            total_dist += d

            if battery < -1e-9:
                return total_dist, False

            if node <= n_customers:  # Customer
                # Customer service
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return total_dist, False
                t += service[node]
            elif node in station_set:
                # Charging station
                # Recharge time: (battery_cap - battery) * recharge_rate
                recharge_time = (battery_cap - battery) * recharge_rate
                t += recharge_time
                battery = battery_cap

            current = node

        # Return to depot
        d = dist[current, 0]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return total_dist, False
        if t > horizon + 1e-9:
            return total_dist, False

        return total_dist, True

    def route_demand(route):
        return sum(demands[v] for v in route if v <= n_customers)

    def total_distance(routes):
        total = 0.0
        for route in routes:
            d, _ = simulate_route(route)
            total += d
        return total

    def is_feasible(routes):
        for route in routes:
            if route_demand(route) > capacity:
                return False
            _, feasible = simulate_route(route)
            if not feasible:
                return False
        # Check all customers served exactly once
        served = []
        for route in routes:
            for c in route:
                if c <= n_customers:
                    served.append(c)
        if len(served) != len(customers):
            return False
        if len(set(served)) != len(customers):
            return False
        return True

    def two_opt(route, max_iter=50):
        """
        Apply 2-opt improvement to a single route.
        """
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = list(route[:i]) + list(reversed(route[i:j+1])) + list(route[j+1:])
                    if route_demand(new_route) > capacity:
                        continue
                    old_dist, old_feas = simulate_route(route)
                    new_dist, new_feas = simulate_route(new_route)
                    if not new_feas:
                        continue
                    if new_dist < old_dist - 1e-6:
                        route = new_route
                        improved = True
                        old_dist = new_dist # update reference
        return route

    def or_opt(route, max_iter=50):
        """
        Apply Or-opt (relocation of 1, 2, or 3 consecutive nodes) to a single route.
        """
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i+length]
                    remaining = route[:i] + route[i+length:]
                    if not remaining:
                        continue
                    # Try inserting segment at various positions in remaining
                    for j in range(len(remaining) + 1):
                        if j >= i and j < i + length:
                            continue # skip original position
                        new_route = remaining[:j] + segment + remaining[j:]
                        if route_demand(new_route) > capacity:
                            continue
                        old_dist, old_feas = simulate_route(route)
                        new_dist, new_feas = simulate_route(new_route)
                        if not new_feas:
                            continue
                        if new_dist < old_dist - 1e-6:
                            route = new_route
                            improved = True
                            old_dist = new_dist
                            break
                    if improved:
                        break
                if improved:
                    break
        return route

    def local_search(routes, time_limit=None):
        """
        Apply 2-opt and Or-opt to all routes until no improvement or time limit.
        """
        if time_limit is None:
            time_limit = deadline - 5.0
        
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for i in range(len(routes)):
                if time.time() > time_limit:
                    break
                old_route = routes[i]
                new_route = two_opt(old_route, max_iter=20)
                if new_route != old_route:
                    routes[i] = new_route
                    improved = True
                new_route = or_opt(routes[i], max_iter=20)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True
        return routes

    def perturbation(routes, num_customers_to_move=3):
        """
        Randomly relocate a few customers to other routes or new routes.
        """
        routes = [list(r) for r in routes]
        all_customers = [c for r in routes for c in r if c <= n_customers]
        random.shuffle(all_customers)
        
        for c in all_customers[:num_customers_to_move]:
            # Find current route and remove c
            current_route_idx = -1
            for i, r in enumerate(routes):
                if c in r:
                    current_route_idx = i
                    break
            if current_route_idx == -1:
                continue
            
            routes[current_route_idx].remove(c)
            
            # Try to insert c into a random other route or create new
            other_routes = [i for i in range(len(routes)) if i != current_route_idx]
            random.shuffle(other_routes)
            
            inserted = False
            for target_idx in other_routes:
                target_route = routes[target_idx]
                # Try random position
                pos = random.randint(0, len(target_route))
                new_route = target_route[:pos] + [c] + target_route[pos:]
                if route_demand(new_route) > capacity:
                    continue
                _, feas = simulate_route(new_route)
                if feas:
                    routes[target_idx] = new_route
                    inserted = True
                    break
            
            if not inserted:
                # Create new route
                new_route = [c]
                _, feas = simulate_route(new_route)
                if feas:
                    routes.append(new_route)
                else:
                    # Put back if can't even serve alone (shouldn't happen)
                    routes[current_route_idx].append(c)

        # Remove empty routes
        routes = [r for r in routes if r]
        return routes

    # 1. Construction: Clarke-Wright Savings
    def savings_construction():
        # Start with separate routes for each customer
        routes = [[c] for c in customers]
        
        # Calculate savings
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                c1 = customers[i]
                c2 = customers[j]
                # Savings = dist(0, c1) + dist(0, c2) - dist(c1, c2)
                s = dist[0, c1] + dist[0, c2] - dist[c1, c2]
                savings.append((s, i, j))
        
        # Sort by savings descending
        savings.sort(reverse=True)
        
        # Track which route each customer is in
        route_of_customer = {c: i for i, c in enumerate(customers)}
        # Track neighbors in each route
        # For simplicity in this heuristic, we just check if they are at ends.
        # This is a simplified CW that works well enough.
        
        for s, i, j in savings:
            c1 = customers[i]
            c2 = customers[j]
            r1 = route_of_customer[c1]
            r2 = route_of_customer[c2]
            
            if r1 == r2:
                continue
            
            route1 = routes[r1]
            route2 = routes[r2]
            
            # Check if c1 is at an end of route1 and c2 is at an end of route2
            if c1 not in [route1[0], route1[-1]] or c2 not in [route2[0], route2[-1]]:
                continue
            
            # Determine orientation
            # We want to merge such that c1 and c2 become adjacent.
            # Possible merges:
            # 1. route1 + route2 (if c1 is end of r1 and c2 is start of r2)
            # 2. route1 + reversed(route2) (if c1 is end of r1 and c2 is end of r2)
            # 3. reversed(route1) + route2 (if c1 is start of r1 and c2 is start of r2)
            # 4. reversed(route1) + reversed(route2) (if c1 is start of r1 and c2 is end of r2)
            
            new_route = None
            
            if route1[-1] == c1 and route2[0] == c2:
                new_route = route1 + route2
            elif route1[-1] == c1 and route2[-1] == c2:
                new_route = route1 + route2[::-1]
            elif route1[0] == c1 and route2[0] == c2:
                new_route = route1[::-1] + route2
            elif route1[0] == c1 and route2[-1] == c2:
                new_route = route1[::-1] + route2[::-1]
            
            if new_route is None:
                continue
            
            # Check feasibility of merged route
            if route_demand(new_route) > capacity:
                continue
            _, feas = simulate_route(new_route)
            if not feas:
                continue
            
            # Merge
            routes[r1] = new_route
            routes[r2] = []
            
            # Update route_of_customer
            for c in new_route:
                if c <= n_customers:
                    route_of_customer[c] = r1
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # Check if all customers are covered. If not, add missing ones.
        covered = set()
        for r in routes:
            for c in r:
                if c <= n_customers:
                    covered.add(c)
        
        missing = [c for c in customers if c not in covered]
        for c in missing:
            routes.append([c])
            
        return routes

    # 2. Initial Solution
    best_routes = savings_construction()
    if not is_feasible(best_routes):
        # Fallback if construction failed
        best_routes = [[c] for c in customers]
    
    best_routes = local_search(best_routes)
    best_dist = total_distance(best_routes)

    # 3. Iterated Local Search
    current_routes = best_routes
    current_dist = best_dist
    
    while time.time() < deadline - 10.0:
        # Perturb
        perturbed_routes = perturbation(current_routes, num_customers_to_move=random.randint(2, 5))
        
        # Local Search
        new_routes = local_search(perturbed_routes, time_limit=deadline - 5.0)
        
        if not is_feasible(new_routes):
            continue
            
        new_dist = total_distance(new_routes)
        
        # Accept if better, or with small probability if worse (simulated annealing style)
        if new_dist < current_dist:
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_routes = [list(r) for r in current_routes]
                best_dist = current_dist
        else:
            # With small probability, accept worse solution to escape local optima
            if random.random() < 0.1:
                current_routes = new_routes
                current_dist = new_dist

    # Final check
    if not is_feasible(best_routes):
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
