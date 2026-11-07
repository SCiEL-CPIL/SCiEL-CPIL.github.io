"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 125.0  # Leave 25s safety margin before 150s kill

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

    # Precompute distance matrix using numpy
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))

    # Simulate a route to check feasibility and get metrics
    def simulate_route(route):
        """
        Simulate a route (list of customer ids).
        Returns (feasible, total_dist, end_time, end_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery_cap

        cur_time = 0.0
        cur_battery = battery_cap
        total_dist = 0.0
        cur_pos = 0

        for cust in route:
            d = dist[cur_pos][cust]
            cur_time += d / speed
            cur_battery -= d
            total_dist += d

            if cur_battery < 0:
                return False, total_dist, cur_time, cur_battery

            # Check time window
            cur_time = max(cur_time, ready[cust])
            if cur_time > due[cust]:
                return False, total_dist, cur_time, cur_battery

            cur_time += service[cust]
            cur_pos = cust

        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d

        if cur_battery < 0:
            return False, total_dist, cur_time, cur_battery

        if cur_time > horizon:
            return False, total_dist, cur_time, cur_battery

        return True, total_dist, cur_time, cur_battery

    def route_total_dist(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i + 1]]
        d += dist[route[-1]][0]
        return d

    # Construction Heuristic: Nearest Neighbor with constraints
    def construct_routes():
        unvisited = set(customers)
        routes = []

        while unvisited:
            if time.time() > deadline:
                break
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery_cap
            route_demand = 0
            total_dist = 0.0

            while unvisited:
                best_cust = None
                best_score = float('inf')

                for cust in unvisited:
                    d = dist[cur_pos][cust]
                    new_time = cur_time + d / speed
                    new_battery = cur_battery - d

                    if new_battery < 0:
                        continue
                    if route_demand + demands[cust] > capacity:
                        continue

                    new_time = max(new_time, ready[cust])
                    if new_time > due[cust]:
                        continue

                    # Check return to depot feasibility
                    return_d = dist[cust][0]
                    if new_battery - return_d < 0:
                        continue
                    if new_time + service[cust] + return_d / speed > horizon:
                        continue

                    # Score: prefer nearest, but also consider time window tightness
                    # Use distance as primary
                    if d < best_score:
                        best_score = d
                        best_cust = cust

                if best_cust is None:
                    break

                # Commit to best_cust
                d = dist[cur_pos][best_cust]
                cur_time += d / speed
                cur_battery -= d
                total_dist += d
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                route_demand += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust

            if route:
                # Final check
                feasible, _, _, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Rollback: put them back and try one-by-one
                    unvisited.update(route)
                    for cust in list(route):
                        if cust in unvisited:
                            one_route = [cust]
                            feasible, _, _, _ = simulate_route(one_route)
                            if feasible:
                                routes.append(one_route)
                                unvisited.remove(cust)
            else:
                # No customer could be added to current route, start a new one?
                # Or handle leftovers
                for cust in list(unvisited):
                    one_route = [cust]
                    feasible, _, _, _ = simulate_route(one_route)
                    if feasible:
                        routes.append(one_route)
                        unvisited.remove(cust)
                        break
                # If nothing works, we have a problem. But assume instances are feasible.

        # Handle any remaining unvisited (should be none if feasible)
        if unvisited:
            for cust in unvisited:
                routes.append([cust])

        return routes

    # Improvement Phase: Or-opt and 2-opt style moves
    def try_improve(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(routes)):
                if not routes[i]:
                    continue
                for j in range(len(routes[i])):
                    cust = routes[i][j]
                    # Remove cust from route i
                    new_route_i = routes[i][:j] + routes[i][j+1:]
                    if not new_route_i:
                        # Route i is now empty
                        feasible_i = True
                    else:
                        feasible_i, _, _, _ = simulate_route(new_route_i)

                    if not feasible_i:
                        continue

                    # Try inserting into all other routes
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        route_k = routes[k]
                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [cust] + route_k[pos:]
                            feasible_k, _, _, _ = simulate_route(new_route_k)
                            if not feasible_k:
                                continue

                            # Check capacity for new_route_k
                            if sum(demands[c] for c in new_route_k) > capacity:
                                continue

                            # Calculate distance change
                            old_dist = route_total_dist(routes[i]) + route_total_dist(route_k)
                            new_dist = route_total_dist(new_route_i) + route_total_dist(new_route_k)

                            if new_dist < old_dist - 1e-6:
                                # Perform move
                                if not new_route_i:
                                    routes[i] = []
                                else:
                                    routes[i] = new_route_i
                                routes[k] = new_route_k
                                improved = True
                                # Break out of loops
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Cleanup empty routes
            new_routes = []
            for r in routes:
                if r:
                    new_routes.append(r)
            if len(new_routes) != len(routes):
                routes = new_routes
                # Note: indices change, so loop restarts

        return routes

    # Further improvement: Intra-route 2-opt
    def try_2opt(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(routes)):
                route = routes[i]
                if len(route) < 3:
                    continue
                changed = False
                for a in range(len(route) - 1):
                    for b in range(a + 1, len(route)):
                        # Reverse segment a+1 to b
                        new_route = route[:a+1] + route[a+1:b+1][::-1] + route[b+1:]
                        feasible, _, _, _ = simulate_route(new_route)
                        if feasible:
                            old_d = route_total_dist(route)
                            new_d = route_total_dist(new_route)
                            if new_d < old_d - 1e-6:
                                routes[i] = new_route
                                improved = True
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
        return routes

    # Genetic Algorithm approach
    def genetic_algorithm(routes):
        # Encode routes as a permutation of customers
        def encode(routes):
            # Create a permutation that, when decoded, gives the routes
            # We'll use a simple encoding: list of customers in order of routes
            perm = []
            for r in routes:
                perm.extend(r)
            return perm

        def decode(perm):
            # Decode permutation into routes using a greedy approach
            # This is a simplified decoder
            unvisited = set(perm)
            decoded_routes = []

            while unvisited:
                route = []
                cur_pos = 0
                cur_time = 0.0
                cur_battery = battery_cap
                route_demand = 0

                for cust in perm:
                    if cust not in unvisited:
                        continue

                    d = dist[cur_pos][cust]
                    new_time = cur_time + d / speed
                    new_battery = cur_battery - d

                    if new_battery < 0:
                        continue
                    if route_demand + demands[cust] > capacity:
                        continue

                    new_time = max(new_time, ready[cust])
                    if new_time > due[cust]:
                        continue

                    # Check return to depot feasibility
                    return_d = dist[cust][0]
                    if new_battery - return_d < 0:
                        continue
                    if new_time + service[cust] + return_d / speed > horizon:
                        continue

                    # Accept this customer
                    cur_time = new_time + service[cust]
                    cur_battery = new_battery
                    route_demand += demands[cust]
                    route.append(cust)
                    unvisited.remove(cust)
                    cur_pos = cust

                if route:
                    decoded_routes.append(route)

            # Handle any remaining unvisited
            if unvisited:
                for cust in unvisited:
                    decoded_routes.append([cust])

            return decoded_routes

        # Initialize population
        pop_size = 50
        population = []

        # Add initial solution
        initial_perm = encode(routes)
        population.append(initial_perm)

        # Generate random permutations
        for _ in range(pop_size - 1):
            perm = list(customers)
            random.shuffle(perm)
            population.append(perm)

        def fitness(perm):
            decoded = decode(perm)
            total_dist = sum(route_total_dist(r) for r in decoded)
            # Check feasibility
            for r in decoded:
                feasible, _, _, _ = simulate_route(r)
                if not feasible:
                    return float('inf')
            return total_dist

        # Evaluate initial population
        pop_fitness = [fitness(perm) for perm in population]

        # Genetic algorithm loop
        gen = 0
        max_gens = 100
        while gen < max_gens and time.time() < deadline:
            gen += 1

            # Sort by fitness
            indexed_pop = sorted(zip(pop_fitness, population))
            pop_fitness = [f for f, _ in indexed_pop]
            population = [p for _, p in indexed_pop]

            # If best is already good enough, break
            if pop_fitness[0] < 1000:
                break

            # Selection: tournament selection
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                candidates = random.sample(range(pop_size), 3)
                best_idx = min(candidates, key=lambda i: pop_fitness[i])
                parent1 = population[best_idx]

                # Another tournament
                candidates2 = random.sample(range(pop_size), 3)
                best_idx2 = min(candidates2, key=lambda i: pop_fitness[i])
                parent2 = population[best_idx2]

                # Crossover: order crossover
                if len(parent1) == 0 or len(parent2) == 0:
                    child = parent1[:]
                else:
                    # Simple crossover
                    cut = random.randint(1, len(parent1) - 1)
                    child = parent1[:cut] + [x for x in parent2 if x not in parent1[:cut]]

                # Mutation
                if random.random() < 0.3:
                    if len(child) > 1:
                        i, j = random.sample(range(len(child)), 2)
                        child[i], child[j] = child[j], child[i]

                new_population.append(child)

            # Evaluate new population
            new_fitness = [fitness(perm) for perm in new_population]

            # Combine and select
            combined = list(zip(pop_fitness, population)) + list(zip(new_fitness, new_population))
            combined.sort(key=lambda x: x[0])
            population = [p for _, p in combined[:pop_size]]
            pop_fitness = [f for f, _ in combined[:pop_size]]

        # Decode best solution
        best_perm = population[0]
        best_routes = decode(best_perm)

        return best_routes

    routes = construct_routes()
    routes = try_improve(routes)
    routes = try_2opt(routes)

    # Try genetic algorithm if time permits
    if time.time() < deadline - 30:
        ga_routes = genetic_algorithm(routes)
        ga_dist = sum(route_total_dist(r) for r in ga_routes)
        current_dist = sum(route_total_dist(r) for r in routes)
        if ga_dist < current_dist:
            routes = ga_routes
            # Further improve GA result
            routes = try_improve(routes)
            routes = try_2opt(routes)

    # Final check: ensure all customers are present
    all_served = set()
    for r in routes:
        all_served.update(r)
    if all_served != set(customers):
        # Fallback
        missing = set(customers) - all_served
        for m in missing:
            routes.append([m])

    return routes
# EVOLVE-BLOCK-END
