<<<<<<< SEARCH
def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 5.0
    end_temp = 0.05
    max_iters = 3000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.5:
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
        elif perturbation_type < 0.8:
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        else:
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]

        new_routes = [r for r in new_routes if len(r) > 0]
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist

        temp *= cooling_rate

    return best_routes
=======
def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    def route_dist(route):
        if not route:
            return 0.0
        d = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_matrix[route[i], route[i+1]]
        d += dist_matrix[route[-1], 0]
        return d

    current_routes = [r[:] for r in routes]
    current_dists = [route_dist(r) for r in current_routes]
    current_loads = [route_load(r) for r in current_routes]
    current_dist = sum(current_dists)
    
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 10.0
    end_temp = 0.1
    max_iters = 5000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        new_dists = current_dists[:]
        new_loads = current_loads[:]
        
        perturbation_type = random.random()
        valid_move = False

        if perturbation_type < 0.6:
            # Relocate
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if new_loads[r] + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
                            new_loads[r1] -= demands[cust]
                            new_loads[r2] += demands[cust]
                            valid_move = True
        elif perturbation_type < 0.9:
            # Swap
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        new_routes[r1][pos1] = cust2
                        new_routes[r2][pos2] = cust1
                        valid_move = True
        else:
            # 2-opt within a route
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]
                    valid_move = True

        if not valid_move:
            continue

        # Check feasibility
        all_feasible = True
        served = set()
        for r_idx, route in enumerate(new_routes):
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            # Calculate new distance
            # Only recalculate for changed routes
            # For simplicity, recalculate all if changed, or track indices
            new_dist = 0.0
            for r in new_routes:
                new_dist += route_dist(r)
            
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dists = [route_dist(r) for r in current_routes]
                current_loads = [route_load(r) for r in current_routes]
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist

        temp *= cooling_rate

    return best_routes
>>>>>>> REPLACE

