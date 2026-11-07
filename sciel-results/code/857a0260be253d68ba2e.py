<<<<<<< SEARCH
    def try_improve(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # 2-opt intra
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                if two_opt_intra(routes, i):
                    improved = True

            # Or-opt moves
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                route = routes[i]
                for j in range(len(route)):
                    for length in [1, 2, 3]:
                        k = j + length
                        if k > len(route):
                            continue
                        imp, routes = or_opt_move(routes, i, j, k)
                        if imp:
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Remove stations
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                if try_remove_station(routes, i):
                    improved = True

            # Swaps (Inter-route)
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline:
                        break
                    route_i = routes[i]
                    route_j = routes[j]
                    # Only swap customers
                    for ci in route_i:
                        if ci in station_set: continue
                        for cj in route_j:
                            if cj in station_set: continue
                            if try_swap(routes, i, j, ci, cj):
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Station insertion for infeasible routes
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                f, _, _, _ = simulate_route(routes[i])
                if not f:
                    # Try to fix
                    fixed = None
                    for st in stations:
                        for pos in range(len(routes[i]) + 1):
                            test = routes[i][:pos] + [st] + routes[i][pos:]
                            if simulate_route(test)[0]:
                                fixed = test
                                break
                        if fixed:
                            break
                    if fixed:
                        routes[i] = fixed
                        improved = True

            # Remove empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes
                improved = True

        return routes
=======
    def construct_greedy_insertion():
        unvisited = set(customers)
        routes = []
        route_dists = []
        
        # Sort customers by due time (most urgent first)
        sorted_customers = sorted(customers, key=lambda c: due[c])
        
        for c in sorted_customers:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            # Try inserting into existing routes
            for i, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    if route_demand(new_route) > capacity:
                        continue
                    f, d, _, _ = simulate_route(new_route)
                    if f:
                        # Cost is the increase in distance
                        cost = d - route_dists[i]
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = i
                            best_pos = pos
            
            if best_route_idx != -1:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
                route_dists[best_route_idx] += best_cost
            else:
                # Create new route
                # Check if [c] is feasible
                f, d, _, _ = simulate_route([c])
                if f:
                    routes.append([c])
                    route_dists.append(d)
                else:
                    # Try with a station
                    fixed = None
                    for st in stations:
                        if simulate_route([st, c])[0]:
                            fixed = [st, c]
                            break
                        if simulate_route([c, st])[0]:
                            fixed = [c, st]
                            break
                    if fixed:
                        routes.append(fixed)
                        route_dists.append(simulate_route(fixed)[1])
                    else:
                        # Should not happen if instance is feasible
                        routes.append([c])
                        route_dists.append(simulate_route([c])[1])
                        
        return routes

    def simulated_annealing(routes, iterations=5000):
        current_routes = [list(r) for r in routes]
        current_dists = [simulate_route(r)[1] for r in current_routes]
        current_cost = sum(current_dists)
        
        if not all(simulate_route(r)[0] for r in current_routes):
            return current_routes

        best_routes = [list(r) for r in current_routes]
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        alpha = 0.9995
        
        for i in range(iterations):
            if time.time() > deadline:
                break
                
            move_type = random.randint(0, 3) # 0: Relocate, 1: Swap, 2: 2-Opt, 3: Or-Opt
            
            new_routes = [list(r) for r in current_routes]
            new_dists = list(current_dists)
            feasible = True
            delta_cost = 0.0
            changed = False
            
            if move_type == 0: # Relocate
                if len(new_routes) < 2:
                    continue
                i = random.randint(0, len(new_routes) - 1)
                route_i = new_routes[i]
                if not route_i: continue
                idx = random.randint(0, len(route_i) - 1)
                c = route_i[idx]
                if c in station_set: continue
                
                new_route_i = route_i[:idx] + route_i[idx+1:]
                
                j = random.randint(0, len(new_routes) - 1)
                if j == i:
                    pos = random.randint(0, len(new_route_i))
                    new_route_i = new_route_i[:pos] + [c] + new_route_i[pos:]
                    new_routes[i] = new_route_i
                    f, d, _, _ = simulate_route(new_route_i)
                    if not f:
                        feasible = False
                    else:
                        delta_cost = d - current_dists[i]
                        new_dists[i] = d
                        changed = True
                else:
                    route_j = new_routes[j]
                    pos = random.randint(0, len(route_j))
                    new_route_j = route_j[:pos] + [c] + route_j[pos:]
                    
                    if route_demand(new_route_j) > capacity:
                        feasible = False
                    else:
                        f_i, d_i, _, _ = simulate_route(new_route_i)
                        f_j, d_j, _, _ = simulate_route(new_route_j)
                        
                        if not (f_i and f_j):
                            feasible = False
                        else:
                            delta_cost = (d_i + d_j) - (current_dists[i] + current_dists[j])
                            new_routes[i] = new_route_i
                            new_routes[j] = new_route_j
                            new_dists[i] = d_i
                            new_dists[j] = d_j
                            changed = True
                            
            elif move_type == 1: # Swap
                if len(new_routes) < 2:
                    continue
                i = random.randint(0, len(new_routes) - 1)
                j = random.randint(0, len(new_routes) - 1)
                if i == j: continue
                
                route_i = new_routes[i]
                route_j = new_routes[j]
                if not route_i or not route_j: continue
                
                custs_i = [c for c in route_i if c not in station_set]
                custs_j = [c for c in route_j if c not in station_set]
                if not custs_i or not custs_j: continue
                
                c_i = random.choice(custs_i)
                c_j = random.choice(custs_j)
                
                idx_i = route_i.index(c_i)
                idx_j = route_j.index(c_j)
                
                new_route_i = route_i[:]
                new_route_j = route_j[:]
                new_route_i[idx_i] = c_j
                new_route_j[idx_j] = c_i
                
                if route_demand(new_route_i) > capacity or route_demand(new_route_j) > capacity:
                    feasible = False
                else:
                    f_i, d_i, _, _ = simulate_route(new_route_i)
                    f_j, d_j, _, _ = simulate_route(new_route_j)
                    
                    if not (f_i and f_j):
                        feasible = False
                    else:
                        delta_cost = (d_i + d_j) - (current_dists[i] + current_dists[j])
                        new_routes[i] = new_route_i
                        new_routes[j] = new_route_j
                        new_dists[i] = d_i
                        new_dists[j] = d_j
                        changed = True
                        
            elif move_type == 2: # 2-Opt
                i = random.randint(0, len(new_routes) - 1)
                route_i = new_routes[i]
                if len(route_i) < 2: continue
                
                a = random.randint(0, len(route_i) - 2)
                b = random.randint(a + 1, len(route_i) - 1)
                
                new_route_i = route_i[:a+1] + route_i[a+1:b+1][::-1] + route_i[b+1:]
                
                f, d, _, _ = simulate_route(new_route_i)
                if not f:
                    feasible = False
                else:
                    delta_cost = d - current_dists[i]
                    new_routes[i] = new_route_i
                    new_dists[i] = d
                    changed = True
                    
            else: # Or-Opt
                i = random.randint(0, len(new_routes) - 1)
                route_i = new_routes[i]
                if len(route_i) < 3: continue
                
                length = random.randint(1, 2)
                start = random.randint(0, len(route_i) - length)
                segment = route_i[start:start+length]
                
                if any(c in station_set for c in segment):
                    feasible = False
                else:
                    new_route_i = route_i[:start] + route_i[start+length:]
                    pos = random.randint(0, len(new_route_i))
                    new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
                    
                    if route_demand(new_route_i_test) > capacity:
                        feasible = False
                    else:
                        f, d, _, _ = simulate_route(new_route_i_test)
                        if not f:
                            feasible = False
                        else:
                            delta_cost = d - current_dists[i]
                            new_routes[i] = new_route_i_test
                            new_dists[i] = d
                            changed = True
            
            if not feasible or not changed:
                continue
                
            new_cost = current_cost + delta_cost
            
            if new_cost < current_cost or random.random() < math.exp(-delta_cost / T):
                current_routes = new_routes
                current_dists = new_dists
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_routes = [list(r) for r in current_routes]
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_routes
>>>>>>> REPLACE

