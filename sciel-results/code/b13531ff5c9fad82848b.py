import numpy as np

def plan(instance):
    coords = np.array(instance['coords'])
    n_nodes = len(coords)
    
    # Compute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    customers = list(range(1, instance['n_customers'] + 1))
    stations = set(instance['stations'])
    capacity = instance['capacity']
    battery_cap = instance['battery']
    recharge_rate = instance['recharge_rate']
    horizon = instance['horizon']
    
    def is_feasible(seq):
        t = 0.0
        b = battery_cap
        loc = 0
        total_dist = 0.0
        
        for node in seq:
            d = dist[loc, node]
            b -= d
            t += d
            total_dist += d
            
            if b < 0:
                return False, total_dist
                
            loc = node
            
            if node in stations:
                charge_time = (battery_cap - b) / recharge_rate
                t += charge_time
                b = battery_cap
            else:
                ready = instance['ready'][node]
                due = instance['due'][node]
                service = instance['service'][node]
                
                if t < ready:
                    t = ready
                
                if t > due:
                    return False, total_dist
                
                t += service
        
        d = dist[loc, 0]
        b -= d
        t += d
        total_dist += d
        
        if b < 0 or t > horizon:
            return False, total_dist
            
        return True, total_dist

    def get_demand(route):
        return sum(instance['demands'][c] for c in route if c not in stations)

    def try_insert(customer, routes, exclude_route_idx=None):
        """
        Tries to insert 'customer' into one of the existing routes.
        Returns (route_index, position, station_info, new_route_distance) or None.
        """
        best_cost = float('inf')
        best_config = None
        
        for r_idx, route in enumerate(routes):
            if exclude_route_idx is not None and r_idx == exclude_route_idx:
                # We still might want to re-insert into the same route at a different position
                # But usually re-insertion into the same route is handled by checking all positions.
                # If we exclude, we skip this route entirely.
                continue
                
            current_demand = get_demand(route)
            if current_demand + instance['demands'][customer] > capacity:
                continue
                
            orig_feas, orig_dist = is_feasible(route)
            if not orig_feas:
                continue
                
            # Try inserting at every position
            for pos in range(len(route) + 1):
                new_seq = route[:pos] + [customer] + route[pos:]
                
                feas, d_cost = is_feasible(new_seq)
                if feas:
                    # The cost is the total distance of the new route.
                    # We want to minimize the sum of all route distances.
                    # Change in cost = d_cost - orig_dist
                    # We store the absolute d_cost for comparison, but we need to account for the fact
                    # that other routes don't change.
                    # So minimizing d_cost is equivalent to minimizing the total sum change.
                    if d_cost < best_cost:
                        best_cost = d_cost
                        best_config = (r_idx, pos, None)
                            
                if not feas:
                    # Try inserting a station
                    for s in stations:
                        for s_pos in range(len(new_seq) + 1):
                            seq_with_s = new_seq[:s_pos] + [s] + new_seq[s_pos:]
                            feas_s, d_cost_s = is_feasible(seq_with_s)
                            if feas_s:
                                if d_cost_s < best_cost:
                                    best_cost = d_cost_s
                                    best_config = (r_idx, pos, (s, s_pos))
        
        return best_config, best_cost

    # 1. Initial Construction
    # Sort by distance to depot
    customers_sorted = sorted(customers, key=lambda c: dist[0, c])
    routes = []
    
    for c in customers_sorted:
        config, cost = try_insert(c, routes)
        if config is not None:
            r_idx, pos, station_info = config
            route = routes[r_idx]
            route.insert(pos, c)
            if station_info:
                s, s_pos = station_info
                route.insert(s_pos, s)
        else:
            # New route
            # Check if [c] is feasible
            feas, _ = is_feasible([c])
            if feas:
                routes.append([c])
            else:
                found = False
                for s in stations:
                    if is_feasible([s, c])[0]:
                        routes.append([s, c])
                        found = True
                        break
                    if is_feasible([c, s])[0]:
                        routes.append([c, s])
                        found = True
                        break
                if not found:
                    routes.append([c])

    # 2. Local Search: Re-insertion
    improved = True
    max_iterations = 50
    iter_count = 0
    current_total_dist = sum(is_feasible(r)[1] for r in routes if is_feasible(r)[0])
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            for i in range(len(route)):
                node = route[i]
                if node in stations:
                    continue # Only move customers
                
                # Remove node
                temp_route = route[:i] + route[i+1:]
                # Check if temp_route is feasible
                feas_temp, _ = is_feasible(temp_route)
                if not feas_temp:
                    continue
                
                # Try to re-insert node
                config, new_cost = try_insert(node, routes, exclude_route_idx=None)
                # Note: try_insert checks all routes, including r_idx.
                # If it finds a better position in the same route, it will return it.
                # If it finds a better position in another route, it will return that.
                
                if config is not None:
                    # We need to compare the new total distance with the old total distance.
                    # Old total = current_total_dist
                    # New total = current_total_dist - dist(temp_route) + dist(route_with_node_inserted)
                    # Actually, try_insert returns the cost of the *new* route containing the node.
                    # The other routes remain unchanged.
                    # So the change is:
                    # delta = new_cost - (is_feasible(route)[1])
                    # Wait, is_feasible(route) is the old route distance.
                    # But we removed the node, so the "old" state for comparison is the sum of all routes.
                    # Let's just compute the total distance after the move and compare.
                    
                    new_routes = [r[:] for r in routes]
                    new_routes[r_idx] = temp_route
                    
                    r_idx_new, pos_new, station_info_new = config
                    new_routes[r_idx_new].insert(pos_new, node)
                    if station_info_new:
                        s, s_pos = station_info_new
                        new_routes[r_idx_new].insert(s_pos, s)
                    
                    # Check if all new routes are feasible
                    valid = True
                    new_total_dist = 0
                    for nr in new_routes:
                        feas, d = is_feasible(nr)
                        if not feas:
                            valid = False
                            break
                        new_total_dist += d
                    
                    if valid and new_total_dist < current_total_dist:
                        routes = new_routes
                        current_total_dist = new_total_dist
                        improved = True
                        break # Restart iteration as routes changed
                else:
                    # Could not re-insert? Should not happen as it was feasible before.
                    pass

    return routes
