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
    demands = np.array(instance['demands'])
    ready = np.array(instance['ready'])
    due = np.array(instance['due'])
    service = np.array(instance['service'])

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
                r = ready[node]
                d_ = due[node]
                s = service[node]
                
                if t < r:
                    t = r
                
                if t > d_:
                    return False, total_dist
                
                t += s
        
        d = dist[loc, 0]
        b -= d
        t += d
        total_dist += d
        
        if b < 0 or t > horizon:
            return False, total_dist
            
        return True, total_dist

    def get_demand(route):
        return sum(demands[c] for c in route if c not in stations)

    # 1. Initial Construction: Greedy Nearest Neighbor
    routes = []
    remaining = set(customers)
    
    while remaining:
        # Start a new route with the customer having the smallest due date
        start_node = min(remaining, key=lambda x: due[x])
        remaining.remove(start_node)
        
        current_route = [start_node]
        
        # Ensure start_node is feasible
        if not is_feasible(current_route)[0]:
             # Try adding a station
             best_seq = None
             best_dist = float('inf')
             for s in stations:
                 for seq in [[s, start_node], [start_node, s]]:
                     if is_feasible(seq)[0]:
                         _, d = is_feasible(seq)
                         if d < best_dist:
                             best_dist = d
                             best_seq = seq
             if best_seq:
                 current_route = best_seq
             else:
                 # Fallback
                 current_route = [start_node]
        
        # Greedily add customers
        added = True
        while added:
            added = False
            current_demand = get_demand(current_route)
            
            # Find the best candidate to insert (first feasible found in nearest order? or just first feasible?)
            # Let's try to find the one that minimizes distance increase.
            best_c = None
            best_pos = None
            best_delta = float('inf')
            
            # To speed up, we can sort remaining by distance to current_route?
            # For now, just iterate and pick the first feasible one with min distance increase.
            
            for c in remaining:
                if current_demand + demands[c] > capacity:
                    continue
                
                # Try all positions
                for pos in range(len(current_route) + 1):
                    new_seq = current_route[:pos] + [c] + current_route[pos:]
                    feas, d_new = is_feasible(new_seq)
                    if feas:
                        # We need d_old to calculate delta.
                        # To avoid recomputing d_old, we can just pick the first feasible one.
                        # Or we can compute d_old once.
                        # Let's just pick the first feasible one for simplicity in construction.
                        # This is fast and good enough for initialization.
                        best_c = c
                        best_pos = pos
                        added = True
                        break
                if added:
                    break
            
            if added:
                current_route.insert(best_pos, best_c)
                remaining.remove(best_c)
        
        routes.append(current_route)

    # 2. Local Search
    def get_total_distance(routes):
        total = 0
        for r in routes:
            if not r: continue
            feas, d = is_feasible(r)
            if feas:
                total += d
            else:
                return float('inf')
        return total

    current_dist = get_total_distance(routes)
    
    improved = True
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. 2-opt (intra-route)
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_seq = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feas, d_new = is_feasible(new_seq)
                    if feas:
                        _, d_old = is_feasible(route)
                        if d_new < d_old - 1e-6:
                            routes[r_idx] = new_seq
                            current_dist = get_total_distance(routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2. Or-opt (Move 1, 2, 3)
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i+length]
                    # Remove segment
                    temp_route = route[:i] + route[i+length:]
                    
                    # Try inserting into any route
                    best_target = None
                    best_pos = None
                    best_delta = float('inf')
                    
                    for t_idx, t_route in enumerate(routes):
                        target_route = temp_route if t_idx == r_idx else t_route
                        
                        # Check capacity
                        if t_idx == r_idx:
                            new_demand = get_demand(target_route)
                        else:
                            new_demand = get_demand(target_route) + sum(demands[c] for c in segment if c not in stations)
                        
                        if new_demand > capacity:
                            continue
                        
                        # Try inserting
                        for pos in range(len(target_route) + 1):
                            new_seq = target_route[:pos] + segment + target_route[pos:]
                            feas, d_new = is_feasible(new_seq)
                            if feas:
                                # Calculate delta
                                if t_idx == r_idx:
                                    _, d_old = is_feasible(route)
                                    delta = d_new - d_old
                                else:
                                    _, d_old_src = is_feasible(route)
                                    _, d_old_tgt = is_feasible(t_route)
                                    _, d_new_src = is_feasible(temp_route)
                                    delta = (d_new_src + d_new) - (d_old_src + d_old_tgt)
                                    
                                if delta < best_delta:
                                    best_delta = delta
                                    best_target = t_idx
                                    best_pos = pos
                    
                    if best_target is not None and best_delta < -1e-6:
                        # Perform move
                        new_routes = [r[:] for r in routes]
                        if best_target == r_idx:
                            new_routes[r_idx] = temp_route
                        else:
                            new_routes[r_idx] = temp_route
                        
                        new_routes[best_target].insert(best_pos, *segment)
                        
                        # Verify
                        valid = True
                        new_total = 0
                        for nr in new_routes:
                            feas, d = is_feasible(nr)
                            if not feas:
                                valid = False
                                break
                            new_total += d
                        
                        if valid and new_total < current_dist:
                            routes = new_routes
                            current_dist = new_total
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 3. Swap
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                route1 = routes[r1_idx]
                route2 = routes[r2_idx]
                
                for i in range(len(route1)):
                    c1 = route1[i]
                    if c1 in stations: continue
                    for j in range(len(route2)):
                        c2 = route2[j]
                        if c2 in stations: continue
                        
                        # Check capacity
                        if get_demand(route1) - demands[c1] + demands[c2] > capacity:
                            continue
                        if get_demand(route2) - demands[c2] + demands[c1] > capacity:
                            continue
                        
                        new_route1 = route1[:i] + [c2] + route1[i+1:]
                        new_route2 = route2[:j] + [c1] + route2[j+1:]
                        
                        feas1, d1 = is_feasible(new_route1)
                        if not feas1: continue
                        feas2, d2 = is_feasible(new_route2)
                        if not feas2: continue
                        
                        _, old_d1 = is_feasible(route1)
                        _, old_d2 = is_feasible(route2)
                        
                        if (d1 + d2) < (old_d1 + old_d2) - 1e-6:
                            routes[r1_idx] = new_route1
                            routes[r2_idx] = new_route2
                            current_dist = get_total_distance(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes
