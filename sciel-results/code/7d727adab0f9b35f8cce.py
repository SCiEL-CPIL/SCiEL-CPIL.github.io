import numpy as np
import math

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
                return False, total_dist, None
                
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
                    return False, total_dist, None
                
                t += s
        
        d = dist[loc, 0]
        b -= d
        t += d
        total_dist += d
        
        if b < 0 or t > horizon:
            return False, total_dist, None
            
        return True, total_dist, None

    def get_demand(route):
        return sum(demands[c] for c in route if c not in stations)

    # 1. Initialization: Sweep Algorithm
    # Calculate angles from depot
    angles = []
    for c in customers:
        dx = coords[c][0] - coords[0][0]
        dy = coords[c][1] - coords[0][1]
        ang = math.atan2(dy, dx)
        angles.append((ang, c))
    
    angles.sort()
    sorted_customers = [c for _, c in angles]
    
    routes = []
    remaining = set(sorted_customers)
    
    while remaining:
        head = min(remaining, key=lambda c: (math.atan2(coords[c][1]-coords[0][1], coords[c][0]-coords[0][0]), c))
        remaining.remove(head)
        route = [head]
        current_demand = demands[head]
        
        candidates = sorted(remaining, key=lambda c: math.atan2(coords[c][1]-coords[0][1], coords[c][0]-coords[0][0]))
        
        for c in candidates:
            if c not in remaining:
                continue
            if current_demand + demands[c] > capacity:
                continue
            
            best_pos = -1
            best_dist = float('inf')
            
            for pos in range(len(route) + 1):
                new_seq = route[:pos] + [c] + route[pos:]
                feas, d, _ = is_feasible(new_seq)
                if feas:
                    if d < best_dist:
                        best_dist = d
                        best_pos = pos
            
            if best_pos != -1:
                route.insert(best_pos, c)
                remaining.remove(c)
                current_demand += demands[c]
        
        routes.append(route)

    # 2. Local Search
    def get_total_distance(routes):
        total = 0
        for r in routes:
            if not r: continue
            feas, d, _ = is_feasible(r)
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
        
        # Or-opt
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            for seg_len in [1, 2, 3]:
                if len(route) < seg_len:
                    continue
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    if any(node in stations for node in segment):
                        continue
                    
                    temp_route = route[:i] + route[i+seg_len:]
                    
                    best_delta = float('inf')
                    best_target = None
                    best_pos = None
                    
                    for t_idx, t_route in enumerate(routes):
                        target_route = temp_route if t_idx == r_idx else t_route
                        
                        if t_idx == r_idx:
                            new_demand = get_demand(target_route)
                        else:
                            new_demand = get_demand(target_route) + sum(demands[node] for node in segment)
                        
                        if new_demand > capacity:
                            continue
                        
                        for pos in range(len(target_route) + 1):
                            new_seq = target_route[:pos] + segment + target_route[pos:]
                            feas, d_new, _ = is_feasible(new_seq)
                            if feas:
                                if t_idx == r_idx:
                                    _, d_old_src, _ = is_feasible(route)
                                    delta = d_new - d_old_src
                                else:
                                    _, d_old_src, _ = is_feasible(route)
                                    _, d_old_tgt, _ = is_feasible(t_route)
                                    _, d_new_src, _ = is_feasible(temp_route)
                                    delta = (d_new_src + d_new) - (d_old_src + d_old_tgt)
                                    
                                if delta < best_delta:
                                    best_delta = delta
                                    best_target = t_idx
                                    best_pos = pos
                    
                    if best_target is not None and best_delta < 0:
                        new_routes = [r[:] for r in routes]
                        if best_target == r_idx:
                            new_routes[r_idx] = temp_route
                        else:
                            new_routes[r_idx] = temp_route
                        
                        new_routes[best_target][best_pos:best_pos] = segment
                        
                        valid = True
                        new_total = 0
                        for nr in new_routes:
                            feas, d, _ = is_feasible(nr)
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

        # 2-star
        for r1 in range(len(routes)):
            for r2 in range(r1 + 1, len(routes)):
                for i in range(len(routes[r1])):
                    n1 = routes[r1][i]
                    if n1 in stations: continue
                    for j in range(len(routes[r2])):
                        n2 = routes[r2][j]
                        if n2 in stations: continue
                        
                        new_r1 = routes[r1][:]
                        new_r2 = routes[r2][:]
                        new_r1[i] = n2
                        new_r2[j] = n1
                        
                        if get_demand(new_r1) > capacity or get_demand(new_r2) > capacity:
                            continue
                        
                        feas1, d1, _ = is_feasible(new_r1)
                        feas2, d2, _ = is_feasible(new_r2)
                        
                        if feas1 and feas2:
                            old_d1, _, _ = is_feasible(routes[r1])
                            old_d2, _, _ = is_feasible(routes[r2])
                            
                            if d1 + d2 < old_d1 + old_d2:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                current_dist = get_total_distance(routes)
                                improved = True
                                break
                    if improved: break
                if improved: break
            if improved: break
        
        if improved:
            continue

        # Re-insertion
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            for i in range(len(route)):
                node = route[i]
                if node in stations:
                    continue
                
                temp_route = route[:i] + route[i+1:]
                
                best_delta = float('inf')
                best_target = None
                best_pos = None
                
                for t_idx, t_route in enumerate(routes):
                    target_route = temp_route if t_idx == r_idx else t_route
                    
                    if t_idx == r_idx:
                        new_demand = get_demand(target_route)
                    else:
                        new_demand = get_demand(target_route) + demands[node]
                    
                    if new_demand > capacity:
                        continue
                    
                    for pos in range(len(target_route) + 1):
                        new_seq = target_route[:pos] + [node] + target_route[pos:]
                        feas, d_new, _ = is_feasible(new_seq)
                        if feas:
                            if t_idx == r_idx:
                                _, d_old_src, _ = is_feasible(route)
                                delta = d_new - d_old_src
                            else:
                                _, d_old_src, _ = is_feasible(route)
                                _, d_old_tgt, _ = is_feasible(t_route)
                                _, d_new_src, _ = is_feasible(temp_route)
                                delta = (d_new_src + d_new) - (d_old_src + d_old_tgt)
                                
                            if delta < best_delta:
                                best_delta = delta
                                best_target = t_idx
                                best_pos = pos
                
                if best_target is not None and best_delta < 0:
                    new_routes = [r[:] for r in routes]
                    new_routes[r_idx] = temp_route
                    new_routes[best_target].insert(best_pos, node)
                    
                    valid = True
                    new_total = 0
                    for nr in new_routes:
                        feas, d, _ = is_feasible(nr)
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

    return routes
