# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _cluster_by_angle(coords, n_customers, n_clusters=4):
    """Cluster customers by angle from depot - creates angular sectors."""
    customers = list(range(1, n_customers + 1))
    depot = np.array(coords[0], dtype=float)
    
    # Compute angles from depot
    angles = []
    for c in customers:
        cpos = np.array(coords[c], dtype=float)
        angle = math.atan2(cpos[1] - depot[1], cpos[0] - depot[0])
        # Normalize to [0, 2*pi)
        if angle < 0:
            angle += 2 * math.pi
        angles.append((angle, c))
    
    # Sort by angle
    angles.sort(key=lambda x: x[0])
    
    # Split into n_clusters groups
    clusters = []
    group_size = len(angles) // n_clusters
    remainder = len(angles) % n_clusters
    
    start = 0
    for i in range(n_clusters):
        size = group_size + (1 if i < remainder else 0)
        if size > 0:
            cluster = [angles[j][1] for j in range(start, start + size)]
            clusters.append(cluster)
            start += size
    
    return clusters


def _nearest_neighbor_route(start_node, unvisited, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Build a route using nearest neighbor heuristic with feasibility checking."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    route = [start_node]
    current_demand = demands[start_node]
    
    while unvisited:
        best_node = None
        best_dist = float('inf')
        
        for c in unvisited:
            if current_demand + demands[c] > capacity:
                continue
            
            # Quick feasibility check: can we reach c and return to depot?
            d_to_c = dist[route[-1]][c]
            d_back = dist[c][0]
            
            # Battery check (rough)
            # Time window check (rough)
            if d_to_c + d_back > battery_max:
                continue
            
            # Try to insert c at the end
            test_route = route + [c]
            feasible, _, _ = _check_route(test_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                if d_to_c < best_dist:
                    best_dist = d_to_c
                    best_node = c
        
        if best_node is None:
            break
        
        route.append(best_node)
        current_demand += demands[best_node]
        unvisited.remove(best_node)
    
    return route, unvisited


def _construct_initial_routes(instance, dist, time_limit=5):
    """Construct initial routes using angle-based clustering + nearest neighbor."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Cluster by angle from depot
    clusters = _cluster_by_angle(coords, n_customers, n_clusters=4)
    
    routes = []
    unvisited = set(range(1, n_customers + 1))
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        
        # Sort cluster by due date for better time window handling
        cluster_sorted = sorted(cluster, key=lambda c: due[c])
        
        if not cluster_sorted:
            continue
        
        # Start with first customer in cluster
        start_node = cluster_sorted[0]
        unvisited.remove(start_node)
        
        # Build route using nearest neighbor
        route, remaining_in_cluster = _nearest_neighbor_route(
            start_node, 
            [c for c in cluster_sorted if c != start_node],
            dist, instance, battery_max, capacity, speed, recharge_rate, horizon
        )
        
        # Add any remaining customers from this cluster that weren't visited
        for c in remaining_in_cluster:
            if c in unvisited:
                unvisited.remove(c)
                # Try to insert at best position
                best_pos = -1
                best_cost = float('inf')
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible and d < best_cost:
                        best_cost = d
                        best_pos = pos
                if best_pos != -1:
                    route = route[:best_pos] + [c] + route[best_pos:]
        
        if route:
            routes.append(route)
    
    # Handle any remaining unvisited customers
    for c in list(unvisited):
        if time.time() > deadline:
            break
        
        # Try to insert into existing routes
        inserted = False
        for i, route in enumerate(routes):
            best_pos = -1
            best_cost = float('inf')
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible and d < best_cost:
                    best_cost = d
                    best_pos = pos
            if best_pos != -1:
                routes[i] = route[:best_pos] + [c] + route[best_pos:]
                unvisited.remove(c)
                inserted = True
                break
        
        if not inserted:
            routes.append([c])
            unvisited.remove(c)
    
    return routes


def _local_search(routes, dist, instance, time_limit=20):
    """Perform local search with Relocate, Swap, Or-Opt, and 2-Opt."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    # Precompute route distances and feasibility
    route_states = []
    for r in routes:
        feasible, d, t = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        route_states.append({
            'route': r,
            'feasible': feasible,
            'dist': d if feasible else float('inf'),
            'time': t
        })
    
    total_dist = sum(s['dist'] for s in route_states)
    
    # Helper to update route state
    def update_state(idx, new_route):
        feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        route_states[idx]['route'] = new_route
        route_states[idx]['feasible'] = feasible
        route_states[idx]['dist'] = d if feasible else float('inf')
        route_states[idx]['time'] = t
        return feasible, d, t
    
    # Remove empty routes
    active_indices = [i for i, s in enumerate(route_states) if s['route']]
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 1 and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Relocate: Move one customer from route i to route j
        for i in active_indices:
            if len(route_states[i]['route']) < 2:
                continue
            for pos in range(len(route_states[i]['route'])):
                c = route_states[i]['route'][pos]
                if c > n_customers:
                    continue
                
                # Remove from i
                new_route_i = route_states[i]['route'][:pos] + route_states[i]['route'][pos+1:]
                if not new_route_i:
                    continue
                feas_i, d_i, t_i = _check_route(new_route_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if not feas_i:
                    continue
                
                # Try to insert into j
                for j in active_indices:
                    if i == j:
                        continue
                    rj = route_states[j]['route']
                    for pos_j in range(len(rj) + 1):
                        new_route_j = rj[:pos_j] + [c] + rj[pos_j:]
                        feas_j, d_j, t_j = _check_route(new_route_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if not feas_j:
                            continue
                        
                        old_total = route_states[i]['dist'] + route_states[j]['dist']
                        new_total = d_i + d_j
                        
                        if new_total < old_total - 1e-9:
                            route_states[i]['route'] = new_route_i
                            route_states[i]['dist'] = d_i
                            route_states[j]['route'] = new_route_j
                            route_states[j]['dist'] = d_j
                            total_dist = new_total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            active_indices = [i for i, s in enumerate(route_states) if s['route']]
            no_improve_count = 0
            continue
        
        # 2. Or-Opt: Move 1-3 consecutive nodes within a route
        for i in active_indices:
            r = route_states[i]['route']
            n = len(r)
            if n < 2:
                continue
            
            for L in [1, 2, 3]:
                if n < L + 1:
                    continue
                for start in range(n - L):
                    moving = r[start:start+L]
                    rest = r[:start] + r[start+L:]
                    
                    for pos in range(len(rest) + 1):
                        new_route = rest[:pos] + moving + rest[pos:]
                        if new_route == r:
                            continue
                        
                        feas, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if feas and d < route_states[i]['dist'] - 1e-9:
                            route_states[i]['route'] = new_route
                            route_states[i]['dist'] = d
                            total_dist = sum(s['dist'] for s in route_states)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve_count = 0
            continue
            
        # 3. 2-Opt within a route
        for i in active_indices:
            r = route_states[i]['route']
            n = len(r)
            if n < 2:
                continue
            
            found_improve = False
            for a in range(n - 1):
                if found_improve:
                    break
                for b in range(a + 1, n):
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    if new_r == r:
                        continue
                    
                    feas, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feas and d < route_states[i]['dist'] - 1e-9:
                        route_states[i]['route'] = new_r
                        route_states[i]['dist'] = d
                        total_dist = sum(s['dist'] for s in route_states)
                        improved = True
                        found_improve = True
                        break
            if improved:
                break
        
        if improved:
            no_improve_count = 0
            continue
            
        # 4. Cross exchange between routes
        if len(active_indices) >= 2:
            for i in active_indices:
                for j in active_indices:
                    if i >= j:
                        continue
                    ri = route_states[i]['route']
                    rj = route_states[j]['route']
                    
                    # Try swapping suffixes
                    for pi in range(1, len(ri)):
                        for pj in range(1, len(rj)):
                            # New route i: ri[:pi] + rj[pj:]
                            # New route j: rj[:pj] + ri[pi:]
                            new_ri = ri[:pi] + rj[pj:]
                            new_rj = rj[:pj] + ri[pi:]
                            
                            if not new_ri or not new_rj:
                                continue
                            
                            feas_i, d_i, t_i = _check_route(new_ri, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if not feas_i:
                                continue
                            feas_j, d_j, t_j = _check_route(new_rj, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if not feas_j:
                                continue
                            
                            old_total = route_states[i]['dist'] + route_states[j]['dist']
                            new_total = d_i + d_j
                            
                            if new_total < old_total - 1e-9:
                                route_states[i]['route'] = new_ri
                                route_states[i]['dist'] = d_i
                                route_states[j]['route'] = new_rj
                                route_states[j]['dist'] = d_j
                                total_dist = new_total
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if improved:
            no_improve_count = 0
            continue
        
        # No improvement in this full pass
        no_improve_count += 1
        
        # Perturbation: randomly relocate 2-3 customers to escape local optimum
        if no_improve_count % 10 == 0 and len(active_indices) >= 2:
            # Pick 2-3 random customers and relocate them
            num_moves = random.randint(2, 3)
            for _ in range(num_moves):
                # Pick random route with at least 2 customers
                candidate_routes = [i for i in active_indices if len(route_states[i]['route']) >= 2]
                if not candidate_routes:
                    break
                i = random.choice(candidate_routes)
                r = route_states[i]['route']
                cust_positions = [pos for pos, node in enumerate(r) if node <= n_customers]
                if not cust_positions:
                    continue
                pos = random.choice(cust_positions)
                c = r[pos]
                
                new_route_i = r[:pos] + r[pos+1:]
                if not new_route_i:
                    continue
                
                feas_i, d_i, t_i = _check_route(new_route_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if not feas_i:
                    continue
                
                # Try to insert into another route
                other_routes = [j for j in active_indices if j != i]
                if not other_routes:
                    continue
                j = random.choice(other_routes)
                rj = route_states[j]['route']
                pos_j = random.randint(0, len(rj))
                new_route_j = rj[:pos_j] + [c] + rj[pos_j:]
                
                feas_j, d_j, t_j = _check_route(new_route_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feas_j:
                    route_states[i]['route'] = new_route_i
                    route_states[i]['dist'] = d_i
                    route_states[j]['route'] = new_route_j
                    route_states[j]['dist'] = d_j
                    total_dist = d_i + d_j + sum(s['dist'] for k, s in enumerate(route_states) if k not in [i, j])
            
            active_indices = [i for i, s in enumerate(route_states) if s['route']]
            no_improve_count = 0
    
    # Return routes
    final_routes = [s['route'] for s in route_states if s['route']]
    return final_routes


def _solve_instance(instance, time_limit=25):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # 1. Construct initial routes
    construct_time = min(5, time_limit * 0.2)
    routes = _construct_initial_routes(instance, dist, construct_time)
    
    # Ensure all customers are served
    all_served = set()
    for r in routes:
        all_served.update(r)
    n_customers = instance["n_customers"]
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
        
    # 2. Local Search
    ls_time = deadline - time.time() - 2
    if ls_time > 1:
        routes = _local_search(routes, dist, instance, ls_time)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
