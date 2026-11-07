"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _dist(a, b, coords):
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    return math.sqrt(dx*dx + dy*dy)

def _euclid(a, b, coords):
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    return math.sqrt(dx*dx + dy*dy)

def _build_dist(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = math.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
            D[i,j] = d
            D[j,i] = d
    return D

def _check_route(route, instance, coords, D):
    """
    Check if a route (list of node ids, no depot) is feasible.
    Returns (feasible, total_distance, end_time)
    """
    cap = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    
    total_dist = 0.0
    total_demand = 0
    battery = battery_max
    time_t = 0.0
    
    prev = 0
    for node in route:
        d = D[prev, node]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return False, total_dist, time_t
        time_t += d / speed
        
        if node in stations:
            # Charging: time added = (battery_max - battery) * recharge_rate
            charge_time = (battery_max - battery) * recharge_rate
            time_t += charge_time
            battery = battery_max
        else:
            # Customer
            total_demand += demands[node]
            if total_demand > cap + 1e-9:
                return False, total_dist, time_t
            time_t = max(time_t, ready[node])
            if time_t > due[node] + 1e-9:
                return False, total_dist, time_t
            time_t += service[node]
        
        prev = node
    
    # Return to depot
    d = D[prev, 0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return False, total_dist, time_t
    time_t += d / speed
    if time_t > horizon + 1e-9:
        return False, total_dist, time_t
    
    return True, total_dist, time_t

def _clarke_wright(instance, coords, D):
    """
    Clarke-Wright savings heuristic to build initial routes.
    Returns list of routes (each a list of customer ids, no depot).
    """
    customers = list(range(1, instance["n_customers"] + 1))
    n_cust = len(customers)
    
    # Compute savings
    savings = []
    for i in range(n_cust):
        for j in range(i+1, n_cust):
            ci = customers[i]
            cj = customers[j]
            s = D[0, ci] + D[0, cj] - D[ci, cj]
            savings.append((s, ci, cj))
    savings.sort(reverse=True)
    
    # Initialize: each customer is its own route
    routes = {c: [c] for c in customers}
    route_of = {c: c for c in customers}  # maps customer to route id (the customer itself initially)
    # We'll use a dict: route_id -> list of nodes in order
    # Actually, let's use a different approach: maintain a list of routes, each is a list
    # and a mapping from node to (route_index, position)
    
    route_list = [[c] for c in customers]
    node_to_route = {c: i for i, c in enumerate(customers)}
    
    def find_route(node):
        return node_to_route[node]
    
    def get_route(ridx):
        return route_list[ridx]
    
    for s, ci, cj in savings:
        if s <= 0:
            break
        ri = find_route(ci)
        rj = find_route(cj)
        if ri == rj:
            continue
        
        # Check if ci and cj are at the ends of their routes
        route_i = route_list[ri]
        route_j = route_list[rj]
        
        ci_at_end = (ci == route_i[0] or ci == route_i[-1])
        cj_at_end = (cj == route_j[0] or cj == route_j[-1])
        
        if not (ci_at_end and cj_at_end):
            continue
        
        # Determine orientation
        # We want to merge so that ci and cj are adjacent
        # Case 1: ci is at start of route_i, cj is at end of route_j -> merge route_i + route_j
        # Case 2: ci is at end of route_i, cj is at start of route_j -> merge route_i + reversed(route_j)
        # Case 3: ci is at start of route_i, cj is at start of route_j -> merge route_i + reversed(route_j)
        # Case 4: ci is at end of route_i, cj is at end of route_j -> merge reversed(route_i) + route_j
        
        # Let's figure out the new combined route
        if ci == route_i[0]:
            ri_dir = 1  # route_i goes from ci outward
        else:
            ri_dir = -1  # route_i ends at ci
        
        if cj == route_j[0]:
            rj_dir = 1
        else:
            rj_dir = -1
        
        # We need to concatenate so that ci and cj are adjacent
        # The combined route should start with the "outer" end of one and end with the "outer" end of the other
        
        # If ri_dir == 1, route_i starts with ci. If ri_dir == -1, route_i ends with ci.
        # Similarly for rj.
        
        # We want to merge: [outer_i ... ci] + [cj ... outer_j]
        # So if ri_dir == 1: left part is route_i as is (starts with ci)
        #    if ri_dir == -1: left part is reversed(route_i) (ends with ci, so reversed starts with ... ci at end? No)
        
        # Let me think differently.
        # route_i is a list. ci is at position 0 or -1.
        # If ci at 0: the "outer" end is route_i[-1]
        # If ci at -1: the "outer" end is route_i[0]
        
        # Similarly for route_j.
        
        # We want to form: [outer_i, ..., ci, cj, ..., outer_j]
        
        if ci == route_i[0]:
            # ci is at front, so route_i = [ci, ..., outer_i]
            # We want outer_i at the far end, so we keep route_i as is but it will be on the "left"
            # Actually, we want: [outer_i, ..., ci, cj, ..., outer_j]
            # So left part should be reversed(route_i) if we want outer_i first? No.
            # route_i = [ci, a, b, outer_i]
            # We want: [outer_i, b, a, ci, cj, ..., outer_j]
            # So left part = reversed(route_i)
            left = route_i[::-1]
        else:
            # ci is at end: route_i = [outer_i, ..., ci]
            # We want: [outer_i, ..., ci, cj, ..., outer_j]
            # So left part = route_i as is
            left = route_i[:]
        
        if cj == route_j[-1]:
            # cj is at end: route_j = [outer_j, ..., cj]
            # We want: [outer_j, ..., cj] but cj should be adjacent to ci, so cj should be at the start of the right part
            # So right part = reversed(route_j)
            right = route_j[::-1]
        else:
            # cj is at start: route_j = [cj, ..., outer_j]
            # We want: [cj, ..., outer_j]
            # So right part = route_j as is
            right = route_j[:]
        
        new_route = left + right
        
        # Check capacity
        total_demand = sum(instance["demands"][n] for n in new_route)
        if total_demand > instance["capacity"]:
            continue
        
        # Check battery feasibility (rough check - we'll do full check later)
        # For now, skip and rely on local search to fix
        
        # Update data structures
        new_rix = min(ri, rj)
        old_rix = max(ri, rj)
        
        route_list[new_rix] = new_route
        route_list[old_rix] = []
        
        for node in new_route:
            node_to_route[node] = new_rix
        
        # Remove empty route reference
        # We'll filter out empty routes at the end
    
    # Collect non-empty routes
    result = []
    for r in route_list:
        if r:
            result.append(r)
    
    return result

def _try_insert(routes, instance, coords, D, customer, deadline=None):
    """
    Try to insert customer into existing routes.
    Returns (success, new_routes)
    """
    best_gain = -1
    best_route_idx = -1
    best_pos = -1
    
    for ri, route in enumerate(routes):
        if not route:
            continue
        # Try inserting at each position
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [customer] + route[pos:]
            
            # Quick capacity check
            total_demand = sum(instance["demands"][n] for n in new_route)
            if total_demand > instance["capacity"]:
                continue
            
            # Check feasibility
            feasible, dist, _ = _check_route(new_route, instance, coords, D)
            if not feasible:
                continue
            
            # Compute gain: reduction in distance
            # Old distance contribution around insertion point
            if pos == 0:
                old_d = D[0, route[0]]
                new_d = D[0, customer] + D[customer, route[0]]
            elif pos == len(route):
                old_d = D[route[-1], 0]
                new_d = D[route[-1], customer] + D[customer, 0]
            else:
                old_d = D[route[pos-1], route[pos]]
                new_d = D[route[pos-1], customer] + D[customer, route[pos]]
            
            gain = old_d - new_d
            if gain > best_gain:
                best_gain = gain
                best_route_idx = ri
                best_pos = pos
    
    if best_route_idx >= 0:
        route = routes[best_route_idx]
        new_route = route[:best_pos] + [customer] + route[best_pos:]
        new_routes = routes[:]
        new_routes[best_route_idx] = new_route
        return True, new_routes
    
    return False, routes

def _two_opt(route, instance, coords, D, max_iters=100):
    """
    Apply 2-opt to a single route.
    """
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Consider reversing route[i:j+1]
                # But we need to be careful with the depot connections
                # 2-opt for a route without depot: reverse segment
                # The connections affected are:
                # (route[i-1], route[i]) and (route[j], route[j+1]) if they exist
                # and (route[i], route[i+1]) and (route[j-1], route[j])
                
                # Let's define the full sequence with depot
                # Actually, let's work with the route as is and consider the depot implicitly
                
                # If i == 0, the left connection is depot->route[0]
                # If j == n-1, the right connection is route[n-1]->depot
                
                if i == 0:
                    left_node = 0
                else:
                    left_node = route[i-1]
                
                if j == n - 1:
                    right_node = 0
                else:
                    right_node = route[j+1]
                
                # Current edges:
                # (left_node, route[i]) and (route[j], right_node)
                # New edges after reversing route[i:j+1]:
                # (left_node, route[j]) and (route[i], right_node)
                
                old_d = D[left_node, route[i]] + D[route[j], right_node]
                new_d = D[left_node, route[j]] + D[route[i], right_node]
                
                if new_d < old_d - 1e-9:
                    # Check feasibility of the modified route
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _, _ = _check_route(new_route, instance, coords, D)
                    if feasible:
                        route = new_route
                        improved = True
    
    return route

def _remove_stations(route, instance, coords, D):
    """
    Try to remove unnecessary charging stations from a route.
    A station can be removed if the route is still feasible without it.
    """
    stations_in_route = [i for i, n in enumerate(route) if n in instance["stations"]]
    for idx in reversed(stations_in_route):
        new_route = route[:idx] + route[idx+1:]
        if not new_route:
            continue
        feasible, _, _ = _check_route(new_route, instance, coords, D)
        if feasible:
            route = new_route
    return route

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    coords = np.array(instance["coords"], dtype=float)
    D = _build_dist(coords)
    
    # Build initial solution using Clarke-Wright
    routes = _clarke_wright(instance, coords, D)
    
    # Ensure all customers are covered
    covered = set()
    for r in routes:
        covered.update(r)
    missing = [c for c in range(1, instance["n_customers"]+1) if c not in covered]
    
    # Insert missing customers
    for c in missing:
        success, routes = _try_insert(routes, instance, coords, D, c)
        if not success:
            routes.append([c])
    
    # Local search: 2-opt on each route, remove stations
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            # Try 2-opt
            new_route = _two_opt(routes[i], instance, coords, D, max_iters=50)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True
            # Remove stations
            new_route = _remove_stations(routes[i], instance, coords, D)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True
    
    # Try or-opt: move a single node to a different position in another route
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(len(routes)):
                if time.time() > deadline:
                    break
                for pos in range(len(routes[ri])):
                    if time.time() > deadline:
                        break
                    node = routes[ri][pos]
                    # Remove node
                    remaining = routes[ri][:pos] + routes[ri][pos+1:]
                    # Try inserting into other routes
                    for rj in range(len(routes)):
                        if rj == ri:
                            continue
                        if not routes[rj] and not remaining:
                            continue
                        # Try all positions in route rj
                        for pos2 in range(len(routes[rj]) + 1):
                            new_route_j = routes[rj][:pos2] + [node] + routes[rj][pos2:]
                            # Check feasibility of both routes
                            feas_j, dist_j, _ = _check_route(new_route_j, instance, coords, D)
                            if not feas_j:
                                continue
                            if remaining:
                                feas_i, dist_i, _ = _check_route(remaining, instance, coords, D)
                            else:
                                feas_i = True
                                dist_i = 0.0
                            if not feas_i:
                                continue
                            
                            # Compute total distance change
                            old_total = 0
                            for r in routes:
                                if r:
                                    _, d, _ = _check_route(r, instance, coords, D)
                                    old_total += d
                            
                            new_total = dist_i + dist_j
                            for r in routes:
                                if r == routes[ri] or r == routes[rj]:
                                    continue
                                if r:
                                    _, d, _ = _check_route(r, instance, coords, D)
                                    new_total += d
                            
                            if new_total < old_total - 1e-9:
                                new_routes = routes[:]
                                if remaining:
                                    new_routes[ri] = remaining
                                else:
                                    new_routes[ri] = []
                                new_routes[rj] = new_route_j
                                routes = [r for r in new_routes if r]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    # Final: ensure all customers are served
    covered = set()
    for r in routes:
        covered.update(r)
    missing = [c for c in range(1, instance["n_customers"]+1) if c not in covered]
    for c in missing:
        routes.append([c])
    
    return routes
# EVOLVE-BLOCK-END
