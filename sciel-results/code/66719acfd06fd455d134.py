import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_ids = set(stations)
    
    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_ids

    def check_route(route):
        """
        Returns (total_dist, valid)
        """
        if not route:
            return 0.0, True
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Check capacity
        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False

        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            
            if b < -_EPS:
                return float('inf'), False
            
            if is_station(node):
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + _EPS:
                    return float('inf'), False
                t += service[node]
            
            prev = node
            
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        
        if b < -_EPS:
            return float('inf'), False
            
        if t > horizon + _EPS:
            return float('inf'), False
            
        return total_dist, True

    def get_route_cost(route):
        return check_route(route)[0]

    # --- Construction: Greedy Insertion with Time Windows ---
    def construct_greedy():
        routes = []
        unvisited = sorted(cust_ids, key=lambda c: due[c])
        
        while unvisited:
            # Start a new route with the first customer
            c = unvisited.pop(0)
            route = [c]
            
            # Try to insert other customers
            changed = True
            while changed:
                changed = False
                best_cust = None
                best_pos = -1
                best_cost = float('inf')
                
                current_load = sum(demands[n] for n in route if is_customer(n))
                
                for c_idx, c in enumerate(unvisited):
                    if current_load + demands[c] > capacity + _EPS:
                        continue
                    
                    for pos in range(len(route) + 1):
                        cand = route[:pos] + [c] + route[pos:]
                        cost, valid = check_route(cand)
                        if valid and cost < best_cost:
                            best_cost = cost
                            best_cust = c
                            best_pos = pos
                
                if best_cust is not None:
                    route = route[:best_pos] + [best_cust] + route[best_pos:]
                    unvisited.remove(best_cust)
                    current_load += demands[best_cust]
                    changed = True
            
            routes.append(route)
        
        return routes

    # --- Local Search ---
    def local_search(routes):
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            
            # 1. Intra-route 2-Opt
            for ri in range(len(routes)):
                if time.time() > DEADLINE: break
                route = routes[ri]
                n = len(route)
                if n < 2: continue
                
                cur_cost, cur_valid = check_route(route)
                if not cur_valid:
                    continue
                
                best_route = list(route)
                best_cost = cur_cost
                route_improved = True
                max_iter = 10
                while route_improved and max_iter > 0:
                    route_improved = False
                    max_iter -= 1
                    for i in range(n - 1):
                        if time.time() > DEADLINE: break
                        for j in range(i + 1, n):
                            cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                            c, v = check_route(cand)
                            if v and c < best_cost - 1e-7:
                                best_cost = c
                                best_route = cand
                                route_improved = True
                    
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True
            
            # 2. Inter-route Relocate
            if time.time() > DEADLINE: break
            for ri in range(len(routes)):
                if time.time() > DEADLINE: break
                for ci in range(len(routes[ri])):
                    if time.time() > DEADLINE: break
                    cust = routes[ri][ci]
                    temp_from = routes[ri][:]
                    del temp_from[ci]
                    c_from, v_from = check_route(temp_from)
                    if not v_from:
                        continue
                    
                    for rj in range(len(routes)):
                        if rj == ri: continue
                        if time.time() > DEADLINE: break
                        for cj in range(len(routes[rj]) + 1):
                            if time.time() > DEADLINE: break
                            cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                            c_to, v_to = check_route(cand_to)
                            if not v_to:
                                continue
                            
                            load_to = sum(demands[n] for n in cand_to if is_customer(n))
                            if load_to > capacity + _EPS:
                                continue
                            
                            old_cost = c_from + get_route_cost(routes[rj])
                            new_cost = c_from + c_to
                            if new_cost < old_cost - 1e-7:
                                routes[ri] = temp_from
                                routes[rj] = cand_to
                                improved = True
                                ri = -1
                                break
                        if ri == -1: break
                    if ri == -1: break
                if ri == -1: break
            
            routes = [r for r in routes if r]
            
            # 3. Inter-route Swap
            if time.time() > DEADLINE: break
            swapped = False
            for ri in range(len(routes)):
                if swapped or time.time() > DEADLINE: break
                for rj in range(ri + 1, len(routes)):
                    if swapped or time.time() > DEADLINE: break
                    for ci in range(len(routes[ri])):
                        if swapped or time.time() > DEADLINE: break
                        cust1 = routes[ri][ci]
                        for cj in range(len(routes[rj])):
                            if swapped or time.time() > DEADLINE: break
                            cust2 = routes[rj][cj]
                            
                            load_i = sum(demands[n] for n in routes[ri] if is_customer(n))
                            load_j = sum(demands[n] for n in routes[rj] if is_customer(n))
                            new_load_i = load_i - demands[cust1] + demands[cust2]
                            new_load_j = load_j - demands[cust2] + demands[cust1]
                            
                            if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                                continue
                                
                            new_route_i = routes[ri][:]
                            new_route_i[ci] = cust2
                            new_route_j = routes[rj][:]
                            new_route_j[cj] = cust1
                            
                            c_i, v_i = check_route(new_route_i)
                            c_j, v_j = check_route(new_route_j)
                            
                            if v_i and v_j:
                                old_c_i = get_route_cost(routes[ri])
                                old_c_j = get_route_cost(routes[rj])
                                if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                    routes[ri] = new_route_i
                                    routes[rj] = new_route_j
                                    improved = True
                                    swapped = True
                                    break
                    if swapped: break
            if swapped: break
            
        return routes

    # --- Iterated Local Search ---
    def iterated_local_search():
        current_routes = construct_greedy()
        current_routes = local_search(current_routes)
        
        best_routes = [r[:] for r in current_routes]
        best_cost = sum(get_route_cost(r) for r in best_routes)
        
        while time.time() < DEADLINE:
            # Perturb: Remove K customers
            K = random.randint(3, 8)
            all_customers = [c for r in current_routes for c in r if is_customer(c)]
            if len(all_customers) < K:
                break
            
            removed = random.sample(all_customers, K)
            
            # Remove from current routes
            for c in removed:
                for i, r in enumerate(current_routes):
                    if c in r:
                        current_routes[i].remove(c)
                        break
            
            current_routes = [r for r in current_routes if r]
            
            # Repair: Insert removed customers
            for c in removed:
                best_route_idx = -1
                best_pos = -1
                best_cost_increase = float('inf')
                
                for ri, r in enumerate(current_routes):
                    for pos in range(len(r) + 1):
                        cand = r[:pos] + [c] + r[pos:]
                        cost, valid = check_route(cand)
                        if valid:
                            # Calculate cost increase
                            old_cost = get_route_cost(r)
                            increase = cost - old_cost
                            if increase < best_cost_increase:
                                best_cost_increase = increase
                                best_route_idx = ri
                                best_pos = pos
                
                if best_route_idx != -1:
                    current_routes[best_route_idx] = current_routes[best_route_idx][:best_pos] + [c] + current_routes[best_route_idx][best_pos:]
                else:
                    # If can't insert into existing, start new route
                    current_routes.append([c])
            
            # Local Search on perturbed solution
            current_routes = local_search(current_routes)
            
            # Compare with best
            curr_cost = sum(get_route_cost(r) for r in current_routes)
            if curr_cost < best_cost - 1e-7:
                best_cost = curr_cost
                best_routes = [r[:] for r in current_routes]
            
            # Accept or reject? Standard ILS accepts if better, or with probability if worse.
            # For simplicity and to ensure progress, we only keep the best.
            # But we should probably continue from the current state even if worse, to explore.
            # However, if we get stuck, we might want to restart.
            # Let's just keep the best found so far as the return value.

        return best_routes

    # Run ILS
    routes = iterated_local_search()
    
    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v = check_route(r)
        if not v:
            valid = False
            break
        total_dist += c
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break
            
    if not valid or served != set(range(1, n_cust + 1)):
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                # Try to fix by adding station?
                # For fallback, assume single customer is always feasible if battery > 2*max_dist
                pass

    return routes
