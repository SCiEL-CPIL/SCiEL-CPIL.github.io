# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import copy

def plan(instance):
    t0 = time.time()
    # Safety margin: evaluator kills at 150s. We aim to return by 145s.
    deadline = t0 + 145.0 

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    # Precompute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)

    # Precompute customer demands for quick access
    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    # Fast simulation function
    # Returns (total_distance, is_valid)
    def simulate(route):
        if not route:
            return 0.0, True
        
        _dist = dist
        _ready = ready
        _due = due
        _service = service
        _bat_cap = battery_cap
        _speed = speed
        _rr = recharge_rate
        _cust_set = cust_set
        _station_set = station_set
        
        t = 0.0
        bat = _bat_cap
        total_d = 0.0
        cur = 0
        
        for node in route:
            d = _dist[cur, node]
            total_d += d
            bat -= d
            t += d / _speed
            
            if bat < -1e-9:
                return total_d, False
            
            if node in _cust_set:
                if t < _ready[node]:
                    t = _ready[node]
                if t > _due[node] + 1e-9:
                    return total_d, False
                t += _service[node]
            elif node in _station_set:
                t += (_bat_cap - bat) * _rr
                bat = _bat_cap
                
            cur = node
            
        d = _dist[cur, 0]
        total_d += d
        bat -= d
        t += d / _speed
        
        if bat < -1e-9:
            return total_d, False
        if t > horizon + 1e-9:
            return total_d, False
            
        return total_d, True

    def route_demand(route):
        return sum(cust_demand[v] for v in route if v in cust_set)

    def fix_battery(route):
        """
        Inserts charging stations into the route if necessary to make it valid.
        Returns (fixed_route, distance) or (None, None) if not fixable.
        """
        d, valid = simulate(route)
        if valid:
            return route, d
        
        # Try to fix by inserting stations
        # Strategy: Find the point where battery goes negative, insert a station before it.
        # Or, try all possible insertions of each station at all positions and pick the best valid one.
        # Given small n, we can be a bit exhaustive but smart.
        
        best_route = None
        best_d = float('inf')
        n = len(route)
        
        # First pass: try inserting single stations
        for st in stations:
            for p in range(n + 1):
                cand = route[:p] + [st] + route[p:]
                cd, cvalid = simulate(cand)
                if cvalid and cd < best_d:
                    best_d = cd
                    best_route = cand
                    
        if best_route is not None:
            return best_route, best_d
            
        # Second pass: try inserting two stations (if one wasn't enough)
        # This is expensive, so we limit it.
        # We only do this if the route is short enough or if we are desperate.
        if n < 15: 
            for st1 in stations:
                for p1 in range(n + 1):
                    r1 = route[:p1] + [st1] + route[p1:]
                    # Check if r1 is valid, if so we already found it in first pass (unless st1==st2 logic, but we handle distinct positions)
                    # Actually, first pass covers all single insertions.
                    # If first pass failed, we need 2.
                    for st2 in stations:
                        for p2 in range(len(r1) + 1):
                             # Avoid inserting st2 at the same spot as st1 if st1==st2? 
                             # Actually, if st1==st2, inserting twice at same spot is redundant.
                             # We just iterate all positions.
                             cand = r1[:p2] + [st2] + r1[p2:]
                             cd, cvalid = simulate(cand)
                             if cvalid and cd < best_d:
                                 best_d = cd
                                 best_route = cand
        else:
            # Heuristic for long routes: Insert station at the point of max battery deficit
            # This is a fallback, might not be optimal but better than None
            # Find where battery is lowest
            min_bat = battery_cap
            min_idx = 0
            bat = battery_cap
            cur = 0
            for i, node in enumerate(route):
                d = dist[cur, node]
                bat -= d
                if bat < min_bat:
                    min_bat = bat
                    min_idx = i
                if node in station_set:
                    bat = battery_cap
                cur = node
            d = dist[cur, 0]
            bat -= d
            if bat < min_bat:
                min_bat = bat
                min_idx = len(route) # End of route (before return)
                
            # Try inserting station at min_idx
            st = min(stations, key=lambda s: dist[route[min_idx-1] if min_idx>0 else 0, s] + dist[s, route[min_idx] if min_idx<len(route) else 0])
            cand = route[:min_idx] + [st] + route[min_idx:]
            cd, cvalid = simulate(cand)
            if cvalid:
                return cand, cd
                
        return None, None

    def construct_savings():
        """
        Clarke-Wright savings construction, battery-aware.
        """
        # Start with individual routes for each customer
        active_routes = []
        for c in customers:
            r, d = fix_battery([c])
            if r is None:
                r = [c] # Should not happen for single customer unless it's unreachable, but assume reachable
            active_routes.append([r, demands[c]])
            
        improved = True
        while improved:
            improved = False
            best_save = 1e-6
            best_i, best_j = -1, -1
            best_new_route = None
            best_new_demand = 0
            
            n_ar = len(active_routes)
            # Calculate current distances for all routes to avoid re-calculation in inner loop if possible
            # But routes change, so we recalc on the fly or cache.
            # For n=25, n_ar is small.
            
            for i in range(n_ar):
                r_i, d_i = active_routes[i]
                d_i_val, _ = simulate(r_i)
                
                for j in range(i+1, n_ar):
                    r_j, d_j = active_routes[j]
                    
                    if d_i + d_j > capacity:
                        continue
                        
                    d_j_val, _ = simulate(r_j)
                    
                    # Try merging r_i + r_j
                    cand = r_i + r_j
                    nr, nd = fix_battery(cand)
                    if nr is not None:
                        save = d_i_val + d_j_val - nd
                        if save > best_save:
                            best_save = save
                            best_i, best_j = i, j
                            best_new_route = nr
                            best_new_demand = d_i + d_j
                            
                    # Try merging r_j + r_i
                    cand = r_j + r_i
                    nr, nd = fix_battery(cand)
                    if nr is not None:
                        save = d_i_val + d_j_val - nd
                        if save > best_save:
                            best_save = save
                            best_i, best_j = i, j
                            best_new_route = nr
                            best_new_demand = d_i + d_j
                            
            if best_i != -1:
                active_routes.pop(best_j)
                active_routes.pop(best_i)
                active_routes.append([best_new_route, best_new_demand])
                improved = True
                if time.time() > deadline - 40.0:
                    break
                    
        return [r[0] for r in active_routes]

    def two_opt(route):
        n = len(route)
        if n < 2:
            return route
        
        best_route = route
        best_d, _ = simulate(route)
        
        improved = True
        it = 0
        while improved and it < 20: # Reduced iterations, 2-opt converges fast
            improved = False
            it += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    nd, valid = simulate(new_route)
                    if valid and nd < best_d - 1e-7:
                        best_d = nd
                        best_route = new_route
                        improved = True
        return best_route

    def relocate(routes):
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 50 and time.time() < deadline - 10.0:
            improved = False
            it += 1
            for i in range(nR):
                for j in range(len(routes[i])):
                    node = routes[i][j]
                    if node not in cust_set:
                        continue
                    rest_i = routes[i][:j] + routes[i][j+1:]
                    d_i_rest, v_i_rest = simulate(rest_i)
                    if not v_i_rest:
                        continue
                        
                    for k in range(nR):
                        rk = routes[k]
                        if route_demand(rk) + demands[node] > capacity:
                            continue
                            
                        for p in range(len(rk) + 1):
                            if k == i and (p == j or p == j+1):
                                continue
                                
                            cand_k = rk[:p] + [node] + rk[p:]
                            d_k_new, v_k_new = simulate(cand_k)
                            if not v_k_new:
                                continue
                                
                            if k == i:
                                old_d_i, _ = simulate(routes[i])
                                new_total = d_k_new
                                old_total = old_d_i
                            else:
                                old_d_k, _ = simulate(routes[k])
                                old_total = d_i_rest + old_d_k
                                new_total = d_i_rest + d_k_new
                                
                            if new_total < old_total - 1e-7:
                                routes[k] = cand_k
                                routes[i] = rest_i
                                routes = [r for r in routes if r]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def swap(routes):
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 50 and time.time() < deadline - 10.0:
            improved = False
            it += 1
            for i in range(nR):
                for j in range(i+1, nR):
                    for pi in range(len(routes[i])):
                        ci = routes[i][pi]
                        if ci not in cust_set: continue
                        for pj in range(len(routes[j])):
                            cj = routes[j][pj]
                            if cj not in cust_set: continue
                            
                            di = route_demand(routes[i])
                            dj = route_demand(routes[j])
                            if di - demands[ci] + demands[cj] > capacity: continue
                            if dj - demands[cj] + demands[ci] > capacity: continue
                            
                            new_i = routes[i][:]
                            new_j = routes[j][:]
                            new_i[pi] = cj
                            new_j[pj] = ci
                            
                            di_new, vi = simulate(new_i)
                            dj_new, vj = simulate(new_j)
                            
                            if vi and vj:
                                old_di, _ = simulate(routes[i])
                                old_dj, _ = simulate(routes[j])
                                if di_new + dj_new < old_di + old_dj - 1e-7:
                                    routes[i] = new_i
                                    routes[j] = new_j
                                    improved = True
                                    break
                        if improved: break
                    if improved: break
                if improved: break
        return routes

    def or_opt(routes):
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 30 and time.time() < deadline - 10.0:
            improved = False
            it += 1
            for i in range(nR):
                ri = routes[i]
                if len(ri) < 3: continue
                for start in range(len(ri) - 2):
                    for length in [2, 3]:
                        if start + length > len(ri): continue
                        segment = ri[start:start+length]
                        
                        rest_i = ri[:start] + ri[start+length:]
                        d_i_rest, v_i_rest = simulate(rest_i)
                        if not v_i_rest:
                            continue
                            
                        for k in range(nR):
                            rk = routes[k]
                            seg_demand = sum(demands[n] for n in segment if n in cust_set)
                            if route_demand(rk) + seg_demand > capacity:
                                continue
                                
                            for p in range(len(rk) + 1):
                                if k == i and (start <= p <= start + length):
                                    continue
                                    
                                cand_k = rk[:p] + segment + rk[p:]
                                d_k_new, v_k_new = simulate(cand_k)
                                if not v_k_new:
                                    continue
                                    
                                if k == i:
                                    old_d_i, _ = simulate(ri)
                                    new_total = d_k_new
                                    old_total = old_d_i
                                else:
                                    old_d_k, _ = simulate(rk)
                                    old_total = d_i_rest + old_d_k
                                    new_total = d_i_rest + d_k_new
                                    
                                if new_total < old_total - 1e-7:
                                    routes[k] = cand_k
                                    routes[i] = rest_i
                                    routes = [r for r in routes if r]
                                    improved = True
                                    break
                            if improved: break
                        if improved: break
                    if improved: break
                if improved: break
        return routes

    def polish(routes):
        """
        Try to remove unnecessary charging stations.
        """
        improved = True
        it = 0
        while improved and it < 20 and time.time() < deadline - 5.0:
            improved = False
            it += 1
            new_routes = []
            for r in routes:
                new_r = r[:]
                # Try removing each station
                for idx in range(len(new_r)):
                    if new_r[idx] in station_set:
                        # Check if removing it keeps the route valid
                        test_r = new_r[:idx] + new_r[idx+1:]
                        d, v = simulate(test_r)
                        if v:
                            new_r = test_r
                            improved = True
                            break # Restart scan for this route
                new_routes.append(new_r)
            routes = new_routes
        return routes

    # --- Main Loop ---
    solutions = []
    
    # 1. Savings Construction
    s1 = construct_savings()
    if s1:
        solutions.append(s1)
        
    # 2. Nearest Neighbor Construction (Battery Aware)
    s2 = []
    unvisited = set(customers)
    while unvisited:
        # Start with the customer with earliest due date or furthest from depot?
        # Let's pick the one with earliest due date to be safe with time windows
        c = min(unvisited, key=lambda x: due[x])
        route = [c]
        unvisited.remove(c)
        
        while True:
            best_c = None
            best_d = float('inf')
            for c in unvisited:
                d, valid = simulate(route + [c])
                if valid and d < best_d:
                    best_d = d
                    best_c = c
            if best_c is None:
                break
            route.append(best_c)
            unvisited.remove(best_c)
            
        fr, _ = fix_battery(route)
        if fr:
            s2.append(fr)
            
    if s2:
        solutions.append(s2)
        
    # 3. Randomized Nearest Neighbor
    for _ in range(3):
        if time.time() > deadline - 30.0: break
        rnd = random.Random(int(time.time()*1000))
        s3 = []
        unvisited = set(customers)
        order = list(customers)
        rnd.shuffle(order)
        for c in order:
            if c in unvisited:
                route = [c]
                unvisited.remove(c)
                while unvisited:
                    best_c = None
                    best_d = float('inf')
                    for cc in unvisited:
                        d, valid = simulate(route + [cc])
                        if valid and d < best_d:
                            best_d = d
                            best_c = cc
                    if best_c is None:
                        break
                    route.append(best_c)
                    unvisited.remove(best_c)
                fr, _ = fix_battery(route)
                if fr:
                    s3.append(fr)
        if s3:
            solutions.append(s3)

    global_best = None
    global_dist = float('inf')
    
    for sol in solutions:
        if time.time() > deadline - 15.0:
            break
            
        valid_sol = []
        served = set()
        for r in sol:
            d, v = simulate(r)
            if v:
                valid_sol.append(r)
                for n in r:
                    if n in cust_set:
                        served.add(n)
                        
        if len(served) != len(customers):
            continue
            
        opt_routes = valid_sol[:]
        
        # Optimization Loop
        improved = True
        it = 0
        while improved and it < 20 and time.time() < deadline - 5.0:
            improved = False
            it += 1
            
            # 1. 2-opt
            new_routes = [two_opt(r) for r in opt_routes]
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
                
            # 2. Relocate
            new_routes = relocate(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
                
            # 3. Swap
            new_routes = swap(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
                
            # 4. Or-opt
            new_routes = or_opt(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
                
            # 5. Polish (remove extra stations)
            new_routes = polish(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
                
            # 6. 2-opt again
            new_routes = [two_opt(r) for r in opt_routes]
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True

        total_d = sum(simulate(r)[0] for r in opt_routes)
        if total_d < global_dist:
            global_dist = total_d
            global_best = opt_routes

    if global_best is None:
        # Fallback: Individual routes
        global_best = [[c] for c in customers]
        
    # Final Validation
    served = []
    valid = True
    for r in global_best:
        d, v = simulate(r)
        if not v:
            valid = False
            break
        for n in r:
            if n in cust_set:
                served.append(n)
                
    if not valid or len(served) != len(customers) or len(set(served)) != len(customers):
        # If invalid, try to fix by splitting? Or just return fallback.
        # Fallback is safer.
        global_best = [[c] for c in customers]

    return global_best

# EVOLVE-BLOCK-END
