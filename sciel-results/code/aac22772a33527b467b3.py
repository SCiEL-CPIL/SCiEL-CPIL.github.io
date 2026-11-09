import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    time_limit = 145.0
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    max_battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            D[i][j] = d
            D[j][i] = d
            
    customers = list(range(1, n_customers + 1))
    
    def evaluate_route(customer_route):
        # Returns (feasible, distance, time)
        if not customer_route:
            return (True, 0.0, 0.0)
            
        curr_node = 0
        curr_time = 0.0
        curr_battery = max_battery
        total_dist = 0.0
        load = 0
        
        for cust in customer_route:
            load += demands[cust]
            if load > capacity:
                return (False, 0.0, 0.0)
                
            next_node = cust
            if D[curr_node][next_node] > curr_battery:
                best_station = -1
                best_dist = float('inf')
                for s in stations:
                    if D[curr_node][s] <= curr_battery:
                        if D[s][next_node] <= max_battery:
                            b_at_s = curr_battery - D[curr_node][s]
                            charge_t = (max_battery - b_at_s) * recharge_rate
                            d_cost = D[curr_node][s] + D[s][next_node]
                            if d_cost < best_dist:
                                best_dist = d_cost
                                best_station = s
                if best_station == -1:
                    return (False, 0.0, 0.0)
                
                curr_battery -= D[curr_node][best_station]
                curr_time += D[curr_node][best_station] / speed
                total_dist += D[curr_node][best_station]
                
                b_at_s = curr_battery
                charge_t = (max_battery - b_at_s) * recharge_rate
                curr_time += charge_t
                curr_battery = max_battery
                
                curr_battery -= D[best_station][next_node]
                curr_time += D[best_station][next_node] / speed
                total_dist += D[best_station][next_node]
            else:
                curr_battery -= D[curr_node][next_node]
                curr_time += D[curr_node][next_node] / speed
                total_dist += D[curr_node][next_node]
                
            curr_node = next_node
            
            if curr_time < ready[next_node]:
                curr_time = ready[next_node]
            if curr_time > due[next_node]:
                return (False, 0.0, 0.0)
            curr_time += service[next_node]
            
        # Return to depot
        next_node = 0
        if D[curr_node][next_node] > curr_battery:
            best_station = -1
            best_dist = float('inf')
            for s in stations:
                if D[curr_node][s] <= curr_battery:
                    if D[s][0] <= max_battery:
                        b_at_s = curr_battery - D[curr_node][s]
                        charge_t = (max_battery - b_at_s) * recharge_rate
                        d_cost = D[curr_node][s] + D[s][0]
                        if d_cost < best_dist:
                            best_dist = d_cost
                            best_station = s
            if best_station == -1:
                return (False, 0.0, 0.0)
                
            curr_battery -= D[curr_node][best_station]
            curr_time += D[curr_node][best_station] / speed
            total_dist += D[curr_node][best_station]
            
            b_at_s = curr_battery
            charge_t = (max_battery - b_at_s) * recharge_rate
            curr_time += charge_t
            curr_battery = max_battery
            
            curr_battery -= D[best_station][0]
            curr_time += D[best_station][0] / speed
            total_dist += D[best_station][0]
        else:
            curr_battery -= D[curr_node][0]
            curr_time += D[curr_node][0] / speed
            total_dist += D[curr_node][0]
            
        if curr_time > horizon:
            return (False, 0.0, 0.0)
            
        return (True, total_dist, curr_time)

    def build_explicit_route(customer_route):
        # Similar to evaluate but builds the list
        if not customer_route:
            return []
        explicit = []
        curr_node = 0
        curr_battery = max_battery
        
        for cust in customer_route:
            next_node = cust
            if D[curr_node][next_node] > curr_battery:
                best_station = -1
                best_dist = float('inf')
                for s in stations:
                    if D[curr_node][s] <= curr_battery:
                        if D[s][next_node] <= max_battery:
                            d_cost = D[curr_node][s] + D[s][next_node]
                            if d_cost < best_dist:
                                best_dist = d_cost
                                best_station = s
                if best_station != -1:
                    explicit.append(best_station)
                    curr_node = best_station
            explicit.append(next_node)
            curr_node = next_node
            curr_battery = max_battery # Reset for next leg? No, battery changes.
            # We need to track battery properly here too, or just trust evaluate_route.
            # Since we already checked feasibility, we can just re-simulate battery.
            
        # This is getting complex. Let's just re-simulate in evaluate and store the path?
        # Or modify evaluate to return path.
        
        # For now, let's just return the customer route if feasible, and assume stations are handled implicitly?
        # NO, the output must include stations.
        
        # Let's modify evaluate_route to also return the explicit path.
        pass

    # Let's rewrite evaluate_route to return (feasible, dist, time, explicit_path)
    
    def evaluate_route_full(customer_route):
        if not customer_route:
            return (True, 0.0, 0.0, [])
            
        curr_node = 0
        curr_time = 0.0
        curr_battery = max_battery
        total_dist = 0.0
        load = 0
        path = []
        
        for cust in customer_route:
            load += demands[cust]
            if load > capacity:
                return (False, 0.0, 0.0, None)
                
            next_node = cust
            if D[curr_node][next_node] > curr_battery:
                best_station = -1
                best_dist = float('inf')
                for s in stations:
                    if D[curr_node][s] <= curr_battery:
                        if D[s][next_node] <= max_battery:
                            b_at_s = curr_battery - D[curr_node][s]
                            charge_t = (max_battery - b_at_s) * recharge_rate
                            d_cost = D[curr_node][s] + D[s][next_node]
                            if d_cost < best_dist:
                                best_dist = d_cost
                                best_station = s
                if best_station == -1:
                    return (False, 0.0, 0.0, None)
                
                path.append(best_station)
                curr_battery -= D[curr_node][best_station]
                curr_time += D[curr_node][best_station] / speed
                total_dist += D[curr_node][best_station]
                
                b_at_s = curr_battery
                charge_t = (max_battery - b_at_s) * recharge_rate
                curr_time += charge_t
                curr_battery = max_battery
                
                curr_battery -= D[best_station][next_node]
                curr_time += D[best_station][next_node] / speed
                total_dist += D[best_station][next_node]
            else:
                curr_battery -= D[curr_node][next_node]
                curr_time += D[curr_node][next_node] / speed
                total_dist += D[curr_node][next_node]
                
            path.append(next_node)
            curr_node = next_node
            
            if curr_time < ready[next_node]:
                curr_time = ready[next_node]
            if curr_time > due[next_node]:
                return (False, 0.0, 0.0, None)
            curr_time += service[next_node]
            
        next_node = 0
        if D[curr_node][next_node] > curr_battery:
            best_station = -1
            best_dist = float('inf')
            for s in stations:
                if D[curr_node][s] <= curr_battery:
                    if D[s][0] <= max_battery:
                        b_at_s = curr_battery - D[curr_node][s]
                        charge_t = (max_battery - b_at_s) * recharge_rate
                        d_cost = D[curr_node][s] + D[s][0]
                        if d_cost < best_dist:
                            best_dist = d_cost
                            best_station = s
            if best_station == -1:
                return (False, 0.0, 0.0, None)
                
            path.append(best_station)
            curr_battery -= D[curr_node][best_station]
            curr_time += D[curr_node][best_station] / speed
            total_dist += D[curr_node][best_station]
            
            b_at_s = curr_battery
            charge_t = (max_battery - b_at_s) * recharge_rate
            curr_time += charge_t
            curr_battery = max_battery
            
            curr_battery -= D[best_station][0]
            curr_time += D[best_station][0] / speed
            total_dist += D[best_station][0]
        else:
            curr_battery -= D[curr_node][0]
            curr_time += D[curr_node][0] / speed
            total_dist += D[curr_node][0]
            
        if curr_time > horizon:
            return (False, 0.0, 0.0, None)
            
        return (True, total_dist, curr_time, path)

    # Construction
    routes = [[c] for c in customers]
    
    # Simple greedy merge
    # Sort customers by angle
    angles = []
    for c in customers:
        dx = coords[c][0] - coords[0][0]
        dy = coords[c][1] - coords[0][1]
        angles.append((math.atan2(dy, dx), c))
    angles.sort()
    
    # Use the angle sorted customers for initial route building?
    # Let's stick to Clarke-Wright as it's standard.
    
    savings = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = D[0][ci] + D[0][cj] - D[ci][cj]
            savings.append((s, i, j))
    savings.sort(reverse=True)
    
    # Union-Find
    parent = list(range(len(customers)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px = find(x)
        py = find(y)
        if px != py:
            parent[px] = py
            
    # Map customer index to route index
    # We will maintain a list of routes.
    # To make it faster, we'll just try to merge and check.
    
    # Re-init routes
    routes = [[c] for c in customers]
    
    # We need to check feasibility of merged routes.
    # This is O(N^2 * N) in worst case.
    
    for s, i, j in savings:
        if time.time() - start_time > time_limit * 0.3:
            break
        # Find routes containing i and j
        ri = -1
        rj = -1
        for k, r in enumerate(routes):
            if customers[i] in r:
                ri = k
            if customers[j] in r:
                rj = k
        if ri == -1 or rj == -1 or ri == rj:
            continue
            
        # Try to merge
        r_i = routes[ri]
        r_j = routes[rj]
        
        best_comb = None
        best_dist = float('inf')
        
        candidates = [
            r_i + r_j,
            r_i + r_j[::-1],
            r_i[::-1] + r_j,
            r_i[::-1] + r_j[::-1]
        ]
        
        for comb in candidates:
            res = evaluate_route_full(comb)
            if res[0]:
                if res[1] < best_dist:
                    best_dist = res[1]
                    best_comb = comb
                    
        if best_comb:
            routes[ri] = best_comb
            routes.pop(rj)
            
    # Local Search
    # 2-opt
    for i in range(len(routes)):
        if time.time() - start_time > time_limit * 0.6:
            break
        improved = True
        while improved:
            improved = False
            r = routes[i]
            n = len(r)
            for a in range(n-1):
                for b in range(a+1, n):
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    res_old = evaluate_route_full(r)
                    res_new = evaluate_route_full(new_r)
                    if res_new[0] and res_old[0]:
                        if res_new[1] < res_old[1]:
                            routes[i] = new_r
                            improved = True
                            break
                if improved:
                    break
                    
    # Relocate
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.8:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                c = routes[i][j]
                for k in range(len(routes)):
                    if k == i: continue
                    for pos in range(len(routes[k]) + 1):
                        new_i = routes[i][:j] + routes[i][j+1:]
                        new_k = routes[k][:pos] + [c] + routes[k][pos:]
                        
                        res_i = evaluate_route_full(new_i)
                        res_k = evaluate_route_full(new_k)
                        if res_i[0] and res_k[0]:
                            old_dist = evaluate_route_full(routes[i])[1] + evaluate_route_full(routes[k])[1]
                            new_dist = res_i[1] + res_k[1]
                            if new_dist < old_dist:
                                routes[i] = new_i
                                routes[k] = new_k
                                improved = True
                                break
                    if improved: break
                if improved: break
            if improved: break

    # Build final routes
    final_routes = []
    for r in routes:
        res = evaluate_route_full(r)
        if res[0]:
            final_routes.append(res[3])
        else:
            for c in r:
                res_c = evaluate_route_full([c])
                if res_c[0]:
                    final_routes.append(res_c[3])
                else:
                    final_routes.append([c])
                    
    return final_routes
