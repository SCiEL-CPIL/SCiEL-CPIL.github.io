import numpy as np
import time
import math
import random

def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    customers = list(range(1, n_customers + 1))
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i, j] = d
            dist[j, i] = d

    # Precompute optimal transition costs between any two nodes (including depot 0)
    # For a transition from u to v, we need to know:
    # 1. Minimal extra distance due to charging (if needed)
    # 2. Minimal time added due to charging (if needed)
    # However, charging is only needed if battery < dist(u, v).
    # But wait, the battery state depends on the previous path. 
    # We cannot precompute a fixed "cost" for u->v because it depends on current battery.
    # Therefore, we must simulate the route to check feasibility.
    
    # To make simulation fast, we will write a very tight simulation function.
    # We will use a "lazy" charging strategy in the simulation:
    # If battery < dist(u, v), we MUST charge. We pick the station that minimizes (dist(u, s) + dist(s, v)) 
    # subject to reaching s and then v.
    # Actually, the problem says "stations may repeat".
    # The optimal charging decision is complex. 
    # However, for local search, a good heuristic is:
    # If battery < dist(u, v), find station s that minimizes time penalty + distance penalty.
    # Time penalty = (dist(u, s) + dist(s, v) - dist(u, v)) / speed + charge_time.
    # Charge time = (battery_cap - (bat - dist(u, s))) * recharge_rate.
    
    # Let's precompute for each pair (u, v) the best station to use IF charging is required.
    # best_station[u][v] = s
    # best_detour_dist[u][v] = dist(u, s) + dist(s, v) - dist(u, v)
    # best_charge_time_base[u][v] = dist(u, s) (energy used to get to s)
    
    # But this depends on 'bat'. 
    # Let's stick to a fast simulation that makes the charging decision on the fly.
    # Since n_customers is small (25), and we use SA, we need millions of evaluations.
    # We must optimize the simulation.
    
    # Optimization: 
    # 1. Use numpy arrays where possible? No, the loop is sequential.
    # 2. Inline the logic.
    # 3. Precompute dist matrix.
    
    # Let's define a fast feasibility checker.
    # It returns (feasible, total_dist, end_time, end_bat)
    
    def simulate_route(route):
        if not route:
            return True, 0.0, 0.0, battery_cap
            
        t = 0.0
        bat = battery_cap
        prev = 0
        total_d = 0.0
        load = 0.0
        
        for v in route:
            load += demands[v]
            if load > capacity:
                return False, 0.0, 0.0, 0.0
                
            d_direct = dist[prev, v]
            
            if bat >= d_direct - 1e-9:
                t += d_direct / speed
                bat -= d_direct
                total_d += d_direct
            else:
                # Need to charge
                best_station = -1
                best_cost = float('inf')
                
                for s in stations:
                    if s == prev or s == v:
                        continue
                    d1 = dist[prev, s]
                    d2 = dist[s, v]
                    if bat < d1 - 1e-9:
                        continue
                    if battery_cap < d2 - 1e-9:
                        continue
                    
                    # Time to reach s
                    t_s = t + d1 / speed
                    bat_at_s = bat - d1
                    charge_time = (battery_cap - bat_at_s) * recharge_rate
                    t_v = t_s + charge_time + d2 / speed
                    
                    # Check time window at v
                    if t_v > due[v] + 1e-9:
                        continue
                        
                    # Cost: primarily distance, secondarily time
                    detour_d = d1 + d2
                    # We want to minimize total distance. 
                    # If multiple stations give same distance, pick faster one?
                    # For now, just minimize detour distance.
                    if detour_d < best_cost:
                        best_cost = detour_d
                        best_station = s
                        best_t_v = t_v
                        best_bat_v = battery_cap - d2

                if best_station == -1:
                    return False, 0.0, 0.0, 0.0
                
                t = best_t_v
                bat = best_bat_v
                total_d += best_cost
            
            # Arrived at v
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, 0.0
            t += service[v]
            prev = v
        
        # Return to depot
        d_return = dist[prev, 0]
        if bat >= d_return - 1e-9:
            t += d_return / speed
            bat -= d_return
            total_d += d_return
        else:
            best_station = -1
            best_cost = float('inf')
            for s in stations:
                if s == prev:
                    continue
                d1 = dist[prev, s]
                d2 = dist[s, 0]
                if bat < d1 - 1e-9:
                    continue
                if battery_cap < d2 - 1e-9:
                    continue
                
                bat_at_s = bat - d1
                charge_time = (battery_cap - bat_at_s) * recharge_rate
                t_end = t + d1 / speed + charge_time + d2 / speed
                
                if t_end > horizon + 1e-9:
                    continue
                
                cost = d1 + d2
                if cost < best_cost:
                    best_cost = cost
                    best_station = s
                    best_t_end = t_end

            if best_station == -1:
                return False, 0.0, 0.0, 0.0
            
            t = best_t_end
            bat = 0.0
            total_d += best_cost
        
        if t > horizon + 1e-9:
            return False, 0.0, 0.0, 0.0
            
        return True, total_d, t, bat

    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += dist[0, r[0]]
            for i in range(len(r) - 1):
                total += dist[r[i], r[i + 1]]
            total += dist[r[-1], 0]
        return total

    # --- Initialization: Greedy Insertion ---
    # Start with empty routes
    routes = []
    # Sort customers by angle for initial clustering
    angles = [math.atan2(coords[v][1], coords[v][0]) for v in customers]
    sorted_customers = [v for _, v in sorted(zip(angles, customers))]
    
    # Try to insert each customer into the best existing route or create new one
    for v in sorted_customers:
        best_route_idx = -1
        best_pos = -1
        best_delta = float('inf')
        
        # Try inserting into existing routes
        for i, r in enumerate(routes):
            # Check capacity
            if sum(demands[x] for x in r) + demands[v] > capacity:
                continue
            
            # Try all positions
            for pos in range(len(r) + 1):
                trial_r = r[:pos] + [v] + r[pos:]
                feasible, d, _, _ = simulate_route(trial_r)
                if not feasible:
                    continue
                
                # Calculate delta distance
                # Old dist of r
                # New dist of trial_r
                # To speed up, we can approximate or just use full calc if route is small
                # Given n=25, full calc is fine.
                
                # Actually, let's just calculate the distance change
                # Old: dist(prev, next)
                # New: dist(prev, v) + dist(v, next)
                prev = 0 if pos == 0 else r[pos-1]
                next = 0 if pos == len(r) else r[pos]
                old_d = dist[prev, next]
                new_d = dist[prev, v] + dist[v, next]
                delta = new_d - old_d
                
                if delta < best_delta:
                    best_delta = delta
                    best_route_idx = i
                    best_pos = pos
        
        if best_route_idx != -1:
            routes[best_route_idx].insert(best_pos, v)
        else:
            # Create new route
            if simulate_route([v])[0]:
                routes.append([v])
            else:
                # Should not happen, but handle it
                routes.append([v])
                
    best_routes = [r[:] for r in routes]
    best_dist = total_distance(best_routes)
    
    # Verify all routes are feasible
    all_feasible = True
    for r in best_routes:
        if not simulate_route(r)[0]:
            all_feasible = False
            break
            
    if not all_feasible:
        # Fallback to singletons if init failed
        best_routes = [[v] for v in customers]
        best_dist = total_distance(best_routes)

    # --- Simulated Annealing ---
    T = 100.0
    alpha = 0.9995
    iters = 0
    
    def get_neighbor(routes):
        n_routes = len(routes)
        if n_routes == 0:
            return routes
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Intra-route 2-opt
            ri = random.randint(0, n_routes - 1)
            r = routes[ri]
            if len(r) < 2:
                return routes
            i, j = sorted(random.sample(range(len(r)), 2))
            new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
            new_routes = [r[:] for r in routes]
            new_routes[ri] = new_r
            return new_routes
        elif move_type < 0.8:
            # Inter-route Relocate
            ri = random.randint(0, n_routes - 1)
            r = routes[ri]
            if not r:
                return routes
            idx = random.randint(0, len(r) - 1)
            cust = r[idx]
            
            new_r_ri = r[:idx] + r[idx+1:]
            
            if n_routes > 1:
                rj = random.choice([x for x in range(n_routes) if x != ri])
            else:
                rj = ri
                
            rj_route = routes[rj]
            
            best_pos = -1
            best_delta = float('inf')
            best_trial_rj = None
            
            for pos in range(len(rj_route) + 1):
                trial_rj = rj_route[:pos] + [cust] + rj_route[pos:]
                
                # Quick feasibility check
                if not simulate_route(trial_rj)[0]:
                    continue
                if new_r_ri and not simulate_route(new_r_ri)[0]:
                    continue
                
                # Delta calc
                prev = 0 if pos == 0 else rj_route[pos-1]
                next = 0 if pos == len(rj_route) else rj_route[pos]
                old_d = dist[prev, next]
                new_d = dist[prev, cust] + dist[cust, next]
                delta_rj = new_d - old_d
                
                prev_ri = 0 if idx == 0 else r[idx-1]
                next_ri = 0 if idx == len(r) - 1 else r[idx+1]
                old_d_ri_part = dist[prev_ri, cust] + dist[cust, next_ri]
                new_d_ri = dist[prev_ri, next_ri]
                delta_ri = new_d_ri - old_d_ri_part
                
                delta = delta_rj + delta_ri
                
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
                    best_trial_rj = trial_rj
            
            if best_pos != -1:
                new_routes = [r[:] for r in routes]
                new_routes[ri] = new_r_ri
                new_routes[rj] = best_trial_rj
                return new_routes
                
            return routes
        else:
            # Inter-route Swap
            if n_routes < 2:
                return routes
            ri, rj = random.sample(range(n_routes), 2)
            r_i = routes[ri]
            r_j = routes[rj]
            if not r_i or not r_j:
                return routes
            
            idx_i = random.randint(0, len(r_i) - 1)
            idx_j = random.randint(0, len(r_j) - 1)
            cust_i = r_i[idx_i]
            cust_j = r_j[idx_j]
            
            new_r_i = r_i[:]
            new_r_j = r_j[:]
            new_r_i[idx_i] = cust_j
            new_r_j[idx_j] = cust_i
            
            if not simulate_route(new_r_i)[0] or not simulate_route(new_r_j)[0]:
                return routes
            
            new_routes = [r[:] for r in routes]
            new_routes[ri] = new_r_i
            new_routes[rj] = new_r_j
            return new_routes

    while time.time() < deadline:
        new_routes = get_neighbor(best_routes)
        
        # Check feasibility
        feasible = True
        for r in new_routes:
            if not simulate_route(r)[0]:
                feasible = False
                break
                
        if not feasible:
            delta = 1000.0
        else:
            new_dist = total_distance(new_routes)
            delta = new_dist - best_dist
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                best_routes = new_routes
                best_dist = new_dist
        
        T *= alpha
        iters += 1
        
        if T < 0.01:
            break

    # Final polish: 2-opt
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        improved = False
        for ri in range(len(best_routes)):
            r = best_routes[ri]
            if len(r) < 2:
                continue
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    if time.time() > deadline - 5:
                        break
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    if not simulate_route(new_r)[0]:
                        continue
                    old_d = 0.0
                    if a == 0: old_d += dist[0, r[a]]
                    else: old_d += dist[r[a-1], r[a]]
                    if b == len(r) - 1: old_d += dist[r[b], 0]
                    else: old_d += dist[r[b], r[b+1]]
                    
                    new_d = 0.0
                    if a == 0: new_d += dist[0, r[b]]
                    else: new_d += dist[r[a-1], r[b]]
                    if b == len(r) - 1: new_d += dist[r[a], 0]
                    else: new_d += dist[r[a], r[b+1]]
                    
                    if new_d < old_d - 1e-9:
                        best_routes[ri] = new_r
                        improved = True
                if time.time() > deadline - 5:
                    break
        if not improved:
            break

    # Final verification
    all_served = [v for r in best_routes for v in r]
    if set(all_served) != set(customers) or len(all_served) != n_customers:
        best_routes = [[v] for v in customers]
        
    final_routes = []
    for r in best_routes:
        if simulate_route(r)[0]:
            final_routes.append(r)
        else:
            for v in r:
                if simulate_route([v])[0]:
                    final_routes.append([v])
                else:
                    final_routes.append([v])
    
    return final_routes
