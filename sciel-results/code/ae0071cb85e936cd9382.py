"""EV routing with time windows – direct multi-route local search with battery-aware heuristics."""
import numpy as np
import time
import math
import random

def plan(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

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
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))
    
    def simulate_route(route):
        """
        Simulate a route. Returns (feasible, total_dist, end_time, end_bat).
        Handles charging stations automatically if needed.
        Note: For simplicity and speed, we assume optimal charging decisions can be approximated
        by charging only when necessary to reach the next node or depot.
        However, the problem statement says "stations may repeat" and "refills".
        A key insight: The optimal policy is to charge only when you cannot reach the next destination 
        without going below 0. But you must also consider time windows.
        Actually, the problem says "a station adds ... time and refills".
        We need to decide WHEN to charge.
        Strategy: We will simulate the route. If battery < dist to next, we must charge.
        But which station? The one that minimizes time cost?
        Given the small number of stations (4), we can try the closest feasible station 
        or the one that allows reaching the next node.
        
        Actually, a simpler and robust approach for the simulator:
        If we run out of battery before reaching the next node, we are infeasible 
        UNLESS we stop at a station.
        To keep the local search fast, we will use a heuristic:
        If battery < dist(prev, next), we find the station 's' that minimizes 
        (dist(prev, s) + dist(s, next)) and time penalty, such that we can reach s 
        and then next.
        If no such station exists, infeasible.
        
        Let's refine: 
        1. Check if direct travel is possible (battery >= dist).
        2. If not, find a station s such that:
           - battery >= dist(prev, s)
           - battery_cap >= dist(s, next)
           - Time: arrival at s + charge_time <= due[next] (roughly)
           We pick the station that minimizes total distance detour + time penalty.
        3. If no station works, infeasible.
        """
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
            
            # Check if we can go directly
            if bat >= d_direct - 1e-9:
                t += d_direct / speed
                bat -= d_direct
                total_d += d_direct
            else:
                # Need to charge
                # Find best station
                best_station = None
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
                    # Charge time: (battery - d1) * recharge_rate? 
                    # "a station adds (battery - energy) * recharge_rate time and refills"
                    # This implies charge_time = (current_bat - energy_used_to_get_there) * recharge_rate?
                    # Wait, "refills" means battery becomes battery_cap.
                    # The text says: "adds (battery - energy) * recharge_rate time".
                    # Let's assume 'energy' is the energy used to get to the station? 
                    # Or is it the deficit? 
                    # Usually charging time = (target - current) * rate.
                    # Here "refills" implies target = battery_cap.
                    # So time = (battery_cap - (bat - d1)) * recharge_rate?
                    # Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
                    # This phrasing is ambiguous. 
                    # Interpretation A: Time = (battery_cap - current_bat_at_station) * recharge_rate.
                    # Interpretation B: Time = (current_bat_at_station) * recharge_rate? No.
                    # Let's look at standard EV problems. Usually time = (full - current) * rate.
                    # Let's assume Interpretation A.
                    
                    bat_at_s = bat - d1
                    charge_time = (battery_cap - bat_at_s) * recharge_rate
                    
                    t_v = t_s + charge_time + d2 / speed
                    total_detour_d = d1 + d2
                    
                    # Feasibility check at v
                    # Must arrive at v by due[v]
                    if t_v > due[v] + 1e-9:
                        continue
                        
                    # Cost metric: distance detour + time penalty (scaled)
                    # We want to minimize total distance primarily.
                    cost = total_detour_d
                    if cost < best_cost:
                        best_cost = cost
                        best_station = s
                        best_t_v = t_v
                        best_bat_v = battery_cap - d2
                        best_total_d_add = d1 + d2

                if best_station is None:
                    return False, 0.0, 0.0, 0.0
                
                # Update state
                t = best_t_v
                bat = best_bat_v
                total_d += best_total_d_add
                # Note: We don't update 'prev' to station, we update to v directly for the loop
                # But logically we visited the station. The distance calculation is handled.
            
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
            # Need to charge to return
            best_station = None
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
                    best_total_d_add = d1 + d2
            
            if best_station is None:
                return False, 0.0, 0.0, 0.0
            
            t = best_t_end
            bat = 0.0 # Doesn't matter, we are back
            total_d += best_total_d_add
        
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

    # --- Initialization: Clarke-Wright Savings ---
    # Start with each customer as a separate route
    routes = [[v] for v in customers]
    
    # Precompute savings: s(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    # But we have time windows and battery. 
    # A simple heuristic: Sort customers by angle or due time?
    # Let's use a greedy insertion based on savings, checking feasibility.
    
    # Better initialization: Cluster by angle from depot, then order by due time within cluster.
    angles = [math.atan2(coords[v][1], coords[v][0]) for v in customers]
    sorted_customers = [v for _, v in sorted(zip(angles, customers))]
    
    # Try to merge adjacent customers in the sorted list if feasible
    routes = []
    current = []
    for v in sorted_customers:
        trial = current + [v]
        if simulate_route(trial)[0]:
            current = trial
        else:
            if current:
                routes.append(current)
            # Check if single is feasible (it should be, but battery might be an issue if far)
            if simulate_route([v])[0]:
                current = [v]
            else:
                # If single is infeasible, it's a problem. But usually single is feasible.
                # If not, we might need to insert a station visit? 
                # The route format is just customer ids. Stations are implicit in the simulation.
                # So if single customer is infeasible, it means we can't reach it or return.
                # Given battery 300 and max dist ~141, single should always be feasible 
                # unless time window is impossible.
                current = [v] # Mark as infeasible route for now, handle later
    if current:
        routes.append(current)
        
    # Ensure all routes are feasible. If not, split further?
    # If a route is infeasible, try to remove customers and put them in new routes.
    # For now, assume initialization is mostly feasible.
    
    best_routes = [r[:] for r in routes]
    best_dist = total_distance(best_routes)
    
    # Check if any route is infeasible
    all_feasible = True
    for r in best_routes:
        if not simulate_route(r)[0]:
            all_feasible = False
            break
    if not all_feasible:
        # Fallback: One customer per route
        best_routes = [[v] for v in customers]
        best_dist = total_distance(best_routes)
        # Verify these are feasible
        for r in best_routes:
            if not simulate_route(r)[0]:
                # This should not happen. If it does, return empty or best effort.
                pass

    # --- Local Search ---
    # Moves:
    # 1. Intra-route 2-opt
    # 2. Inter-route Relocate (move customer from route A to route B)
    # 3. Inter-route Swap (swap customer in A with customer in B)
    
    T = 50.0
    alpha = 0.999
    
    def get_neighbor(routes):
        n_routes = len(routes)
        if n_routes == 0:
            return routes
        
        move_type = random.random()
        
        if move_type < 0.4:
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
            
        elif move_type < 0.7:
            # Inter-route Relocate
            # Pick random customer from random route, try to insert into another
            ri = random.randint(0, n_routes - 1)
            r = routes[ri]
            if not r:
                return routes
            idx = random.randint(0, len(r) - 1)
            cust = r[idx]
            
            # Remove from ri
            new_r_ri = r[:idx] + r[idx+1:]
            
            # Pick target route
            if n_routes > 1:
                rj = random.choice([x for x in range(n_routes) if x != ri])
            else:
                rj = ri
            
            rj_route = routes[rj]
            
            # Try all insertion points in rj_route
            best_pos = -1
            best_delta = float('inf')
            
            # Calculate current dist contribution of cust in ri
            # This is complex. Let's just evaluate full distance of new solution.
            # To speed up, we can estimate.
            
            new_routes = [r[:] for r in routes]
            new_routes[ri] = new_r_ri
            
            # Try inserting into rj
            for pos in range(len(rj_route) + 1):
                trial_rj = rj_route[:pos] + [cust] + rj_route[pos:]
                # Check feasibility of trial_rj and new_r_ri
                if not simulate_route(trial_rj)[0]:
                    continue
                if new_r_ri and not simulate_route(new_r_ri)[0]:
                    continue
                
                # Calculate delta distance
                # This is expensive. Let's just do it if we have time.
                # For now, accept if feasible and random.
                # Better: Compute delta.
                
                # Dist change in rj:
                # Old: dist(prev, next)
                # New: dist(prev, cust) + dist(cust, next)
                prev = 0 if pos == 0 else rj_route[pos-1]
                next = 0 if pos == len(rj_route) else rj_route[pos]
                old_d = dist[prev, next]
                new_d = dist[prev, cust] + dist[cust, next]
                delta_rj = new_d - old_d
                
                # Dist change in ri:
                # Removed cust.
                prev_ri = 0 if idx == 0 else r[idx-1]
                next_ri = 0 if idx == len(r) - 1 else r[idx+1]
                old_d_ri = dist[prev_ri, next_ri]
                # New:
                new_d_ri = 0
                # The link prev_ri -> next_ri is now direct
                # Old was: prev_ri -> cust + cust -> next_ri
                # Wait, we are removing cust.
                # Old dist part: dist(prev_ri, cust) + dist(cust, next_ri)
                # New dist part: dist(prev_ri, next_ri)
                old_d_ri_part = dist[prev_ri, cust] + dist[cust, next_ri]
                delta_ri = old_d_ri - old_d_ri_part
                
                delta = delta_rj + delta_ri
                
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
                    best_trial_rj = trial_rj
            
            if best_pos != -1:
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

    # Main SA loop
    iters = 0
    while time.time() < deadline:
        new_routes = get_neighbor(best_routes)
        
        new_dist = total_distance(new_routes)
        
        # Check feasibility of all routes
        feasible = True
        for r in new_routes:
            if not simulate_route(r)[0]:
                feasible = False
                break
        
        if not feasible:
            delta = 1000.0
        else:
            delta = new_dist - best_dist
        
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            best_routes = new_routes
            best_dist = new_dist
            
        T *= alpha
        iters += 1
        
        if T < 0.01:
            break

    # Final polish: 2-opt on each route
    for _ in range(5):
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
                    # Delta calc
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
        
    # Ensure all routes are feasible in final output
    final_routes = []
    for r in best_routes:
        if simulate_route(r)[0]:
            final_routes.append(r)
        else:
            # Split into singletons
            for v in r:
                if simulate_route([v])[0]:
                    final_routes.append([v])
                else:
                    # Should not happen
                    final_routes.append([v])
    
    return final_routes
