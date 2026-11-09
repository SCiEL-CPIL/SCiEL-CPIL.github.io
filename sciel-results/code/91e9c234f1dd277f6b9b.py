import numpy as np
import math
import time
import random


def plan(instance):
    start_wall = time.time()
    total_deadline = start_wall + 120

    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            dist[i,j] = d
            dist[j,i] = d

    def check_route(route):
        if not route:
            return True, 0.0, 0.0
        
        total_demand = 0
        for v in route:
            if v <= n_customers:
                total_demand += demands[v]
        if total_demand > capacity:
            return False, 0.0, 0.0
        
        battery = battery_max
        t = 0.0
        total_d = 0.0
        prev = 0
        
        for v in route:
            d = dist[prev][v]
            battery -= d
            t += d / speed
            total_d += d
            
            if battery < -1e-9:
                return False, 0.0, 0.0
            
            if v <= n_customers:
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False, 0.0, 0.0
                t += service[v]
            else:
                energy_needed = battery_max - battery
                t += energy_needed * recharge_rate
                battery = battery_max
            
            prev = v
        
        d_back = dist[prev][0]
        battery -= d_back
        t += d_back / speed
        total_d += d_back
        
        if battery < -1e-9 or t > horizon + 1e-9:
            return False, 0.0, 0.0
            
        return True, total_d, t

    def route_distance(route):
        if not route:
            return 0.0
        total = dist[0][route[0]]
        for i in range(len(route)-1):
            total += dist[route[i]][route[i+1]]
        total += dist[route[-1]][0]
        return total

    def solve_one():
        # 1. Construct Initial Routes (Savings)
        routes = [[i] for i in range(1, n_customers+1)]
        route_of = {i: i-1 for i in range(1, n_customers+1)}
        route_demand = [demands[i] for i in range(1, n_customers+1)]
        
        savings = []
        for i in range(1, n_customers+1):
            for j in range(i+1, n_customers+1):
                s = dist[0][i] + dist[0][j] - dist[i][j]
                savings.append((s, i, j))
        savings.sort(reverse=True)
        
        for s, ci, cj in savings:
            ri = route_of.get(ci, -1)
            rj = route_of.get(cj, -1)
            if ri == -1 or rj == -1 or ri == rj:
                continue
            if route_demand[ri] + route_demand[rj] > capacity:
                continue
                
            ri_route = routes[ri]
            rj_route = routes[rj]
            
            # Try all 4 merge patterns
            candidates = [
                ri_route + rj_route,
                rj_route + ri_route,
                ri_route[::-1] + rj_route,
                rj_route[::-1] + ri_route,
            ]
            
            best = None
            best_cost = float('inf')
            for cand in candidates:
                if len(cand) == 0: continue
                feas, d, _ = check_route(cand)
                if feas and d < best_cost:
                    best = cand
                    best_cost = d
                    
            if best is not None:
                routes[ri] = best
                routes[ri][0] = ri # dummy marker? No, just store route
                # Actually let's just keep list of lists
                routes[ri] = best
                # update mapping
                for node in best:
                    route_of[node] = ri
                routes[rj] = []
                route_demand[ri] += route_demand[rj]
                route_demand[rj] = 0
                
        routes = [r for r in routes if r]
        
        # Safety: ensure all customers are in
        covered = set()
        for r in routes:
            covered.update(r)
        for c in range(1, n_customers+1):
            if c not in covered:
                # Try to insert
                inserted = False
                for idx, r in enumerate(routes):
                    cur_dem = sum(demands[v] for v in r if v <= n_customers)
                    if cur_dem + demands[c] > capacity: continue
                    # Try positions
                    for pos in range(len(r)+1):
                        new_r = r[:pos] + [c] + r[pos:]
                        feas, _, _ = check_route(new_r)
                        if feas:
                            routes[idx] = new_r
                            for node in new_r:
                                route_of[node] = idx
                            inserted = True
                            break
                    if inserted: break
                if not inserted:
                    routes.append([c])
                    route_of[c] = len(routes)-1

        # 2. Local Search (2-opt) with Time Limit
        current_deadline = time.time() + 30
        improved = True
        passes = 0
        while improved and passes < 5 and time.time() < current_deadline:
            improved = False
            passes += 1
            for i in range(len(routes)):
                if time.time() > current_deadline: break
                r = routes[i]
                if len(r) < 2: continue
                n = len(r)
                for a in range(n-1):
                    if time.time() > current_deadline: break
                    prev_a = r[a-1] if a > 0 else 0
                    next_b = r[n-1+1] if n-1 < n-1 else 0 # careful with index
                    
                    # Standard 2-opt: reverse segment [a, b]
                    for b in range(a+1, n):
                        if time.time() > current_deadline: break
                        
                        # Calculate distance change
                        # Old: prev_a -> r[a] ... r[b] -> next_b
                        # New: prev_a -> r[b] ... r[a] -> next_b
                        
                        # next_b is r[b+1] if b < n-1 else 0
                        next_b = r[b+1] if b < n-1 else 0
                        
                        d_old = dist[prev_a][r[a]] + dist[r[b]][next_b]
                        d_new = dist[prev_a][r[b]] + dist[r[a]][next_b]
                        
                        if d_new < d_old - 1e-9:
                            new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                            feas, _, _ = check_route(new_r)
                            if feas:
                                routes[i] = new_r
                                improved = True
                                # Restart loops? For speed, just continue scanning
                                # But we changed the route, so indices are same, but values reversed
                                # To be safe, break inner loop
                                break

        # 3. Inter-route moves (Move & Swap)
        current_deadline = time.time() + 40
        improved = True
        passes = 0
        while improved and passes < 5 and time.time() < current_deadline:
            improved = False
            passes += 1
            for i in range(len(routes)):
                if time.time() > current_deadline: break
                for j in range(len(routes)):
                    if time.time() > current_deadline: break
                    if i == j: continue
                    # Move
                    for idx, node in enumerate(routes[i]):
                        if node > n_customers: continue
                        # Check capacity of dest
                        dem_j = sum(demands[v] for v in routes[j] if v <= n_customers)
                        if dem_j + demands[node] > capacity: continue
                        
                        new_i = routes[i][:idx] + routes[i][idx+1:]
                        if not new_i: continue
                        feas_i, d_i, _ = check_route(new_i)
                        if not feas_i: continue
                        
                        # Try inserting node into j
                        best_pos = -1
                        best_d_j = float('inf')
                        for pos in range(len(routes[j])+1):
                            new_j = routes[j][:pos] + [node] + routes[j][pos:]
                            feas_j, d_j, _ = check_route(new_j)
                            if feas_j and d_j < best_d_j:
                                best_d_j = d_j
                                best_pos = pos
                                
                        if best_pos >= 0:
                            old_total = route_distance(routes[i]) + route_distance(routes[j])
                            new_total = d_i + best_d_j
                            if new_total < old_total - 1e-9:
                                routes[i] = new_i
                                routes[j] = routes[j][:best_pos] + [node] + routes[j][best_pos:]
                                improved = True
                                # Update route_of? Not strictly needed if we rebuild it, but good for logic
                                # Actually, let's just rebuild route_of after inter-route phase to be safe? 
                                # Or keep it updated.
                                # For speed, let's just break and let next pass handle consistency if needed.
                                # But we need route_of to be correct for subsequent operations if we use it.
                                # Let's just break out of loops.
                                break
                    if improved: break
                if improved: break
            # Cleanup empty routes
            routes = [r for r in routes if r]
            # Rebuild route_of
            route_of = {}
            for idx, r in enumerate(routes):
                for node in r:
                    route_of[node] = idx
            route_demand = [0] * len(routes)
            for idx, r in enumerate(routes):
                for node in r:
                    if node <= n_customers:
                        route_demand[idx] += demands[node]

        # Final Polish 2-opt
        current_deadline = time.time() + 20
        for i in range(len(routes)):
            if time.time() > current_deadline: break
            r = routes[i]
            if len(r) < 2: continue
            improved = True
            while improved and time.time() < current_deadline:
                improved = False
                n = len(r)
                for a in range(n-1):
                    if time.time() > current_deadline: break
                    prev_a = r[a-1] if a > 0 else 0
                    for b in range(a+1, n):
                        if time.time() > current_deadline: break
                        next_b = r[b+1] if b < n-1 else 0
                        d_old = dist[prev_a][r[a]] + dist[r[b]][next_b]
                        d_new = dist[prev_a][r[b]] + dist[r[a]][next_b]
                        if d_new < d_old - 1e-9:
                            new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                            feas, _, _ = check_route(new_r)
                            if feas:
                                r = new_r
                                improved = True
                                break
                routes[i] = r

        return routes

    # Run multiple restarts and keep the best
    best_routes = None
    best_dist = float('inf')
    
    # Time management: 150s total limit.
    # We have 3 instances? No, "returns routes for 3 fixed generated instances".
    # Wait, the prompt says: "Write plan(instance), which returns routes for 3 fixed generated instances, each with 25 customers..."
    # "instance is a dict..."
    # Does plan take ONE instance and return routes for THAT instance?
    # "Returns a list of routes... Each route starts at time 0..."
    # "mean over instances of total Euclidean distance"
    # "instance" in plan(instance) suggests plan is called PER INSTANCE.
    # "Write plan(instance), which returns routes for 3 fixed generated instances" is slightly ambiguous.
    # Usually in these tasks, plan(instance) is called once per instance, and the evaluator calls it 3 times (or the function is expected to handle 3?).
    # Let's look at "CURRENT PROGRAM". It calls `_solve_instance(instance, per_instance_time)` once.
    # So `plan` handles ONE instance.
    # I will assume `plan` is called 3 times, each with 150s limit?
    # "The evaluator KILLS the program after 150s wall-clock."
    # This usually means the TOTAL time for the whole evaluation (all 3 instances) is 150s?
    # Or is it 150s PER INSTANCE?
    # "Each evaluation must finish within 120 s." (from task description)
    # "The evaluator KILLS the program after 150s wall-clock." (from directive)
    # If `plan` is called 3 times, and each call has a 150s limit, that's 450s total.
    # But if the evaluator runs the script once, and the script calls plan 3 times?
    # No, standard interface: `plan(instance)` is the entry point.
    # If the evaluator calls `plan` 3 times, it likely enforces the time limit per call OR total.
    # Given "KILLS the program after 150s wall-clock", it's safer to assume the TOTAL budget for the whole script execution is 150s.
    # So if I am called 3 times, I have 50s per call on average.
    # However, if I only see one `instance` argument, I don't know if I'm call #1, #2, or #3.
    # To be safe, I will assume I have ~40-50 seconds available for this specific call, to leave room for the others if they are sequential.
    # Or, if the evaluator resets the timer per call, I have 150s.
    # Let's look at the current program: `per_instance_time = 140`. This implies it expects ~150s per call.
    # BUT the directive says "KILLS the program after 150s wall-clock". This is a hard limit on the process.
    # If the process is reused for 3 calls, the 150s is total.
    # If the process is restarted for each call, 150s is per call.
    # Given the ambiguity, I will use a conservative time limit of 45s for this instance.
    # This ensures that even if called 3 times sequentially in the same process, I finish in 135s.
    
    # Let's structure the restarts within this 45s window.
    
    # Actually, if I just return the first solution, it might be suboptimal.
    # I'll try 3 restarts if time permits.
    
    deadline = time.time() + 45
    
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        routes = solve_one()
        
        # Verify
        total_d = 0
        all_feasible = True
        covered = set()
        for r in routes:
            feas, d, _ = check_route(r)
            if not feas:
                all_feasible = False
                break
            total_d += d
            covered.update(r)
            
        # Check all customers covered
        missing = [c for c in range(1, n_customers+1) if c not in covered]
        if missing:
            # Fix missing
            for c in missing:
                # Try to insert
                inserted = False
                for idx, r in enumerate(routes):
                    cur_dem = sum(demands[v] for v in r if v <= n_customers)
                    if cur_dem + demands[c] > capacity: continue
                    for pos in range(len(r)+1):
                        new_r = r[:pos] + [c] + r[pos:]
                        feas, d, _ = check_route(new_r)
                        if feas:
                            routes[idx] = new_r
                            total_d += d # approx update
                            inserted = True
                            break
                    if inserted: break
                if not inserted:
                    routes.append([c])
                    total_d += route_distance([c])
                    # Re-verify total dist
                    total_d = 0
                    for r in routes:
                        _, d, _ = check_route(r)
                        total_d += d
        
        if all_feasible and (best_routes is None or total_d < best_dist):
            best_dist = total_d
            best_routes = routes
            
    if best_routes is None:
        # Fallback: single customer per route
        best_routes = [[i] for i in range(1, n_customers+1)]
        
    return best_routes
