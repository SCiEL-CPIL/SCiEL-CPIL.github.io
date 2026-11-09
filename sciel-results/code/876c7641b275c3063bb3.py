# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt, inf, pi, sin, cos
from heapq import heappush, heappop
import copy

def _euclidean_dist_matrix(coords):
    """Compute full Euclidean distance matrix."""
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _check_route_feasibility(route, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    """
    Check if a route is feasible. Returns (feasible, total_dist, total_time, end_battery).
    route is a list of node ids (customers and stations in visit order).
    Depot (0) is implicit at start and end.
    """
    if not route:
        return True, 0.0, 0.0, battery_cap
    
    bat = battery_cap
    t = 0.0
    total_dist = 0.0
    cur = 0
    
    for node in route:
        d = dist[cur, node]
        if bat < d - 1e-9:
            return False, total_dist, t, bat
        t += d / speed
        bat -= d
        total_dist += d
        
        if node >= 1 and node <= 25:  # customer
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        # else: station, just pass through (recharge already handled in expansion)
        cur = node
    
    d = dist[cur, 0]
    if bat < d - 1e-9:
        return False, total_dist, t, bat
    t += d / speed
    bat -= d
    total_dist += d
    t += 0  # no service at depot
    
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    
    return True, total_dist, t, bat

def _expand_with_charging(customer_seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    """
    Given a sequence of customer IDs, insert charging stations as needed.
    Returns (expanded_route, total_dist, total_time) or (None, 0, 0) if infeasible.
    """
    if not customer_seq:
        return [], 0.0, 0.0
    
    route = []
    bat = battery_cap
    t = 0.0
    total_dist = 0.0
    cur = 0
    
    for c in customer_seq:
        # Check if we can reach customer directly
        d = dist[cur, c]
        if bat >= d - 1e-9:
            t += d / speed
            bat -= d
            total_dist += d
            t = max(t, ready[c])
            if t > due[c] + 1e-9:
                return None, 0.0, 0.0
            t += service[c]
            cur = c
            route.append(c)
        else:
            # Need to charge first - find best station
            best_s = None
            best_cost = inf
            for s in stations:
                d_to_s = dist[cur, s]
                if bat >= d_to_s - 1e-9:
                    d_s_to_c = dist[s, c]
                    if d_s_to_c <= battery_cap + 1e-9:
                        # Cost is just the detour distance
                        cost = d_to_s + d_s_to_c
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
            
            if best_s is None:
                return None, 0.0, 0.0
            
            d_to_s = dist[cur, best_s]
            t += d_to_s / speed
            bat -= d_to_s
            total_dist += d_to_s
            # Recharge
            recharge_time = (battery_cap - bat) * recharge_rate
            t += recharge_time
            bat = battery_cap
            route.append(best_s)
            cur = best_s
            
            # Now go to customer
            d = dist[cur, c]
            t += d / speed
            bat -= d
            total_dist += d
            t = max(t, ready[c])
            if t > due[c] + 1e-9:
                return None, 0.0, 0.0
            t += service[c]
            cur = c
            route.append(c)
    
    # Return to depot
    d = dist[cur, 0]
    if bat >= d - 1e-9:
        t += d / speed
        bat -= d
        total_dist += d
    else:
        # Need to charge before returning
        best_s = None
        best_cost = inf
        for s in stations:
            d_to_s = dist[cur, s]
            if bat >= d_to_s - 1e-9:
                d_s_to_0 = dist[s, 0]
                if d_s_to_0 <= battery_cap + 1e-9:
                    cost = d_to_s + d_s_to_0
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
        
        if best_s is None:
            return None, 0.0, 0.0
        
        d_to_s = dist[cur, best_s]
        t += d_to_s / speed
        bat -= d_to_s
        total_dist += d_to_s
        recharge_time = (battery_cap - bat) * recharge_rate
        t += recharge_time
        bat = battery_cap
        route.append(best_s)
        cur = best_s
        
        d = dist[cur, 0]
        t += d / speed
        bat -= d
        total_dist += d
    
    if t > horizon + 1e-9:
        return None, 0.0, 0.0
    
    return route, total_dist, t

def _solve_instance(instance, time_limit=120):
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    dist = _euclidean_dist_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    start_time = time.time()
    deadline = start_time + time_limit - 2  # safety margin
    
    # --- Phase 1: Greedy Construction with 2-opt refinement ---
    routes = []  # Each: [customer_seq, demand_sum, expanded_route, total_dist]
    
    # Sort customers by due date for better initial construction
    cust_order = sorted(customers, key=lambda c: due[c])
    
    for c in cust_order:
        best_route_idx = -1
        best_pos = -1
        best_new_dist = inf
        best_new_seq = None
        best_new_expanded = None
        
        for i, r in enumerate(routes):
            if r[1] + demands[c] > capacity + 1e-9:
                continue
            seq = r[0]
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [c] + seq[pos:]
                expanded, d_exp, t_exp = _expand_with_charging(
                    new_seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                )
                if expanded is not None:
                    if d_exp < best_new_dist:
                        best_new_dist = d_exp
                        best_route_idx = i
                        best_pos = pos
                        best_new_seq = new_seq
                        best_new_expanded = expanded
        
        if best_route_idx != -1:
            routes[best_route_idx][0] = best_new_seq
            routes[best_route_idx][1] += demands[c]
            routes[best_route_idx][2] = best_new_expanded
            routes[best_route_idx][3] = best_new_dist
        else:
            expanded, d_exp, t_exp = _expand_with_charging(
                [c], dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
            )
            if expanded is not None:
                routes.append([[c], demands[c], expanded, d_exp])
            else:
                routes.append([[c], demands[c], None, inf])
    
    # --- Phase 2: Parallel Tempering Simulated Annealing ---
    # Maintain multiple independent chains (temperatures)
    n_chains = 3
    chains = []
    
    for ci in range(n_chains):
        if ci == 0:
            # First chain starts from greedy solution
            chain_routes = [list(r) for r in routes]
        else:
            # Other chains start with perturbed versions
            chain_routes = [list(r) for r in routes]
            rng = random.Random(ci + 100)
            # Apply a random perturbation
            if len(chain_routes) > 0:
                ri = rng.randint(0, len(chain_routes) - 1)
                seq = chain_routes[ri][0]
                if len(seq) >= 2:
                    a, b = rng.sample(range(len(seq)), 2)
                    if a > b:
                        a, b = b, a
                    seq[a:b+1] = seq[a:b+1][::-1]
                    chain_routes[ri][0] = seq
                    expanded, d_exp, t_exp = _expand_with_charging(
                        seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    if expanded is not None:
                        chain_routes[ri][2] = expanded
                        chain_routes[ri][3] = d_exp
                    else:
                        chain_routes[ri][2] = None
                        chain_routes[ri][3] = inf
    
    for ci in range(n_chains):
        chains.append({
            'routes': [list(r) for r in chains[ci]['routes']] if ci == 0 else [list(r) for r in chain_routes],
            'temp': 100.0 / (1.0 + ci * 0.5),  # Higher temperature for later chains
            'best': None,
            'best_dist': inf
        })
    
    # Initialize best for each chain
    for ci in range(n_chains):
        total_d = sum(r[3] for r in chains[ci]['routes'] if r[3] < inf)
        chains[ci]['best_dist'] = total_d
        chains[ci]['best'] = [list(r) for r in chains[ci]['routes']]
    
    # Global best
    global_best_routes = None
    global_best_dist = inf
    
    for ci in range(n_chains):
        if chains[ci]['best_dist'] < global_best_dist:
            global_best_dist = chains[ci]['best_dist']
            global_best_routes = [list(r) for r in chains[ci]['best']]
    
    # SA loop
    sa_iter = 0
    while time.time() < deadline:
        # Pick a chain to operate on (round-robin with some randomness)
        ci = sa_iter % n_chains
        sa_iter += 1
        
        routes_ci = chains[ci]['routes']
        temp = chains[ci]['temp']
        
        # Generate a move
        move_type = random.random()
        new_routes = [list(r) for r in routes_ci]
        valid = True
        old_total = sum(r[3] for r in routes_ci if r[3] < inf)
        
        if move_type < 0.4 and len(routes_ci) >= 1:
            # Relocate a customer from one route to another
            ri = random.randint(0, len(routes_ci) - 1)
            seq_i = routes_ci[ri][0]
            if len(seq_i) < 1:
                valid = False
            else:
                k = random.randint(0, len(seq_i) - 1)
                c = seq_i[k]
                new_seq_i = seq_i[:k] + seq_i[k+1:]
                
                # Remove from route i
                if new_seq_i:
                    exp_i, d_i, t_i = _expand_with_charging(
                        new_seq_i, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    if exp_i is None:
                        valid = False
                    else:
                        new_routes[ri] = [new_seq_i, routes_ci[ri][1] - demands[c], exp_i, d_i]
                else:
                    new_routes[ri] = [[], 0.0, None, 0.0]
                
                # Insert into another route
                if valid:
                    rj_candidates = [j for j in range(len(routes_ci)) if j != ri]
                    if not rj_candidates:
                        valid = False
                    else:
                        rj = random.choice(rj_candidates)
                        if routes_ci[rj][1] + demands[c] > capacity + 1e-9:
                            valid = False
                        else:
                            seq_j = routes_ci[rj][0]
                            pos = random.randint(0, len(seq_j))
                            new_seq_j = seq_j[:pos] + [c] + seq_j[pos:]
                            exp_j, d_j, t_j = _expand_with_charging(
                                new_seq_j, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                            )
                            if exp_j is None:
                                valid = False
                            else:
                                new_routes[rj] = [new_seq_j, routes_ci[rj][1] + demands[c], exp_j, d_j]
        
        elif move_type < 0.7 and len(routes_ci) >= 1:
            # 2-opt within a route
            ri = random.randint(0, len(routes_ci) - 1)
            seq = routes_ci[ri][0]
            if len(seq) < 2:
                valid = False
            else:
                a, b = random.sample(range(len(seq)), 2)
                if a > b:
                    a, b = b, a
                new_seq = seq[:a] + seq[a:b+1][::-1] + seq[b+1:]
                exp, d_exp, t_exp = _expand_with_charging(
                    new_seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                )
                if exp is None:
                    valid = False
                else:
                    new_routes[ri] = [new_seq, routes_ci[ri][1], exp, d_exp]
        
        elif move_type < 0.85 and len(routes_ci) >= 2:
            # Swap two customers between routes
            ri, rj = random.sample(range(len(routes_ci)), 2)
            seq_i = routes_ci[ri][0]
            seq_j = routes_ci[rj][0]
            if len(seq_i) < 1 or len(seq_j) < 1:
                valid = False
            else:
                ci_idx = random.randint(0, len(seq_i) - 1)
                cj_idx = random.randint(0, len(seq_j) - 1)
                c_i = seq_i[ci_idx]
                c_j = seq_j[cj_idx]
                
                new_seq_i = seq_i[:]
                new_seq_i[ci_idx] = c_j
                new_seq_j = seq_j[:]
                new_seq_j[cj_idx] = c_i
                
                # Check capacity
                if routes_ci[ri][1] - demands[c_i] + demands[c_j] > capacity + 1e-9:
                    valid = False
                elif routes_ci[rj][1] - demands[c_j] + demands[c_i] > capacity + 1e-9:
                    valid = False
                else:
                    exp_i, d_i, t_i = _expand_with_charging(
                        new_seq_i, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    exp_j, d_j, t_j = _expand_with_charging(
                        new_seq_j, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    if exp_i is None or exp_j is None:
                        valid = False
                    else:
                        new_routes[ri] = [new_seq_i, routes_ci[ri][1] - demands[c_i] + demands[c_j], exp_i, d_i]
                        new_routes[rj] = [new_seq_j, routes_ci[rj][1] - demands[c_j] + demands[c_i], exp_j, d_j]
        
        elif move_type < 0.95 and len(routes_ci) >= 1:
            # LNS-style: remove k customers from a route, re-insert
            ri = random.randint(0, len(routes_ci) - 1)
            seq = routes_ci[ri][0]
            if len(seq) < 2:
                valid = False
            else:
                k_remove = min(3, random.randint(1, len(seq) - 1))
                removed_idx = random.sample(range(len(seq)), k_remove)
                removed = [seq[i] for i in sorted(removed_idx)]
                new_seq_i = [seq[i] for i in range(len(seq)) if i not in set(removed_idx)]
                
                if new_seq_i:
                    exp_i, d_i, t_i = _expand_with_charging(
                        new_seq_i, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    if exp_i is None:
                        valid = False
                    else:
                        new_routes[ri] = [new_seq_i, routes_ci[ri][1] - sum(demands[c] for c in removed), exp_i, d_i]
                else:
                    new_routes[ri] = [[], 0.0, None, 0.0]
                
                if valid:
                    for c in removed:
                        placed = False
                        for j in range(len(new_routes)):
                            if new_routes[j][1] + demands[c] > capacity + 1e-9:
                                continue
                            seq_j = new_routes[j][0]
                            best_pos = -1
                            best_d = inf
                            for pos in range(len(seq_j) + 1):
                                new_seq_j = seq_j[:pos] + [c] + seq_j[pos:]
                                exp, d_exp, t_exp = _expand_with_charging(
                                    new_seq_j, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                                )
                                if exp is not None and d_exp < best_d:
                                    best_d = d_exp
                                    best_pos = pos
                                    best_exp = exp
                            if best_pos != -1:
                                new_routes[j] = [
                                    new_routes[j][0][:best_pos] + [c] + new_routes[j][0][best_pos:],
                                    new_routes[j][1] + demands[c],
                                    best_exp,
                                    best_d
                                ]
                                placed = True
                                break
                        if not placed:
                            valid = False
                            break
        
        else:
            # Do nothing (keep current)
            valid = True
        
        if not valid:
            continue
        
        new_total = sum(r[3] for r in new_routes if r[3] < inf)
        delta = new_total - old_total
        
        # Acceptance: always accept if better, else with probability exp(-delta/temp)
        if delta <= 0:
            chains[ci]['routes'] = new_routes
            if new_total < chains[ci]['best_dist']:
                chains[ci]['best_dist'] = new_total
                chains[ci]['best'] = [list(r) for r in new_routes]
                if new_total < global_best_dist:
                    global_best_dist = new_total
                    global_best_routes = [list(r) for r in new_routes]
        else:
            if random.random() < 2.718281828 ** (-delta / temp):
                chains[ci]['routes'] = new_routes
        
        # Temperature update: cool down slowly
        chains[ci]['temp'] *= 0.9995
        
        # Parallel tempering swap: with small probability, swap solutions between adjacent chains
        if sa_iter > 100 and random.random() < 0.01:
            other_ci = (ci + 1) % n_chains
            # Metropolis criterion for swap
            d1 = chains[ci]['best_dist']
            d2 = chains[other_ci]['best_dist']
            t1 = chains[ci]['temp']
            t2 = chains[other_ci]['temp']
            # Delta in energy*temperature space
            swap_prob = 1.0 if d1 * t2 <= d2 * t1 else 2.718281828 ** (-(d1 * t2 - d2 * t1) / max(t1, t2))
            if random.random() < swap_prob:
                chains[ci]['routes'], chains[other_ci]['routes'] = chains[other_ci]['routes'], chains[ci]['routes']
                chains[ci]['temp'], chains[other_ci]['temp'] = chains[other_ci]['temp'], chains[ci]['temp']
        
        # Occasional restart of a cold chain if stuck
        if sa_iter % 500 == 0:
            for ci2 in range(n_chains):
                if chains[ci2]['temp'] < 1.0:
                    chains[ci2]['temp'] = 50.0
                    # Perturb the chain
                    routes_ci2 = chains[ci2]['routes']
                    if len(routes_ci2) > 0:
                        ri2 = random.randint(0, len(routes_ci2) - 1)
                        seq2 = routes_ci2[ri2][0]
                        if len(seq2) >= 2:
                            a2, b2 = random.sample(range(len(seq2)), 2)
                            if a2 > b2:
                                a2, b2 = b2, a2
                            seq2[a2:b2+1] = seq2[a2:b2+1][::-1]
                            routes_ci2[ri2][0] = seq2
                            exp2, d_exp2, t_exp2 = _expand_with_charging(
                                seq2, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                            )
                            if exp2 is not None:
                                routes_ci2[ri2][2] = exp2
                                routes_ci2[ri2][3] = d_exp2
    
    # Final: return global best
    if global_best_routes is None:
        # Fallback: use greedy solution
        global_best_routes = routes
    
    # Build final expanded routes
    final_routes = []
    for r in global_best_routes:
        if r[2] is not None:
            final_routes.append(r[2])
        else:
            # Try to expand
            expanded, d_exp, t_exp = _expand_with_charging(
                r[0], dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
            )
            if expanded is not None:
                final_routes.append(expanded)
            else:
                # Emergency: put each customer in its own route
                for c in r[0]:
                    exp, _, _ = _expand_with_charging(
                        [c], dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    if exp is not None:
                        final_routes.append(exp)
    
    return final_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145
    routes = _solve_instance(instance, time_limit=140)
    return routes
# EVOLVE-BLOCK-END
