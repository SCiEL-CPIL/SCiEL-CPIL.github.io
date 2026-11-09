"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _build_dist_matrix(coords):
    n = len(coords)
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist


def _check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
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


def _fast_check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Fast feasibility check with early exits. Returns (feasible, total_distance)."""
    if not route:
        return True, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed
        
        if battery < -1e-9:
            return False, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed
    
    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist


def _route_distance(route, dist):
    """Compute total distance of a route including depot legs."""
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i+1]]
    total += dist[route[-1]][0]
    return total


def _total_route_distance(routes, dist):
    """Compute total distance across all routes."""
    return sum(_route_distance(r, dist) for r in routes)


def _construct_initial_routes(instance, dist, rng=None):
    """Construct initial routes using a savings-based heuristic with optional randomization."""
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
    
    if rng is None:
        rng = random.Random(42)
    
    routes = [[i] for i in range(1, n_customers + 1)]
    
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            # Add small random perturbation for diversity
            s += rng.uniform(0, 0.1)
            savings.append((s, ci, cj))
    
    savings.sort(reverse=True)
    
    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx
    
    route_demand = [0] * len(routes)
    for idx, r in enumerate(routes):
        for c in r:
            if c <= n_customers:
                route_demand[idx] += demands[c]
    
    for s, ci, cj in savings:
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        
        best_new_route = None
        best_dist = float('inf')
        
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist = _fast_check_route(merged, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged
        
        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            route_demand[ri] += route_demand[rj]
            route_demand[rj] = 0
            for c in best_new_route:
                route_of[c] = ri
    
    routes = [r for r in routes if r]
    
    all_served = set()
    for r in routes:
        all_served.update(r)
    
    for c in range(1, n_customers + 1):
        if c not in all_served:
            inserted = False
            for idx, r in enumerate(routes):
                current_demand = sum(demands[v] for v in r if v <= n_customers)
                if current_demand + demands[c] > capacity:
                    continue
                best_pos = -1
                best_cost = float('inf')
                for pos in range(len(r) + 1):
                    new_route = r[:pos] + [c] + r[pos:]
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        if pos == 0:
                            inc = dist[0][c] + dist[c][r[0]] - dist[0][r[0]]
                        elif pos == len(r):
                            inc = dist[r[-1]][c] + dist[c][0] - dist[r[-1]][0]
                        else:
                            inc = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = pos
                if best_pos >= 0:
                    routes[idx] = r[:best_pos] + [c] + r[best_pos:]
                    inserted = True
                    break
            if not inserted:
                routes.append([c])
    
    return routes


def _two_opt_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route with distance-based pruning."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _or_opt_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply OR-opt (relocate 1-2 consecutive nodes) improvements."""
    if len(route) < 3:
        return route
    
    n = len(route)
    improved = True
    max_passes = 12
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    return route
                segment = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                
                for pos in range(len(rest) + 1):
                    if time.time() > deadline:
                        return route
                    if pos == i or (pos > i and pos <= i + seg_len):
                        continue
                    
                    new_route = rest[:pos] + segment + rest[pos:]
                    
                    # Quick distance check
                    if len(new_route) == 0:
                        continue
                    
                    # Compute approximate distance change
                    if pos == 0:
                        if len(rest) > 0:
                            d_old = dist[0][rest[0]] + dist[rest[-1]][0]
                            d_new = dist[0][segment[0]] + dist[segment[-1]][rest[0]] + dist[rest[-1]][0]
                        else:
                            d_old = 0
                            d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                    elif pos == len(rest):
                        if len(rest) > 0:
                            d_old = dist[rest[-1]][0]
                            d_new = dist[rest[-1]][segment[0]] + dist[segment[-1]][0]
                        else:
                            d_old = 0
                            d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                    else:
                        if len(rest) > 0:
                            d_old = dist[rest[pos-1]][rest[pos]]
                            d_new = dist[rest[pos-1]][segment[0]] + dist[segment[-1]][rest[pos]]
                        else:
                            d_old = 0
                            d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                    
                    if d_new < d_old - 1e-9:
                        feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return route


def _swap_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply swap improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                # Only swap customers (not stations) for now
                if route[i] > n_customers or route[j] > n_customers:
                    continue
                
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                # Quick distance check
                if i == 0:
                    prev_i = 0
                else:
                    prev_i = route[i-1]
                if i == n-1:
                    next_i = 0
                else:
                    next_i = route[i+1]
                if j == 0:
                    prev_j = 0
                else:
                    prev_j = route[j-1]
                if j == n-1:
                    next_j = 0
                else:
                    next_j = route[j+1]
                
                d_old = dist[prev_i][route[i]] + dist[route[i]][next_i] + dist[prev_j][route[j]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[j]][next_i] + dist[prev_j][route[i]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _local_search(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply local search to all routes."""
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i] = _two_opt_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline)
        if time.time() > deadline:
            break
        routes[i] = _or_opt_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline)
        if time.time() > deadline:
            break
        routes[i] = _swap_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline)
    return routes


def _perturb_routes(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, rng, strength=1.0):
    """Perturb routes by random relocations and swaps. Returns new routes."""
    new_routes = [r[:] for r in routes]
    n_perturbations = max(2, int(3 * strength))
    
    for _ in range(n_perturbations):
        if time.time() > horizon:  # safety
            break
        
        # Randomly select a customer to relocate
        all_customers = []
        for r in new_routes:
            for v in r:
                if v <= n_customers:
                    all_customers.append(v)
        
        if not all_customers:
            continue
        
        cust = rng.choice(all_customers)
        
        # Remove from current route
        from_route_idx = -1
        from_pos = -1
        for idx, r in enumerate(new_routes):
            if cust in r:
                from_route_idx = idx
                from_pos = r.index(cust)
                break
        
        if from_route_idx == -1:
            continue
        
        new_routes[from_route_idx].pop(from_pos)
        
        # Try to insert into a random route
        target_routes = [i for i in range(len(new_routes)) if i != from_route_idx]
        if not target_routes:
            target_routes = [from_route_idx]
        
        rng.shuffle(target_routes)
        inserted = False
        
        for tr_idx in target_routes:
            tr = new_routes[tr_idx]
            current_demand = sum(demands[v] for v in tr if v <= n_customers)
            if current_demand + demands[cust] > capacity:
                continue
            
            for pos in range(len(tr) + 1):
                new_route = tr[:pos] + [cust] + tr[pos:]
                feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible:
                    new_routes[tr_idx] = new_route
                    inserted = True
                    break
            if inserted:
                break
        
        if not inserted:
            # Put it back
            new_routes[from_route_idx].insert(from_pos, cust)
    
    # Remove empty routes
    new_routes = [r for r in new_routes if r]
    
    # Ensure all customers are present
    served = set()
    for r in new_routes:
        served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in served]
    for c in missing:
        # Insert into first feasible route
        for idx, r in enumerate(new_routes):
            current_demand = sum(demands[v] for v in r if v <= n_customers)
            if current_demand + demands[c] > capacity:
                continue
            for pos in range(len(r) + 1):
                new_route = r[:pos] + [c] + r[pos:]
                feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible:
                    new_routes[idx] = new_route
                    break
            else:
                continue
            break
        else:
            new_routes.append([c])
    
    return new_routes


def _parallel_tempering_sa(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline, rng, n_chains=4, t_start=10.0, t_end=0.1):
    """
    Parallel tempering simulated annealing.
    Multiple chains at different temperatures share solutions.
    """
    # Initialize chains: one is the current best, others are perturbed
    chains = [routes[:]]
    chain_costs = [_total_route_distance(chains[0], dist)]
    
    for i in range(1, n_chains):
        perturbed = _perturb_routes(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, rng, strength=1.0 + i * 0.5)
        chains.append(perturbed)
        chain_costs.append(_total_route_distance(chains[i], dist))
    
    # Temperature schedule
    total_steps = max(1, int((deadline - time.time()) * 10))  # ~10 steps per second
    n_steps = min(total_steps, 500)  # cap at 500 steps
    t_schedule = [t_start * (t_end / t_start) ** (i / max(1, n_steps - 1)) for i in range(n_steps)]
    
    best_routes = min(chains, key=lambda r: _total_route_distance(r, dist))
    best_cost = _total_route_distance(best_routes, dist)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        t = t_schedule[step]
        
        # Each chain attempts a move
        for c_idx in range(n_chains):
            if time.time() > deadline:
                break
            
            # Generate a perturbation
            new_routes = _perturb_routes(chains[c_idx], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, rng, strength=0.5 + 0.5 * (c_idx / n_chains))
            new_cost = _total_route_distance(new_routes, dist)
            
            old_cost = chain_costs[c_idx]
            delta = new_cost - old_cost
            
            # Metropolis acceptance with temperature
            if delta < 0:
                chains[c_idx] = new_routes
                chain_costs[c_idx] = new_cost
            else:
                # Accept with probability exp(-delta / t)
                if rng.random() < math.exp(-delta / max(0.01, t)):
                    chains[c_idx] = new_routes
                    chain_costs[c_idx] = new_cost
        
        # Swap chains: hotter chains (higher index) swap with cooler if beneficial
        # This is a simplified swap - in full PT, we'd swap based on detailed balance
        if step % 5 == 0 and n_chains > 1:
            # Try to swap the two hottest chains
            i, j = n_chains - 1, n_chains - 2
            delta_swap = chain_costs[i] * chains[j][0][0] if chains[j] else 0  # simplified
            # Actually, let's just swap if it reduces the max cost
            if chain_costs[i] > chain_costs[j]:
                chains[i], chains[j] = chains[j], chains[i]
                chain_costs[i], chain_costs[j] = chain_costs[j], chain_costs[i]
        
        # Update best
        for c_idx in range(n_chains):
            cost = chain_costs[c_idx]
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_routes = chains[c_idx][:]
        
        # Periodic local search on the best chain
        if step % 20 == 0 and time.time() < deadline - 1:
            chains[0] = _local_search(chains[0], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 1)
            chain_costs[0] = _total_route_distance(chains[0], dist)
            if chain_costs[0] < best_cost - 1e-9:
                best_cost = chain_costs[0]
                best_routes = chains[0][:]
    
    return best_routes


def _solve_instance(instance, time_limit=30):
    """Solve a single instance with time limit using parallel tempering SA."""
    start_time = time.time()
    deadline = start_time + time_limit - 3  # safety margin
    
    coords = instance["coords"]
    dist = _build_dist_matrix(coords)
    
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
    
    rng = random.Random(int(start_time * 1000) % 100000)
    
    # Phase 1: Construct and locally optimize
    routes = _construct_initial_routes(instance, dist, rng)
    routes = _local_search(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline)
    
    # Phase 2: Parallel tempering SA
    if time.time() < deadline - 2:
        routes = _parallel_tempering_sa(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline, rng, n_chains=4, t_start=15.0, t_end=0.05)
    
    # Phase 3: Final local search
    routes = _local_search(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    per_instance_time = 35
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
