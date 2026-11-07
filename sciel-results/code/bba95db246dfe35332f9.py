import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))
    
    # Helper to evaluate a route: returns (valid, distance, time)
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, 0.0
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, 0.0
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, 0.0
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, 0.0
        if t > horizon + _EPS:
            return False, 0.0, 0.0
        d_total += d
        return True, d_total, t

    def evaluate_routes(routes):
        """Returns (valid, total_distance) for a list of routes."""
        total_d = 0.0
        for r in routes:
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Construct Initial Solution ---
    # Use a simple insertion heuristic to create a good starting point.
    # Sort by due date, insert into best position in existing routes.
    
    # Start with empty routes
    routes = []
    
    # Sort customers by due date (earliest first) to help with time windows
    cust_sorted = sorted(customers, key=lambda c: due[c])
    
    for c in cust_sorted:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        # Try inserting into existing routes
        for i, r in enumerate(routes):
            # Try all positions
            for pos in range(len(r) + 1):
                trial = r[:pos] + [c] + r[pos:]
                valid, d, _ = check_route(trial)
                if valid:
                    # Cost is the increase in distance
                    # We need old distance to calculate delta, but for simplicity 
                    # and speed in construction, we can just use total distance 
                    # or a heuristic. Let's use total distance of the new route.
                    # Actually, minimizing delta is better.
                    # Let's just pick the first valid one with lowest resulting route distance?
                    # No, that's expensive. 
                    # Simple heuristic: insert where it fits and minimizes distance increase.
                    # We don't have old distance stored easily. 
                    # Let's just pick the position with minimum distance in the new route?
                    # No, that's not right either.
                    
                    # Let's just use the total distance of the trial route as a proxy 
                    # for "goodness" if we don't track deltas.
                    # Actually, let's just track the total distance of the new route.
                    if d < best_cost:
                        best_cost = d
                        best_route_idx = i
                        best_pos = pos
        
        if best_route_idx != -1:
            r = routes[best_route_idx]
            routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
        else:
            # Create new route
            # Check if single customer route is valid
            valid, d, _ = check_route([c])
            if valid:
                routes.append([c])
            else:
                # This should not happen with full battery and small demands, but just in case
                # If a single customer is invalid, the instance is likely infeasible or my check is too strict.
                # Given constraints (battery 300, coords 0-100, max dist ~141), single cust should always be valid.
                routes.append([c])

    # --- Simulated Annealing ---
    
    # Initial temperature
    T = 100.0
    T_min = 0.1
    alpha = 0.9995  # Cooling rate
    
    best_routes = [list(r) for r in routes]
    best_val, best_dist = evaluate_routes(best_routes)
    if not best_val:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
        best_val, best_dist = evaluate_routes(best_routes)

    current_routes = [list(r) for r in best_routes]
    current_val, current_dist = evaluate_routes(current_routes)
    current_cost = current_dist if current_val else float('inf')
    
    rng = np.random.default_rng(42)

    def get_neighbor(routes):
        """Generate a neighbor solution."""
        new_routes = [list(r) for r in routes]
        
        # Randomly choose a move type
        # 1. Relocate (within same route)
        # 2. Relocate (to another route)
        # 3. 2-Opt (within same route)
        # 4. Swap (between two routes)
        
        move_type = rng.integers(0, 4)
        
        # Find non-empty routes
        non_empty_idx = [i for i, r in enumerate(new_routes) if len(r) > 0]
        
        if not non_empty_idx:
            return new_routes
            
        if len(non_empty_idx) == 1:
            # Only one route, only internal moves
            i = non_empty_idx[0]
            r = new_routes[i]
            if len(r) < 2:
                return new_routes
            
            if move_type in [0, 2]: # Relocate or 2-Opt
                if move_type == 0:
                    # Relocate
                    j = rng.integers(0, len(r))
                    c = r.pop(j)
                    k = rng.integers(0, len(r) + 1)
                    r.insert(k, c)
                else:
                    # 2-Opt
                    j = rng.integers(0, len(r) - 1)
                    k = rng.integers(j + 1, len(r))
                    r[j:k+1] = r[j:k+1][::-1]
            else:
                # If move type requires 2 routes, do nothing
                pass
        else:
            # Multiple routes
            i = non_empty_idx[rng.integers(0, len(non_empty_idx))]
            r_i = new_routes[i]
            
            if move_type == 0: # Relocate within same
                if len(r_i) >= 2:
                    j = rng.integers(0, len(r_i))
                    c = r_i.pop(j)
                    k = rng.integers(0, len(r_i) + 1)
                    r_i.insert(k, c)
            elif move_type == 1: # Relocate to other route
                # Pick another route
                other_idx = [idx for idx in non_empty_idx if idx != i]
                if other_idx:
                    m = other_idx[rng.integers(0, len(other_idx))]
                    r_m = new_routes[m]
                    if len(r_i) > 0:
                        j = rng.integers(0, len(r_i))
                        c = r_i.pop(j)
                        k = rng.integers(0, len(r_m) + 1)
                        r_m.insert(k, c)
            elif move_type == 2: # 2-Opt within same
                if len(r_i) >= 2:
                    j = rng.integers(0, len(r_i) - 1)
                    k = rng.integers(j + 1, len(r_i))
                    r_i[j:k+1] = r_i[j:k+1][::-1]
            elif move_type == 3: # Swap between two routes
                other_idx = [idx for idx in non_empty_idx if idx != i]
                if other_idx:
                    m = other_idx[rng.integers(0, len(other_idx))]
                    r_m = new_routes[m]
                    if len(r_i) > 0 and len(r_m) > 0:
                        j = rng.integers(0, len(r_i))
                        k = rng.integers(0, len(r_m))
                        # Swap nodes
                        c1 = r_i[j]
                        c2 = r_m[k]
                        r_i[j] = c2
                        r_m[k] = c1
                        
        return new_routes

    # Main SA Loop
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        new_routes = get_neighbor(current_routes)
        
        # Remove empty routes for evaluation efficiency? 
        # No, keep structure stable to avoid index issues, but evaluate only non-empty.
        # Actually, evaluate_routes handles empty routes (returns 0 dist).
        
        new_val, new_dist = evaluate_routes(new_routes)
        new_cost = new_dist if new_val else float('inf')
        
        if new_cost < current_cost:
            # Accept better solution
            current_routes = new_routes
            current_cost = new_cost
            if new_cost < best_dist:
                best_routes = [list(r) for r in new_routes]
                best_dist = new_cost
        else:
            # Accept worse solution with probability
            delta = new_cost - current_cost
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            
            if rng.random() < prob:
                current_routes = new_routes
                current_cost = new_cost
        
        T *= alpha

    # Final Validation
    final_val, final_dist = evaluate_routes(best_routes)
    if not final_val:
        # Fallback to valid baseline
        best_routes = [[c] for c in customers]
        
    return best_routes
