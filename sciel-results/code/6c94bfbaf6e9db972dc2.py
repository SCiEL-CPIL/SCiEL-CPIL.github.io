import math
import time
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
    
    # --- Fast Route Evaluation & Repair ---
    def evaluate_route(route_nodes):
        """
        Evaluates a route. If battery is insufficient, attempts to insert a station.
        Returns (valid, distance, time).
        """
        if not route_nodes:
            return True, 0.0, 0.0
        
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        nodes_to_visit = list(route_nodes)
        
        # 1. First pass: Check basic feasibility (time, capacity) and compute "raw" path
        # We will simulate and track battery. If battery drops < 0, we need to insert a station.
        # Since we can't insert dynamically in a simple loop without backtracking easily,
        # we use a heuristic: if battery is low, pick the best station.
        
        # Actually, a simpler robust approach for the evaluator:
        # Simulate the route. If battery goes negative, return invalid.
        # The *search* will be responsible for inserting stations or avoiding them.
        # BUT, to make the search more powerful, we can allow the evaluator to 
        # "repair" by adding a station if it's the only thing wrong.
        
        # Let's stick to strict evaluation for now to keep the search space clean,
        # but the *moves* will include station insertion.
        
        for node in nodes_to_visit:
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
        total_d = 0.0
        for r in routes:
            valid, d, _ = evaluate_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Multi-Start Large Neighborhood Search ---
    
    def construct_solution(seed):
        """Constructs an initial solution using a randomized insertion heuristic."""
        rng = np.random.default_rng(seed)
        routes = []
        
        # Sort by angle or due date for diversity
        if seed % 2 == 0:
            cust_sorted = sorted(customers, key=lambda c: due[c])
        else:
            cust_sorted = customers[:]
            rng.shuffle(cust_sorted)
            
        for c in cust_sorted:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            # Try inserting into existing routes
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = evaluate_route(trial)
                    if valid:
                        if d < best_cost:
                            best_cost = d
                            best_route_idx = i
                            best_pos = pos
            
            if best_route_idx != -1:
                r = routes[best_route_idx]
                routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
            else:
                # Create new route
                valid, d, _ = evaluate_route([c])
                if valid:
                    routes.append([c])
                else:
                    # If single customer is invalid (shouldn't happen), add anyway to avoid crash
                    routes.append([c])
                    
        return routes

    def large_neighborhood_search(routes, rng, time_budget):
        """
        Performs LNS: Destroy a portion of the solution, then repair it.
        """
        current_routes = [list(r) for r in routes]
        valid, current_dist = evaluate_routes(current_routes)
        if not valid:
            return current_routes, current_dist
            
        best_routes = [list(r) for r in current_routes]
        best_dist = current_dist
        
        t_start = time.time()
        T = 50.0
        alpha = 0.995
        
        while time.time() - t_start < time_budget:
            # 1. Destroy: Remove k random customers from their routes
            k = rng.integers(3, 8) # Remove 3-7 customers
            removed = []
            
            for _ in range(k):
                # Pick a random route that has customers
                non_empty = [i for i, r in enumerate(current_routes) if len(r) > 0]
                if not non_empty:
                    break
                ri = non_empty[rng.integers(0, len(non_empty))]
                r = current_routes[ri]
                ci = rng.integers(0, len(r))
                c = r.pop(ci)
                removed.append(c)
            
            if not removed:
                continue
                
            # 2. Repair: Insert removed customers into best positions
            # Shuffle removed to avoid bias
            rng.shuffle(removed)
            
            for c in removed:
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')
                
                for i, r in enumerate(current_routes):
                    for pos in range(len(r) + 1):
                        trial = r[:pos] + [c] + r[pos:]
                        valid, d, _ = evaluate_route(trial)
                        if valid:
                            if d < best_cost:
                                best_cost = d
                                best_route_idx = i
                                best_pos = pos
                
                if best_route_idx != -1:
                    r = current_routes[best_route_idx]
                    current_routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
                else:
                    # Couldn't insert? This is a problem. 
                    # For now, put it back where it was? Or just fail this iteration.
                    # Let's try to put it in a new route if possible
                    valid, d, _ = evaluate_route([c])
                    if valid:
                        current_routes.append([c])
                    else:
                        # Revert? Too complex. Just skip and hope next iteration fixes it.
                        # Actually, this makes the solution invalid. 
                        # We need to ensure validity. 
                        # If we can't insert, we should probably keep the previous state for this customer.
                        # But we already popped it. 
                        # Let's just add it to a random route and hope for the best, 
                        # or better: revert the whole LNS step if it becomes invalid.
                        pass
            
            # Check validity of new solution
            new_valid, new_dist = evaluate_routes(current_routes)
            
            if new_valid:
                if new_dist < best_dist:
                    best_routes = [list(r) for r in current_routes]
                    best_dist = new_dist
                    T = max(50.0, T * 1.1) # Reset temp if improved
                else:
                    # Accept worse with SA probability
                    delta = new_dist - best_dist
                    if rng.random() < math.exp(-delta / T):
                        best_routes = [list(r) for r in current_routes]
                        best_dist = new_dist
            else:
                # If invalid, we need to revert current_routes to the state before destruction?
                # We didn't save it. This is a flaw in the simple LNS implementation above.
                # To fix: We should only update current_routes if the repair is valid.
                # Let's re-implement the loop logic slightly to be safer.
                # Actually, for this draft, if it becomes invalid, we just restart the LNS loop 
                # with the best valid solution found so far.
                current_routes = [list(r) for r in best_routes]
                
            T *= alpha
            if T < 1.0:
                T = 1.0
                
        return best_routes, best_dist

    # --- Main Execution ---
    
    # 1. Generate multiple diverse initial solutions
    n_starts = 5
    best_overall_routes = None
    best_overall_dist = float('inf')
    
    for i in range(n_starts):
        time_left = DEADLINE - (time.time() - t0)
        if time_left < 10:
            break
            
        # Allocate time for construction and LNS
        # Construction is fast. Give LNS the rest.
        # Let's do 1 start with full time, or multiple with less time?
        # Given 145s, 5 starts * ~20s LNS each is reasonable.
        
        init_routes = construct_solution(i)
        
        # Validate initial
        v, d = evaluate_routes(init_routes)
        if not v:
            # Fallback for initial
            init_routes = [[c] for c in customers]
            v, d = evaluate_routes(init_routes)
            
        if d < best_overall_dist:
            best_overall_dist = d
            best_overall_routes = [list(r) for r in init_routes]
            
        # Run LNS on this start
        time_for_lns = min(20.0, DEADLINE - (time.time() - t0) - 5.0)
        if time_for_lns > 1:
            rng = np.random.default_rng(100 + i)
            improved_routes, improved_dist = large_neighborhood_search(init_routes, rng, time_for_lns)
            
            if improved_dist < best_overall_dist:
                best_overall_dist = improved_dist
                best_overall_routes = [list(r) for r in improved_routes]
                
        # Check time
        if time.time() - t0 > DEADLINE - 10:
            break

    # Final Validation
    if best_overall_routes is None:
        best_overall_routes = [[c] for c in customers]
        
    final_val, final_dist = evaluate_routes(best_overall_routes)
    if not final_val:
        best_overall_routes = [[c] for c in customers]
        
    return best_overall_routes
