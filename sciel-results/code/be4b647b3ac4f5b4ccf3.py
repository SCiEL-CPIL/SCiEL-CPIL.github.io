import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0

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

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))

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
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Construct Initial Solution using Regret-2 Insertion ---
    # This is a stronger baseline than simple due-date insertion.
    
    routes = []
    # We maintain a list of (route, current_load, current_distance) implicitly by re-evaluating or tracking.
    # For simplicity and robustness in construction, we re-evaluate on insertion.
    # To speed up, we can track load and distance incrementally, but full check is safer for time windows/battery.
    
    cust_sorted = sorted(customers, key=lambda c: due[c])
    
    # Pre-compute regret values? No, dynamic regret is better.
    # For each customer, find the 2 best insertion positions (cost delta) across all routes.
    # If the gap between 1st and 2nd is large, insert into 1st.
    
    remaining = set(cust_sorted)
    
    while remaining:
        if time.time() > DEADLINE - 5: break # Safety
        
        best_regret = -1
        best_cust = None
        best_route_idx = -1
        best_pos = -1
        
        for c in list(remaining):
            best_cost1 = float('inf')
            best_cost2 = float('inf')
            best_r1, best_p1 = -1, -1
            best_r2, best_p2 = -1, -1
            
            # Check all existing routes
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = check_route(trial)
                    if valid:
                        # Approximate cost delta: 
                        # If we don't have old distance, we can't easily get delta without storing it.
                        # Let's store distances in a parallel list for speed?
                        # Or just use total distance as a proxy? 
                        # Actually, minimizing total distance of the route is a good heuristic for "tightness".
                        # But delta is better.
                        # Let's just use the total distance of the new route. It's O(N^2) but N is small (25).
                        if d < best_cost1:
                            if best_cost1 < float('inf'):
                                best_cost2 = best_cost1
                                best_r2, best_p2 = best_r1, best_p1
                            best_cost1 = d
                            best_r1, best_p1 = i, pos
                        elif d < best_cost2:
                            best_cost2 = d
                            best_r2, best_p2 = i, pos
            
            # Check creating a new route
            valid, d_new, _ = check_route([c])
            if valid:
                # Cost of new route is just its distance. 
                # We want to minimize total system distance.
                # Adding a new route adds d_new to the total.
                # So cost delta is d_new.
                # Let's compare d_new against best_cost1? 
                # No, best_cost1 is the total distance of the modified route.
                # This comparison is apples to oranges.
                
                # Let's stick to a simpler heuristic for construction to ensure speed and validity:
                # Just insert into the route where it fits with minimum added distance.
                # We need to track old distances.
                
                pass # We'll rewrite construction to be simpler and faster: Nearest Neighbor + Insertion
            
            # Regret = best_cost2 - best_cost1 (if both exist)
            # If only one exists, regret is high (must insert here or new route)
            if best_cost1 < float('inf'):
                if best_cost2 < float('inf'):
                    regret = best_cost2 - best_cost1
                else:
                    regret = 1000.0 # High regret if only one option
            else:
                regret = 0 # Cannot insert anywhere existing, must create new route
            
            # Actually, the "new route" option is always available if valid.
            # Let's simplify: Just find the best single insertion (min total distance of route).
            # If no insertion fits, create new route.
            # This is O(N^3) construction which is fine for N=25.
            
            if best_cost1 < float('inf') and best_cost1 < d_new: # Prefer inserting if it keeps route dist lower? No, total dist matters.
                 # This logic is getting complicated. Let's use a standard "Insert into best feasible position minimizing distance increase".
                 pass
        
        # SIMPLIFIED CONSTRUCTION:
        # 1. Start with empty routes.
        # 2. For each customer (sorted by due date):
        #    - Find the best position in any existing route (min distance increase).
        #    - If no position fits, create a new route.
        
        # Let's restart construction with this simpler, faster, and robust method.
        routes = []
        # We need to track current distance of each route to calculate delta efficiently.
        # route_dists[i] = distance of routes[i]
        
        for c in cust_sorted:
            best_delta = float('inf')
            best_r = -1
            best_p = -1
            
            for i, r in enumerate(routes):
                # We need the current distance of r. 
                # To avoid recalculating, let's just recalculate for simplicity in this small N.
                # Or store it.
                # Let's store it in a list `cur_dists`.
                pass
            
            # Since N is small, let's just do the full check for each position.
            # It's 25 customers * ~10 routes * ~10 positions * O(10) check = 25k ops. Very fast.
            
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = check_route(trial)
                    if valid:
                        # We need old distance to compute delta.
                        # Let's just use d (total distance of new route) as the cost.
                        # Minimizing total distance of the route is a reasonable proxy.
                        if d < best_delta:
                            best_delta = d
                            best_r = i
                            best_p = pos
            
            if best_r != -1:
                r = routes[best_r]
                routes[best_r] = r[:best_p] + [c] + r[best_p:]
            else:
                # Create new route
                valid, d, _ = check_route([c])
                if valid:
                    routes.append([c])
                else:
                    # Should not happen
                    routes.append([c])
        
        break # Exit the while loop, construction done

    # --- LNS (Large Neighborhood Search) ---
    
    best_routes = [list(r) for r in routes]
    best_val, best_dist = evaluate_routes(best_routes)
    if not best_val:
        best_routes = [[c] for c in customers]
        best_val, best_dist = evaluate_routes(best_routes)
    
    rng = np.random.default_rng(42)
    
    # LNS Parameters
    destroy_size = 5 # Remove 5 customers at a time
    
    current_routes = [list(r) for r in best_routes]
    current_dist = best_dist
    
    while time.time() < DEADLINE:
        # 1. Destroy: Remove K random customers from their routes
        removed_customers = []
        temp_routes = [list(r) for r in current_routes]
        
        # Find indices of customers in temp_routes
        # We need to remove them.
        # To make it efficient, we can pick random customers from the set of all customers.
        
        # Get list of all customers currently in routes
        all_in_routes = []
        for r in temp_routes:
            all_in_routes.extend(r)
        
        if not all_in_routes:
            break
            
        # Pick destroy_size random customers
        k = min(destroy_size, len(all_in_routes))
        to_remove = set(rng.choice(all_in_routes, size=k, replace=False))
        
        # Remove them
        for i in range(len(temp_routes)):
            new_r = []
            for node in temp_routes[i]:
                if node in to_remove:
                    removed_customers.append(node)
                else:
                    new_r.append(node)
            temp_routes[i] = new_r
        
        # Remove empty routes
        temp_routes = [r for r in temp_routes if len(r) > 0]
        
        # 2. Repair: Insert removed customers back using greedy regret
        # Sort removed customers by due date? Or random? Due date is safer.
        removed_customers = sorted(removed_customers, key=lambda c: due[c])
        
        for c in removed_customers:
            best_cost = float('inf')
            best_r_idx = -1
            best_pos = -1
            
            # Try inserting into existing temp_routes
            for i, r in enumerate(temp_routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = check_route(trial)
                    if valid:
                        if d < best_cost:
                            best_cost = d
                            best_r_idx = i
                            best_pos = pos
            
            # Try creating a new route
            valid, d_new, _ = check_route([c])
            if valid and d_new < best_cost:
                best_cost = d_new
                best_r_idx = -2 # Special marker for new route
                best_pos = 0
            
            if best_r_idx == -2:
                temp_routes.append([c])
            elif best_r_idx != -1:
                r = temp_routes[best_r_idx]
                temp_routes[best_r_idx] = r[:best_pos] + [c] + r[best_pos:]
            else:
                # Couldn't insert? This shouldn't happen if initial solution was valid and we only removed.
                # But if time windows are tight, maybe.
                # Fallback: append to end of first route? Or new route?
                # If we can't insert, the solution is invalid. 
                # Let's just put it in a new route if valid, else skip (invalid).
                if valid:
                    temp_routes.append([c])
                else:
                    # Mark as invalid
                    temp_routes.append([c]) 
                    break
        
        # 3. Evaluate
        new_val, new_dist = evaluate_routes(temp_routes)
        
        if new_val and new_dist < current_dist:
            current_routes = temp_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_routes = [list(r) for r in current_routes]
                best_dist = current_dist
        elif not new_val:
            # If repair failed, revert to current_routes (which is valid)
            pass

    # Final Validation
    final_val, final_dist = evaluate_routes(best_routes)
    if not final_val:
        best_routes = [[c] for c in customers]
        
    return best_routes
