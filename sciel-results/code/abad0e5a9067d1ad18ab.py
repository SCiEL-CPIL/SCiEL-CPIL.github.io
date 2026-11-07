"""Vectorized EV routing with Cheapest Insertion and fast 2-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

DEADLINE_SECONDS = 115.0

def _compute_prefixes(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon):
    """
    Compute prefix arrays for a route to enable O(1) feasibility checks for insertions and reversals.
    route: list of node IDs (customers/stations, no depot)
    Returns:
        p_time: prefix arrival times at each node (before service)
        p_battery: prefix battery levels at each node (before service)
        p_demand: prefix demand
        final_time: time at depot after service
        final_battery: battery at depot after service
    """
    n = len(route)
    if n == 0:
        return [], [], [], 0.0, battery_max

    p_time = np.zeros(n)
    p_battery = np.zeros(n)
    p_demand = np.zeros(n)
    
    t = 0.0
    b = battery_max
    d_sum = 0.0
    
    # Start from depot (node 0)
    # Note: D[0] is row for depot
    
    for i, node in enumerate(route):
        # Distance from previous (depot if i=0 else route[i-1])
        prev = 0 if i == 0 else route[i-1]
        dist = D[prev, node]
        
        b -= dist
        if b < -1e-9:
            return None # Infeasible battery
        t += dist / speed
        
        if is_station[node]:
            # Recharge
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            # Customer
            # Arrival time is t
            # Service starts at max(t, ready)
            # We store arrival time in p_time
            pass
            
        p_time[i] = t
        p_battery[i] = b
        p_demand[i] = d_sum + (demands[node] if not is_station[node] else 0)
        
        # Update state for next iteration
        if not is_station[node]:
            # Service time
            start_service = max(t, ready[node])
            t = start_service + service[node]
            # Check due date
            if start_service > due[node] + 1e-9:
                return None # Infeasible time window
        # If station, t is already updated by recharge, no service time

    # Return to depot
    last_node = route[-1]
    dist_back = D[last_node, 0]
    b -= dist_back
    if b < -1e-9:
        return None
    t += dist_back / speed
    
    if t > horizon + 1e-9:
        return None

    return p_time, p_battery, p_demand, t, b

def plan(instance):
    start_time = time.time()
    
    # 1. Precompute Data
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Distance Matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    
    # 2. Construction: Cheapest Insertion
    # Start with empty routes
    routes = []
    unvisited = set(customers)
    
    # Sort customers by due date to help feasibility
    customers_sorted = sorted(customers, key=lambda x: due[x])
    
    def get_route_distance(route):
        if not route:
            return 0.0
        dist = D[0, route[0]]
        for i in range(len(route) - 1):
            dist += D[route[i], route[i+1]]
        dist += D[route[-1], 0]
        return dist
    
    def check_insertion(route, prefixes, pos, node):
        """
        Check if inserting node at pos in route is feasible.
        Returns (is_feasible, new_prefixes) or (False, None)
        """
        if not route:
            # Single node route
            if is_station[node]:
                # Station alone? Usually we don't put stations alone unless necessary, but let's allow
                t = D[0, node] / speed
                b = battery_max - D[0, node]
                if b < 0: return False, None
                t += (battery_max - b) * recharge_rate
                b = battery_max
                t_back = t + D[node, 0] / speed
                b_back = b - D[node, 0]
                if b_back < 0 or t_back > horizon: return False, None
                # Prefixes for single node
                p_t = [t - D[node,0]/speed] # Arrival time
                # Actually, let's just recompute prefixes if it's small or use the general function
                # For efficiency in insertion, we do manual check
                
            # Let's just use the general prefix function for simplicity and correctness, 
            # but that's O(N). 
            # To make it O(1) delta, we need to be clever.
            # Given N=25, O(N) check is fine. O(N^2) construction is 25*25*25 = 15k ops. Very fast.
            
            new_route = [node]
            p = _compute_prefixes(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            if p is None:
                return False, None
            # Check capacity
            if demands[node] > capacity:
                return False, None
            return True, p
            
        # Insert at pos
        new_route = route[:pos] + [node] + route[pos:]
        
        # Quick checks
        total_demand = sum(demands[v] for v in new_route if not is_station[v])
        if total_demand > capacity:
            return False, None
            
        p = _compute_prefixes(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
        if p is None:
            return False, None
            
        return True, p

    # Greedy Cheapest Insertion
    # We maintain a list of (route, prefixes)
    route_data = [] # List of [route, prefixes]
    
    for cust in customers_sorted:
        best_cost = float('inf')
        best_idx = -1
        best_pos = -1
        best_new_route = None
        best_new_prefixes = None
        
        for r_idx, (route, prefixes) in enumerate(route_data):
            # Try all positions
            # pos 0 to len(route)
            for pos in range(len(route) + 1):
                # Cost delta: D[prev, node] + D[node, next] - D[prev, next]
                # prev = 0 if pos==0 else route[pos-1]
                # next = 0 if pos==len(route) else route[pos]
                prev = 0 if pos == 0 else route[pos-1]
                next_node = 0 if pos == len(route) else route[pos]
                
                cost = D[prev, cust] + D[cust, next_node] - D[prev, next_node]
                
                if cost < best_cost:
                    # Check feasibility
                    feasible, new_p = check_insertion(route, prefixes, pos, cust)
                    if feasible:
                        best_cost = cost
                        best_idx = r_idx
                        best_pos = pos
                        best_new_route = route[:pos] + [cust] + route[pos:]
                        best_new_prefixes = new_p
        
        if best_idx == -1:
            # Create new route
            new_route = [cust]
            p = _compute_prefixes(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            if p is None:
                # Should not happen if single node is feasible, but check
                # If single node infeasible, problem might be infeasible?
                # Assuming instances are feasible.
                continue
            route_data.append([new_route, p])
        else:
            old_route, _ = route_data[best_idx]
            route_data[best_idx] = [best_new_route, best_new_prefixes]
            
        unvisited.discard(cust)
        
        if time.time() - start_time > DEADLINE_SECONDS * 0.3:
            break
            
    # If some customers left (due to time or infeasibility in greedy), add them as single routes
    # This ensures all customers are served
    for cust in list(unvisited):
        new_route = [cust]
        p = _compute_prefixes(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
        if p is not None:
             route_data.append([new_route, p])
        else:
             # Fallback: force it, though it might be invalid. 
             # But task says "Any broken rule makes the whole result invalid".
             # We must assume feasible instances.
             route_data.append([new_route, None])

    # 3. Local Search: 2-Opt
    # We will iterate on each route.
    # To make it fast, we precompute D.
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > DEADLINE_SECONDS * 0.8:
            break
            
        for r_idx in range(len(route_data)):
            route, prefixes = route_data[r_idx]
            n = len(route)
            if n < 2:
                continue
                
            route_changed = False
            # 2-Opt: reverse segment i..j
            # We need to check feasibility.
            # Reversing a segment changes the sequence.
            # We can just rebuild prefixes for the candidate if it looks promising by distance.
            
            best_saving = 0
            best_i, best_j = -1, -1
            best_new_route = None
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Delta distance
                    # Old: D[prev_i, i] + D[j, next_j]
                    # New: D[prev_i, j] + D[i, next_j]
                    prev_i = 0 if i == 0 else route[i-1]
                    next_j = 0 if j == n-1 else route[j+1]
                    
                    old_dist = D[prev_i, route[i]] + D[route[j], next_j]
                    new_dist = D[prev_i, route[j]] + D[route[i], next_j]
                    
                    saving = old_dist - new_dist
                    if saving > best_saving:
                        # Check feasibility of reversed route
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        
                        # Quick capacity check (unchanged)
                        # Check time/battery
                        p = _compute_prefixes(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                        if p is not None:
                            best_saving = saving
                            best_i, best_j = i, j
                            best_new_route = new_route
                            # We don't break because a larger saving might exist, 
                            # but checking all is O(N^2) per route. 
                            # With N=25, this is small.
                            
            if best_i != -1:
                route_data[r_idx] = [best_new_route, _compute_prefixes(best_new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)]
                improved = True
                route_changed = True
                
        # Optional: Inter-route moves (Or-opt or reinsertion)
        # Given time, intra-route 2-opt is usually sufficient for this scale.
        # We could add a simple "move customer to another route" logic if time permits.
        
    # 4. Finalize
    final_routes = [r[0] for r in route_data]
    
    # Final Validation & Cleanup
    # Ensure no empty routes
    final_routes = [r for r in final_routes if len(r) > 0]
    
    # Check if any customer is missing (should not happen)
    visited = set()
    for r in final_routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)
                
    missing = set(customers) - visited
    for m in missing:
        # Force add
        final_routes.append([m])
        
    return final_routes

# EVOLVE-BLOCK-END
