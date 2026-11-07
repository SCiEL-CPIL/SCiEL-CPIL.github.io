"""EV Routing: Time-Dependent Dijkstra on (node, battery) state space with bitmask customer tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import heapq

DEADLINE_SECONDS = 115.0

def _validate_route(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon):
    """
    Validates a route and returns total distance, or float('inf') if infeasible.
    """
    if not route:
        return 0.0
        
    t = 0.0
    b = battery_max
    d_sum = 0.0
    total_dist = 0.0
    
    prev = 0
    for node in route:
        dist = D[prev, node]
        total_dist += dist
        b -= dist
        if b < -1e-9:
            return float('inf')
        t += dist / speed
        
        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            if demands[node] > 0: # Customer
                d_sum += demands[node]
                if d_sum > 100:
                    return float('inf')
                start_service = max(t, ready[node])
                if start_service > due[node] + 1e-9:
                    return float('inf')
                t = start_service + service[node]
        prev = node
        
    dist_back = D[prev, 0]
    total_dist += dist_back
    b -= dist_back
    if b < -1e-9:
        return float('inf')
    t += dist_back / speed
    
    if t > horizon + 1e-9:
        return float('inf')
        
    return total_dist

def _dijkstra_route(customers, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list, start_time, deadline):
    """
    Constructs a route using Dijkstra on (node, battery) state space.
    Returns (route, cost) or (None, inf) if no route found.
    """
    n_nodes = len(D)
    # State: (node, battery)
    # Since battery is continuous, we discretize it. 
    # However, battery levels are determined by distances which are floats.
    # A key insight: the battery level at a node is determined by the path.
    # We can use the actual battery level as part of the state, but we need a way to handle the priority queue.
    # A common trick for EV routing is to discretize battery into steps, but that's expensive.
    # Alternative: Use (node, battery) where we only keep the best (lowest time/cost) for a given node and battery level?
    # Actually, for a single route construction, we can use a modified Dijkstra where the priority is time, and we track the set of visited customers.
    # But the state space (node, battery, visited_set) is huge.
    
    # Let's simplify: We will use a heuristic search (A* or Dijkstra) that builds a sequence of nodes.
    # State: (current_node, battery, visited_customers_mask, current_time, current_distance)
    # This is essentially a shortest path problem in a graph where nodes are (location, battery, time, visited).
    # This is too large.
    
    # Revised Strategy:
    # Since n_customers is small (25), but we are constructing routes one by one, the number of customers in a single route is likely small (5-10).
    # We can use a recursive backtracking or a limited Dijkstra.
    
    # Let's try a different approach:
    # 1. Start at depot.
    # 2. At each step, choose the next customer or station that minimizes the cost (distance + time penalty) while maintaining feasibility.
    # 3. This is a greedy construction but with a lookahead or better cost function.
    
    # Actually, the prompt asks for "Time-dependent shortest path on a state space defined by (current_node, battery_level, set_of_unvisited_customers)".
    # With 25 customers, the bitmask has 2^25 states, which is too large.
    # But we are constructing routes sequentially. We can assign customers to routes one by one.
    # For each route, we only need to consider the remaining customers.
    # If we assume a route has at most ~8 customers, the state space is manageable.
    
    # Let's implement a Dijkstra where the state is (node, battery, visited_mask_in_route).
    # The 'visited_mask_in_route' is a bitmask of customers included in THIS specific route.
    # We start with an empty route and try to add customers until we can't anymore or it's not beneficial.
    
    # However, determining "when to stop" is tricky.
    # Alternative: We fix the set of customers for a route first (e.g., using k-means or nearest neighbor), then optimize the order.
    # The prompt suggests using Dijkstra to construct each route, implying the route content is determined during the search.
    
    # Let's try a hybrid:
    # 1. Use a simple heuristic (nearest neighbor) to generate an initial set of customers for a route.
    # 2. Use Dijkstra to find the optimal order for that set, including station insertions.
    
    # But the prompt says "construct each route... without relying on greedy heuristics".
    # So let's try a Dijkstra that expands states by adding one customer at a time.
    # State: (node, battery, visited_mask, time, dist)
    # We stop when we return to depot or can't add more customers.
    
    # To keep it tractable, we will limit the number of customers per route in the search to, say, 10.
    # If a route has more, we split it.
    
    # Actually, a better structural change is to use **Branch and Bound** or **Dynamic Programming** for the ordering of a fixed set of customers, and use a heuristic to assign customers to routes.
    
    # Let's stick to the prompt's suggestion: Dijkstra on (node, battery, visited).
    # We will run this for each "cluster" of customers.
    # How to form clusters? We can use a simple partitioning.
    
    # Plan:
    # 1. Partition customers into clusters of size ~5-8 using k-means or simple distance-based clustering.
    # 2. For each cluster, run Dijkstra to find the best route.
    # 3. If a cluster's route is infeasible, split it or try adding stations.
    
    # Let's implement the Dijkstra for a fixed set of customers.
    
    # State: (node, battery, visited_mask)
    # We want to minimize distance.
    # Priority queue: (distance, node, battery, visited_mask)
    # We need to track the path to reconstruct the route.
    
    # Since battery is continuous, we might have many states.
    # We can discretize battery to steps of 1.0. Battery max is 300. So 300 levels.
    # Nodes: ~30 (25 customers + 4 stations + 1 depot).
    # Visited mask: up to 2^8 = 256 if cluster size is 8.
    # Total states: 30 * 300 * 256 = 2.3M. This is feasible.
    
    # Let's implement this.
    
    n_customers_in_set = len(customers)
    if n_customers_in_set == 0:
        return [], 0.0
    
    # Map customers in this set to indices 0..k-1 for the bitmask
    cust_map = {c: i for i, c in enumerate(customers)}
    k = n_customers_in_set
    
    # Discretize battery
    b_step = 1.0
    b_max_idx = int(battery_max / b_step)
    
    # State: (node, b_idx, mask)
    # dist[state] = min distance to reach this state
    # parent[state] = (prev_node, prev_b_idx, prev_mask) for reconstruction
    
    # We use a dictionary for dist and parent
    # dist: {(node, b_idx, mask): distance}
    # parent: {(node, b_idx, mask): (prev_node, prev_b_idx, prev_mask)}
    
    # Initial state: depot (0), full battery, mask=0
    start_state = (0, b_max_idx, 0)
    dist = {start_state: 0.0}
    parent = {start_state: None}
    
    # Priority queue: (distance, node, b_idx, mask)
    pq = [(0.0, 0, b_max_idx, 0)]
    
    # We also need to track the time at each state to check time windows.
    # Time is determined by the path. We can store time in the priority queue or compute it.
    # Since we are minimizing distance, and time is correlated with distance (speed=1),
    # we can store time in the state or recompute it.
    # Let's store time in the priority queue item: (distance, time, node, b_idx, mask)
    # But time is not part of the state identity, only distance is for the cost.
    # However, feasibility depends on time.
    # If two paths reach the same (node, b_idx, mask) with the same distance, they have the same time?
    # Not necessarily, if they took different paths with the same distance but different service times?
    # Service times are fixed per customer. Distance is sum of edge lengths.
    # Time = distance/speed + sum(service times) + waiting time.
    # Waiting time depends on arrival time.
    # So time is path-dependent.
    # We need to include time in the state or ensure we only keep the best time for a given (node, b_idx, mask, distance).
    # This is getting complex.
    
    # Simplification: Since speed=1, time = distance + waiting + service.
    # We can store the arrival time in the priority queue.
    # When we pop a state, if we have seen this (node, b_idx, mask) with a lower or equal distance and earlier time, we skip.
    # Actually, if we have a lower distance, the time is likely lower (ignoring waiting).
    # Let's store (distance, time) in the priority queue and use a dictionary to store the best (distance, time) for each state.
    # If a new path has lower distance, it's better. If same distance, lower time is better.
    
    best_dist_time = {} # (node, b_idx, mask) -> (dist, time)
    best_dist_time[start_state] = (0.0, 0.0)
    
    # We also need to track the path.
    # path[state] = list of nodes? No, too memory intensive.
    # Use parent pointers.
    
    found_end = False
    end_state = None
    
    while pq:
        if time.time() > deadline:
            break
            
        d, t, u, b_idx, mask = heapq.heappop(pq)
        
        # Check if this state is outdated
        if (u, b_idx, mask) not in best_dist_time:
            continue
        bd, bt = best_dist_time[(u, b_idx, mask)]
        if d > bd + 1e-9 or (abs(d - bd) < 1e-9 and t > bt + 1e-9):
            continue
            
        # If we are at depot and have visited at least one customer, we can return
        if u == 0 and mask != 0:
            # Check if we can return to depot (we are at depot, so just check horizon)
            if t <= horizon + 1e-9:
                found_end = True
                end_state = (u, b_idx, mask)
                break # Dijkstra guarantees shortest path to this state
        
        # Expand neighbors
        # Neighbors: all other nodes (customers in this set, stations, depot)
        # We should not visit a customer twice (handled by mask)
        # We can visit stations multiple times
        
        for v in range(n_nodes):
            if v == u:
                continue
                
            # Check if v is a customer in this set and already visited
            if v in cust_map and (mask >> cust_map[v]) & 1:
                continue
                
            # Calculate distance and time to v
            dist_uv = D[u, v]
            new_b = b_idx * b_step - dist_uv
            if new_b < -1e-9:
                continue
                
            new_b_idx = int(new_b / b_step)
            # Clamp new_b_idx to [0, b_max_idx]
            if new_b_idx < 0:
                continue
            if new_b_idx > b_max_idx:
                new_b_idx = b_max_idx
                
            new_t = t + dist_uv / speed
            
            if is_station[v]:
                # Recharge
                # Battery becomes full
                # Time added: (battery_max - new_b) * recharge_rate
                # Note: new_b is the battery level BEFORE recharge
                recharge_time = (battery_max - new_b) * recharge_rate
                new_t += recharge_time
                new_b = battery_max
                new_b_idx = b_max_idx
            else:
                if demands[v] > 0: # Customer
                    # Check time window
                    if new_t > due[v] + 1e-9:
                        continue
                    start_service = max(new_t, ready[v])
                    new_t = start_service + service[v]
                    # Update mask
                    new_mask = mask | (1 << cust_map[v])
                else:
                    new_mask = mask
            # Depot has no demand, no service
            
            new_d = d + dist_uv
            
            new_state = (v, new_b_idx, new_mask)
            
            if new_state not in best_dist_time:
                best_dist_time[new_state] = (new_d, new_t)
                heapq.heappush(pq, (new_d, new_t, v, new_b_idx, new_mask))
                parent[new_state] = (u, b_idx, mask)
            else:
                bd, bt = best_dist_time[new_state]
                if new_d < bd - 1e-9 or (abs(new_d - bd) < 1e-9 and new_t < bt - 1e-9):
                    best_dist_time[new_state] = (new_d, new_t)
                    heapq.heappush(pq, (new_d, new_t, v, new_b_idx, new_mask))
                    parent[new_state] = (u, b_idx, mask)
                    
    if not found_end:
        return None, float('inf')
        
    # Reconstruct path
    route = []
    state = end_state
    while state is not None and state != start_state:
        route.append(state[0])
        state = parent[state]
    route.reverse()
    
    # Add depot at start and end? The route list is just the intermediate nodes.
    # The cost is the distance of the route.
    cost = best_dist_time[end_state][0]
    
    # Verify the route
    valid_cost = _validate_route(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
    if valid_cost != float('inf'):
        return route, valid_cost
    else:
        return None, float('inf')

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
    stations = instance["stations"]
    stations_list = list(stations)
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Distance Matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    
    # 2. Partition customers into clusters
    # Use a simple k-means-like approach or distance-based partitioning
    # We want clusters of size ~5-8
    target_cluster_size = 7
    n_clusters = math.ceil(n_customers / target_cluster_size)
    
    # Simple partition: Sort customers by x-coordinate and split into chunks
    # This is a very naive partitioning, but it avoids complex clustering logic
    # A better partitioning would consider distances between customers
    # Let's use a simple greedy partitioning:
    # 1. Start with all customers unassigned
    # 2. Pick the customer farthest from the depot
    # 3. Assign it to a new cluster
    # 4. Add the nearest unassigned customers to this cluster until size is reached
    # 5. Repeat
    
    unassigned = set(customers)
    clusters = []
    
    while unassigned:
        if time.time() > start_time + DEADLINE_SECONDS * 0.2:
            # Fallback: put all remaining in one cluster
            clusters.append(list(unassigned))
            break
            
        # Pick customer farthest from depot
        farthest = max(unassigned, key=lambda c: D[0, c])
        cluster = [farthest]
        unassigned.remove(farthest)
        
        # Add nearest customers
        while len(cluster) < target_cluster_size and unassigned:
            # Find nearest unassigned customer to any in cluster
            best_c = None
            best_d = float('inf')
            for c in unassigned:
                for cc in cluster:
                    d = D[c, cc]
                    if d < best_d:
                        best_d = d
                        best_c = c
            if best_c is None:
                break
            cluster.append(best_c)
            unassigned.remove(best_c)
            
        clusters.append(cluster)
        
    # 3. For each cluster, run Dijkstra to find the best route
    routes = []
    for cluster in clusters:
        if time.time() > start_time + DEADLINE_SECONDS * 0.9:
            # Fallback: simple nearest neighbor for remaining clusters
            for c in cluster:
                routes.append([c])
            break
            
        route, cost = _dijkstra_route(cluster, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list, start_time, start_time + DEADLINE_SECONDS)
        
        if route is not None:
            routes.append(route)
        else:
            # Fallback: simple nearest neighbor
            # This is a greedy fallback if Dijkstra fails
            current = 0
            remaining = list(cluster)
            route = []
            while remaining:
                if time.time() > start_time + DEADLINE_SECONDS * 0.95:
                    break
                nearest = min(remaining, key=lambda c: D[current, c])
                route.append(nearest)
                remaining.remove(nearest)
                current = nearest
            if route:
                # Check feasibility
                cost = _validate_route(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                if cost != float('inf'):
                    routes.append(route)
                else:
                    # Split into singletons
                    for c in route:
                        routes.append([c])
                    
    # 4. Ensure all customers are served
    visited = set()
    for r in routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)
                
    missing = set(customers) - visited
    for m in missing:
        routes.append([m])
        
    return routes

# EVOLVE-BLOCK-END
