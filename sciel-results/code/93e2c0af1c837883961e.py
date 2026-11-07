"""Reformulated EV routing: Time-dependent Dijkstra on (node, mask) state space with Pareto labels."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq
import math

def plan(instance):
    n_cust = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    n_nodes = len(coords)
    dist_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_mat[i][j] = math.sqrt(dx*dx + dy*dy)
            
    customers = list(range(1, n_cust + 1))
    
    # Helper to check if a set of customers can fit in capacity
    def check_capacity(cust_list):
        return sum(demands[c] for c in cust_list) <= capacity

    # Dijkstra-like solver for a specific set of customers
    # State: (current_node, unvisited_mask)
    # Label: (distance, time, battery, path)
    def solve_route(cust_set):
        if not cust_set:
            return []
            
        # Check capacity
        if not check_capacity(cust_set):
            return None
            
        cust_list = sorted(cust_set)
        k = len(cust_list)
        cust_to_bit = {c: i for i, c in enumerate(cust_list)}
        initial_mask = (1 << k) - 1
        
        # Priority Queue: (distance, time, battery, node, mask, path)
        pq = []
        heapq.heappush(pq, (0.0, 0.0, battery_max, 0, initial_mask, [0]))
        
        # best_labels[state] = list of (dist, time, battery)
        best_labels = {}
        
        while pq:
            d, t, b, node, mask, path = heapq.heappop(pq)
            
            state = (node, mask)
            # Check if this label is dominated
            if state in best_labels:
                dominated = False
                for (d_p, t_p, b_p) in best_labels[state]:
                    if d_p <= d and t_p <= t and b_p >= b:
                        dominated = True
                        break
                if dominated:
                    continue
            
            # If we are at depot and all visited, we found a solution
            if node == 0 and mask == 0:
                return path[1:] # Exclude starting depot
            
            # Generate neighbors
            
            # 1. Try unvisited customers
            for c in cust_list:
                bit = 1 << cust_to_bit[c]
                if mask & bit:
                    new_mask = mask & ~bit
                    dist = dist_mat[node][c]
                    if b < dist:
                        continue
                    new_b = b - dist
                    new_t = t + dist / speed
                    # Time window
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c]:
                        continue
                    new_t += service[c]
                    if new_t > horizon:
                        continue
                    
                    new_d = d + dist
                    new_path = path + [c]
                    
                    # Check if new label is dominated at (c, new_mask)
                    n_state = (c, new_mask)
                    is_dom = False
                    if n_state in best_labels:
                        for (d_p, t_p, b_p) in best_labels[n_state]:
                            if d_p <= new_d and t_p <= new_t and b_p >= new_b:
                                is_dom = True
                                break
                    if not is_dom:
                        # Remove labels dominated by the new one
                        if n_state in best_labels:
                            best_labels[n_state] = [
                                (d_p, t_p, b_p) for (d_p, t_p, b_p) in best_labels[n_state]
                                if not (new_d <= d_p and new_t <= t_p and new_b >= b_p)
                            ]
                        else:
                            best_labels[n_state] = []
                        
                        best_labels[n_state].append((new_d, new_t, new_b))
                        heapq.heappush(pq, (new_d, new_t, new_b, c, new_mask, new_path))

            # 2. Try stations
            for s in stations:
                dist = dist_mat[node][s]
                if b < dist:
                    continue
                b_after_travel = b - dist
                t_after_travel = t + dist / speed
                # Recharge
                recharge_time = (battery_max - b_after_travel) * recharge_rate
                new_t = t_after_travel + recharge_time
                if new_t > horizon:
                    continue
                new_b = battery_max
                new_d = d + dist
                new_path = path + [s]
                
                # State is (s, mask) - mask unchanged
                n_state = (s, mask)
                is_dom = False
                if n_state in best_labels:
                    for (d_p, t_p, b_p) in best_labels[n_state]:
                        if d_p <= new_d and t_p <= new_t and b_p >= new_b:
                            is_dom = True
                            break
                if not is_dom:
                    if n_state in best_labels:
                        best_labels[n_state] = [
                            (d_p, t_p, b_p) for (d_p, t_p, b_p) in best_labels[n_state]
                            if not (new_d <= d_p and new_t <= t_p and new_b >= b_p)
                        ]
                    else:
                        best_labels[n_state] = []
                    
                    best_labels[n_state].append((new_d, new_t, new_b))
                    heapq.heappush(pq, (new_d, new_t, new_b, s, mask, new_path))

            # 3. Try returning to depot (only if all visited)
            if mask == 0:
                dist = dist_mat[node][0]
                if b < dist:
                    continue
                new_b = b - dist
                new_t = t + dist / speed
                if new_t > horizon:
                    continue
                new_d = d + dist
                new_path = path + [0]
                
                n_state = (0, 0)
                is_dom = False
                if n_state in best_labels:
                    for (d_p, t_p, b_p) in best_labels[n_state]:
                        if d_p <= new_d and t_p <= new_t and b_p >= new_b:
                            is_dom = True
                            break
                if not is_dom:
                    if n_state in best_labels:
                        best_labels[n_state] = [
                            (d_p, t_p, b_p) for (d_p, t_p, b_p) in best_labels[n_state]
                            if not (new_d <= d_p and new_t <= t_p and new_b >= b_p)
                        ]
                    else:
                        best_labels[n_state] = []
                    
                    best_labels[n_state].append((new_d, new_t, new_b))
                    heapq.heappush(pq, (new_d, new_t, new_b, 0, 0, new_path))

        return None

    # Partition customers into clusters
    def get_angle(c):
        dx = coords[c][0] - coords[0][0]
        dy = coords[c][1] - coords[0][1]
        return math.atan2(dy, dx)
    
    sorted_custs = sorted(customers, key=get_angle)
    
    # We'll use a recursive splitting approach starting with all customers
    # But to speed up, we can start with angular sectors
    n_clusters = 6
    angle_step = 360.0 / n_clusters
    sectors = [[] for _ in range(n_clusters)]
    for c in customers:
        ang = get_angle(c)
        ang_deg = math.degrees(ang)
        if ang_deg < 0:
            ang_deg += 360
        sector_idx = int(ang_deg / angle_step)
        if sector_idx >= n_clusters:
            sector_idx = n_clusters - 1
        sectors[sector_idx].append(c)
        
    final_routes = []
    
    def solve_recursive_routes(cust_list, depth=0):
        if not cust_list:
            return []
        if depth > 8: # Prevent infinite recursion
            res = []
            for c in cust_list:
                r = solve_route([c])
                if r:
                    res.append(r)
            return res
        
        route = solve_route(cust_list)
        if route:
            return [route]
        
        # Split in half
        mid = len(cust_list) // 2
        if mid == 0:
             # Single customer failed, return empty (should not happen)
             return []
        left = cust_list[:mid]
        right = cust_list[mid:]
        
        r1 = solve_recursive_routes(left, depth+1)
        r2 = solve_recursive_routes(right, depth+1)
        return r1 + r2

    for sec in sectors:
        if sec:
            final_routes.extend(solve_recursive_routes(sec, 0))
            
    # Check if all customers are covered
    covered = set()
    for r in final_routes:
        for node in r:
            if 1 <= node <= n_cust:
                covered.add(node)
                
    missing = set(customers) - covered
    for c in missing:
        r = solve_route([c])
        if r:
            final_routes.append(r)
        else:
            final_routes.append([c])

    return final_routes
# EVOLVE-BLOCK-END
