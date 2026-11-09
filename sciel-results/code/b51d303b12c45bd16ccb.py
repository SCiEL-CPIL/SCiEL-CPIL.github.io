"""EV routing via K-Means clustering and battery-aware cheapest insertion."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import distance

def plan(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    depot = 0
    
    n_total = len(coords)
    dist = distance.cdist(coords, coords, metric='euclidean')
    
    def get_next_state(u, v, t_u, b_u):
        """
        Returns (t_v, b_v, dist_added) if feasible, else None.
        v is a customer.
        """
        # Try direct
        d = dist[u][v]
        if b_u >= d - 1e-9:
            t_arr = t_u + d / speed
            t_v = max(t_arr, ready[v])
            if t_v <= due[v] + 1e-9:
                t_serv = t_v + service[v]
                b_v = b_u - d
                return (t_serv, b_v, d)
        
        # Try via station
        best = None
        for s in stations:
            d_us = dist[u][s]
            if b_u >= d_us - 1e-9:
                t_arr_s = t_u + d_us / speed
                b_s = b_u - d_us
                t_recharge = (battery - b_s) * recharge_rate
                t_s = t_arr_s + t_recharge
                b_s_full = battery
                
                d_sv = dist[s][v]
                if b_s_full >= d_sv - 1e-9:
                    t_arr_v = t_s + d_sv / speed
                    t_v = max(t_arr_v, ready[v])
                    if t_v <= due[v] + 1e-9:
                        t_serv = t_v + service[v]
                        b_v = b_s_full - d_sv
                        total_d = d_us + d_sv
                        if best is None or total_d < best[2]:
                            best = (t_serv, b_v, total_d)
        
        return best

    def check_return(last_node, t_last, b_last):
        """
        Returns distance if return to depot is feasible, else None.
        """
        d_ret = dist[last_node][depot]
        if b_last >= d_ret - 1e-9:
            t_ret = t_last + d_ret / speed
            if t_ret <= horizon + 1e-9:
                return d_ret
        
        # Try via station
        best = None
        for s in stations:
            d_ls = dist[last_node][s]
            if b_last >= d_ls - 1e-9:
                t_arr_s = t_last + d_ls / speed
                b_s = b_last - d_ls
                t_recharge = (battery - b_s) * recharge_rate
                t_s = t_arr_s + t_recharge
                b_s_full = battery
                
                d_sd = dist[s][depot]
                if b_s_full >= d_sd - 1e-9:
                    t_ret = t_s + d_sd / speed
                    if t_ret <= horizon + 1e-9:
                        total_d = d_ls + d_sd
                        if best is None or total_d < best:
                            best = total_d
                            
        return best

    # Clustering
    K = 5
    customers = list(range(1, n_customers + 1))
    
    # Simple K-Means init
    centroids = []
    used = set()
    centroids.append(customers[0])
    used.add(customers[0])
    for _ in range(1, K):
        max_min_dist = -1
        best_c = None
        for c in customers:
            if c in used: continue
            min_d = min(dist[c][cent] for cent in centroids)
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_c = c
        centroids.append(best_c)
        used.add(best_c)
        
    clusters = [[] for _ in range(K)]
    for c in customers:
        min_d = float('inf')
        best_k = 0
        for k, cent in enumerate(centroids):
            d = dist[c][cent]
            if d < min_d:
                min_d = d
                best_k = k
        clusters[best_k].append(c)
        
    all_routes = []
    
    for cluster in clusters:
        if not cluster:
            continue
            
        # Sort by due date
        cluster.sort(key=lambda c: due[c])
        
        current_route_nodes = []
        current_route_states = [] # List of (node, time, battery) for customers in route
        
        for c in cluster:
            best_pos = -1
            best_cost = float('inf')
            best_new_states = None
            
            L = len(current_route_nodes)
            
            for pos in range(L + 1):
                new_nodes = current_route_nodes[:pos] + [c] + current_route_nodes[pos:]
                
                # Simulate
                sim_states = []
                feasible = True
                total_dist = 0.0
                
                prev_node = 0
                prev_t = 0.0
                prev_b = battery
                
                for node in new_nodes:
                    res = get_next_state(prev_node, node, prev_t, prev_b)
                    if res is None:
                        feasible = False
                        break
                    t_node, b_node, d_leg = res
                    sim_states.append((node, t_node, b_node))
                    total_dist += d_leg
                    prev_node = node
                    prev_t = t_node
                    prev_b = b_node
                
                if not feasible:
                    continue
                
                # Check return
                last_node = new_nodes[-1]
                last_t = sim_states[-1][1]
                last_b = sim_states[-1][2]
                
                ret_dist = check_return(last_node, last_t, last_b)
                if ret_dist is None:
                    continue
                
                total_dist += ret_dist
                
                if total_dist < best_cost:
                    best_cost = total_dist
                    best_pos = pos
                    best_new_states = sim_states
            
            if best_pos >= 0:
                current_route_nodes = current_route_nodes[:best_pos] + [c] + current_route_nodes[best_pos:]
                current_route_states = best_new_states
            else:
                # Fallback: new route
                new_route = [c]
                res = get_next_state(0, c, 0.0, battery)
                if res:
                    t_c, b_c, d_c = res
                    ret_dist = check_return(c, t_c, b_c)
                    if ret_dist is not None:
                        all_routes.append(new_route)
        
        if current_route_nodes:
            all_routes.append(current_route_nodes)
            
    return all_routes
# EVOLVE-BLOCK-END
