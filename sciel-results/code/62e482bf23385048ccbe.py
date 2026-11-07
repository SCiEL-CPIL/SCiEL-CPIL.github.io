    def solve_cluster(cluster_customers):
        if not cluster_customers:
            return []
            
        total_demand = sum(demands[c] for c in cluster_customers)
        if total_demand > capacity:
            return None

        n_cluster = len(cluster_customers)
        full_mask = (1 << n_cluster) - 1
        
        # Allowed nodes: Depot, Stations, Customers in cluster
        allowed_nodes = {0} | stations | set(cluster_customers)
        
        # Map allowed nodes to indices in a local list for faster iteration?
        # Or just iterate over all_nodes and check membership.
        
        # State: (node_id, battery_idx, visited_mask)
        # We use global node indices from all_nodes
        
        B_STEP = 5
        B_LEVELS = int(battery_max / B_STEP) + 1
        
        def b_to_idx(b):
            return int(b / B_STEP)
            
        def idx_to_b(i):
            return i * B_STEP
            
        # dist[node_idx][b_idx][mask]
        dist = [[[float('inf')] * (1 << n_cluster) for _ in range(B_LEVELS)] for _ in range(n_nodes)]
        pred = [[ [None] * (1 << n_cluster) for _ in range(B_LEVELS) ] for _ in range(n_nodes)]
        
        start_node = 0
        start_b_idx = b_to_idx(battery_max)
        start_mask = 0
        
        start_n_idx = node_to_idx[start_node]
        dist[start_n_idx][start_b_idx][start_mask] = 0.0
        time_at_state = [[ [0.0] * (1 << n_cluster) for _ in range(B_LEVELS) ] for _ in range(n_nodes)]
        time_at_state[start_n_idx][start_b_idx][start_mask] = 0.0
        
        heap = []
        heapq.heappush(heap, (0.0, 0.0, start_n_idx, start_b_idx, start_mask))
        
        while heap:
            d, t, u_idx, u_b_idx, u_mask = heapq.heappop(heap)
            
            if d > dist[u_idx][u_b_idx][u_mask] + 1e-9:
                continue
                
            if u_idx == start_n_idx and u_mask == full_mask:
                path = []
                cur_idx = u_idx
                cur_b_idx = u_b_idx
                cur_mask = u_mask
                while pred[cur_idx][cur_b_idx][cur_mask] is not None:
                    prev_n_idx, prev_b_idx, prev_mask = pred[cur_idx][cur_b_idx][cur_mask]
                    path.append(all_nodes[cur_idx])
                    cur_idx, cur_b_idx, cur_mask = prev_n_idx, prev_b_idx, prev_mask
                path.append(all_nodes[cur_idx])
                path.reverse()
                return path[1:-1]

            u_node = all_nodes[u_idx]
            
            for v_idx in range(n_nodes):
                v_node = all_nodes[v_idx]
                
                # Optimization: Skip nodes not in allowed set
                if v_node not in allowed_nodes:
                    continue
                
                travel_dist = dist_mat[u_idx][v_idx]
                travel_time = travel_dist / speed
                
                current_battery = idx_to_b(u_b_idx)
                if current_battery < travel_dist:
                    continue
                    
                new_battery = current_battery - travel_dist
                new_time = t + travel_time
                
                new_mask = u_mask
                
                if v_node in stations:
                    recharge_time = (battery_max - new_battery) * recharge_rate
                    new_time += recharge_time
                    new_battery = battery_max
                elif 1 <= v_node <= n_cust:
                    # Must be in cluster
                    try:
                        c_idx = cluster_customers.index(v_node)
                    except ValueError:
                        continue
                    
                    if (u_mask >> c_idx) & 1:
                        continue
                        
                    if new_time < ready[v_node]:
                        new_time = ready[v_node]
                    if new_time > due[v_node]:
                        continue
                        
                    new_time += service[v_node]
                    new_mask = u_mask | (1 << c_idx)
                else:
                    # Depot
                    if v_node == 0 and new_mask != full_mask:
                        continue

                if new_time > horizon:
                    continue
                
                new_b_idx = b_to_idx(new_battery)
                
                if d + travel_dist < dist[v_idx][new_b_idx][new_mask] - 1e-9:
                    dist[v_idx][new_b_idx][new_mask] = d + travel_dist
                    time_at_state[v_idx][new_b_idx][new_mask] = new_time
                    pred[v_idx][new_b_idx][new_mask] = (u_idx, u_b_idx, u_mask)
                    heapq.heappush(heap, (d + travel_dist, new_time, v_idx, new_b_idx, new_mask))
        
        return None
