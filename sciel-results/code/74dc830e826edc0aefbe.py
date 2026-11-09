# EVOLVE-BLOCK-START
import numpy as np
import heapq

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Node indices
    source = 0
    customer_start = 1
    facility_start = customer_start + n_customers
    sink = facility_start + n_facilities
    n_nodes = sink + 1
    
    # Initial state: Open all facilities
    is_open = np.ones(n_facilities, dtype=bool)
    
    def solve_flow(open_mask):
        """
        Solve min-cost flow for a given set of open facilities.
        Returns (cost, assignment_array) or (inf, None) if infeasible.
        """
        edges = []
        
        # Source -> Customer
        for i in range(n_customers):
            u = source
            v = customer_start + i
            edges.append((u, v, 1, 0))
            
        # Customer -> Facility
        for i in range(n_customers):
            u = customer_start + i
            for f in range(n_facilities):
                if open_mask[f]:
                    v = facility_start + f
                    cost = assign_cost[i, f]
                    edges.append((u, v, 1, cost))
                    
        # Facility -> Sink
        for f in range(n_facilities):
            if open_mask[f]:
                u = facility_start + f
                v = sink
                edges.append((u, v, capacities[f], 0))
        
        # Build adjacency list
        adj = [[] for _ in range(n_nodes)]
        for u, v, cap, cost in edges:
            adj[u].append([v, cap, cost, len(adj[v])])
            adj[v].append([u, 0, -cost, len(adj[u]) - 1])
            
        potentials = [0] * n_nodes
        total_cost = 0
        total_flow = 0
        
        while True:
            # Dijkstra on residual graph with reduced costs
            dist = [float('inf')] * n_nodes
            prev_node = [-1] * n_nodes
            prev_edge_idx = [-1] * n_nodes
            dist[source] = 0
            pq = [(0, source)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for idx, (v, cap, cost, rev_idx) in enumerate(adj[u]):
                    if cap > 0:
                        # Reduced cost
                        rcost = cost + potentials[u] - potentials[v]
                        if dist[u] + rcost < dist[v]:
                            dist[v] = dist[u] + rcost
                            prev_node[v] = u
                            prev_edge_idx[v] = idx
                            heapq.heappush(pq, (dist[v], v))
                            
            if dist[sink] == float('inf'):
                break
                
            # Update potentials
            for i in range(n_nodes):
                if dist[i] < float('inf'):
                    potentials[i] += dist[i]
                    
            # Find bottleneck capacity
            path_flow = float('inf')
            v = sink
            while v != source:
                u = prev_node[v]
                idx = prev_edge_idx[v]
                path_flow = min(path_flow, adj[u][idx][1])
                v = u
                
            # Augment flow
            v = sink
            while v != source:
                u = prev_node[v]
                idx = prev_edge_idx[v]
                adj[u][idx][1] -= path_flow
                # Update reverse edge
                rev_idx = adj[u][idx][3]
                adj[v][rev_idx][1] += path_flow
                v = u
                
            # Update total cost
            total_cost += path_flow * potentials[sink]
            total_flow += path_flow
            
        if total_flow < n_customers:
            return float('inf'), None
            
        # Extract assignment
        assignment = np.zeros(n_customers, dtype=int)
        for i in range(n_customers):
            u = customer_start + i
            found = False
            for v, cap, cost, rev_idx in adj[u]:
                # v should be a facility node
                if v >= facility_start and v < sink:
                    f = v - facility_start
                    # Check if flow went from u to v
                    # Original cap was 1. Current cap is `cap`.
                    # If `cap` is 0, flow is 1.
                    if cap == 0:
                        assignment[i] = f
                        found = True
                        break
            if not found:
                # Should not happen if feasible, but fallback to 0
                assignment[i] = 0
                
        return total_cost, assignment

    current_assignment = None
    current_flow_cost = float('inf')
    current_open_cost = sum(open_cost[is_open])
    current_total_cost = float('inf')
    
    # Initial solve
    flow_cost, assignment = solve_flow(is_open)
    if assignment is not None:
        current_assignment = assignment
        current_flow_cost = flow_cost
        current_total_cost = flow_cost + current_open_cost
    
    improved = True
    max_iterations = 20 # Reduced to ensure speed, 20 passes is usually enough for this size
    iter = 0
    
    while improved and iter < max_iterations:
        improved = False
        iter += 1
        
        # 1. Try to close an open facility
        for f in range(n_facilities):
            if not is_open[f]:
                continue
            new_open = is_open.copy()
            new_open[f] = False
            
            flow_cost, assignment = solve_flow(new_open)
            if assignment is None:
                continue
                
            new_open_cost = current_open_cost - open_cost[f]
            new_total_cost = flow_cost + new_open_cost
            
            if new_total_cost < current_total_cost - 1e-9:
                is_open = new_open
                current_assignment = assignment
                current_flow_cost = flow_cost
                current_open_cost = new_open_cost
                current_total_cost = new_total_cost
                improved = True
                
        # 2. Try to open a closed facility
        for f in range(n_facilities):
            if is_open[f]:
                continue
            new_open = is_open.copy()
            new_open[f] = True
            
            flow_cost, assignment = solve_flow(new_open)
            if assignment is None:
                continue
                
            new_open_cost = current_open_cost + open_cost[f]
            new_total_cost = flow_cost + new_open_cost
            
            if new_total_cost < current_total_cost - 1e-9:
                is_open = new_open
                current_assignment = assignment
                current_flow_cost = flow_cost
                current_open_cost = new_open_cost
                current_total_cost = new_total_cost
                improved = True
                
    return current_assignment
# EVOLVE-BLOCK-END
