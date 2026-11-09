"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import heapq

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    def solve_mcf(open_facilities, demands, capacities, assign_cost):
        """
        Solve min-cost flow for assignment given open facilities.
        Returns (assignment, assign_cost_total) or (None, inf).
        
        Network: source -> facilities (cap=capacity) -> customers (cap=demand, cost=assign_cost) -> sink (cap=demand)
        """
        n_f = len(open_facilities)
        n_c = len(demands)
        
        total_cap = sum(capacities[f] for f in open_facilities)
        total_demand = sum(demands)
        if total_cap < total_demand - 1e-9:
            return None, float('inf')
        
        # Nodes: 0=source, 1..n_f=facilities, n_f+1..n_f+n_c=customers, n_f+n_c+1=sink
        n_nodes = 1 + n_f + n_c + 1
        src = 0
        snk = n_nodes - 1
        
        # graph[node] = list of [to, residual_cap, cost, rev_edge_idx_in_to_node]
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap, cost):
            fwd = [v, cap, cost, len(graph[v])]
            bwd = [u, 0, -cost, len(graph[u])]
            graph[u].append(fwd)
            graph[v].append(bwd)
        
        # Source -> facilities
        for i, f in enumerate(open_facilities):
            add_edge(src, 1 + i, capacities[f], 0.0)
        
        # Facilities -> customers
        for i, f in enumerate(open_facilities):
            for c in range(n_c):
                add_edge(1 + i, 1 + n_f + c, demands[c], assign_cost[c, f])
        
        # Customers -> sink
        for c in range(n_c):
            add_edge(1 + n_f + c, snk, demands[c], 0.0)
        
        # Successive shortest path
        flow = 0.0
        total_cost = 0.0
        
        while flow < total_demand - 1e-9:
            # Dijkstra
            dist = [float('inf')] * n_nodes
            dist[src] = 0.0
            prev_node = [-1] * n_nodes
            prev_edge = [-1] * n_nodes
            visited = [False] * n_nodes
            
            for _ in range(n_nodes):
                u = -1
                best = float('inf')
                for i in range(n_nodes):
                    if not visited[i] and dist[i] < best:
                        best = dist[i]
                        u = i
                if u == -1:
                    break
                visited[u] = True
                for ei, (v, cap, cost, rev) in enumerate(graph[u]):
                    if cap > 1e-9 and not visited[v]:
                        nd = dist[u] + cost
                        if nd < dist[v] - 1e-9:
                            dist[v] = nd
                            prev_node[v] = u
                            prev_edge[v] = ei
            
            if dist[snk] >= float('inf') - 1e-9:
                break
            
            # Find bottleneck
            bottleneck = total_demand - flow
            v = snk
            while v != src:
                u = prev_node[v]
                ei = prev_edge[v]
                bottleneck = min(bottleneck, graph[u][ei][1])
                v = u
            
            if bottleneck <= 1e-9:
                break
            
            # Augment
            flow += bottleneck
            total_cost += bottleneck * dist[snk]
            v = snk
            while v != src:
                u = prev_node[v]
                ei = prev_edge[v]
                graph[u][ei][1] -= bottleneck
                rev = graph[u][ei][3]
                graph[v][rev][1] += bottleneck
                v = u
        
        if flow < total_demand - 1e-9:
            return None, float('inf')
        
        # Extract assignment: for each customer, find which facility sent flow to it
        assignment = np.zeros(n_c, dtype=int)
        assign_cost_total = 0.0
        
        for c in range(n_c):
            cust_node = 1 + n_f + c
            best_f = -1
            best_flow = 0.0
            for i, f in enumerate(open_facilities):
                fac_node = 1 + i
                # Find the forward edge from fac_node to cust_node
                for edge in graph[fac_node]:
                    if edge[0] == cust_node:
                        # Original capacity was demands[c]
                        # Residual = original - net_flow
                        # So net_flow = demands[c] - residual
                        residual = edge[1]
                        net_flow = demands[c] - residual
                        if net_flow > best_flow - 1e-6:
                            best_flow = net_flow
                            best_f = f
            if best_f >= 0:
                assignment[c] = best_f
                assign_cost_total += assign_cost[c, best_f]
            else:
                # Fallback: assign to first open facility
                assignment[c] = open_facilities[0]
                assign_cost_total += assign_cost[c, open_facilities[0]]
        
        return assignment, assign_cost_total
    
    # Start with all facilities
    all_fac = list(range(n_facilities))
    result, cost = solve_mcf(all_fac, demands, capacities, assign_cost)
    
    if result is None:
        # Fallback greedy
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_c = float('inf')
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    if assign_cost[i, f] < best_c:
                        best_c = assign_cost[i, f]
                        best_f = f
            if best_f == -1:
                best_f = int(np.argmax(capacities - load))
            result[i] = best_f
            load[best_f] += demands[i]
        return result.astype(int)
    
    # Compute current total
    open_set = set(result.tolist())
    current_total = cost + sum(open_cost[f] for f in open_set)
    
    # Try closing facilities one at a time
    for _ in range(50):
        improved = False
        open_set = set(result.tolist())
        
        for f in sorted(open_set, key=lambda x: -open_cost[x]):  # Try closing high-cost facilities first
            new_open = [ff for ff in open_set if ff != f]
            if not new_open:
                continue
            
            total_cap = sum(capacities[ff] for ff in new_open)
            if total_cap < sum(demands) - 1e-9:
                continue
            
            new_result, new_cost = solve_mcf(new_open, demands, capacities, assign_cost)
            if new_result is None:
                continue
            
            new_open_set = set(new_result.tolist())
            new_total = new_cost + sum(open_cost[ff] for ff in new_open_set)
            
            if new_total < current_total - 1e-6:
                result = new_result
                current_total = new_total
                improved = True
        
        if not improved:
            break
    
    # Local search: try reassigning customers
    load = np.zeros(n_facilities)
    for c in range(n_customers):
        load[result[c]] += demands[c]
    
    for _ in range(200):
        improved = False
        order = np.argsort(-demands)
        
        for c in order:
            cur_f = result[c]
            cur_cost = assign_cost[c, cur_f]
            best_f = cur_f
            best_delta = 0.0
            
            for f in range(n_facilities):
                if f == cur_f:
                    continue
                if load[f] + demands[c] <= capacities[f] + 1e-9:
                    delta_assign = assign_cost[c, f] - cur_cost
                    
                    # Open cost change
                    delta_open = 0.0
                    if load[cur_f] - demands[c] <= 1e-9:
                        delta_open -= open_cost[cur_f]
                    if load[f] <= 1e-9:
                        delta_open += open_cost[f]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < best_delta - 1e-9:
                        best_delta = total_delta
                        best_f = f
            
            if best_f != cur_f:
                load[cur_f] -= demands[c]
                load[best_f] += demands[c]
                result[c] = best_f
                improved = True
        
        if not improved:
            break
    
    # Final validation
    load = np.zeros(n_facilities)
    for c in range(n_customers):
        f = int(result[c])
        if f < 0 or f >= n_facilities:
            raise ValueError(f"Invalid facility index {f}")
        load[f] += demands[c]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            raise ValueError(f"Facility {f} over capacity")
    
    return result.astype(int)

# EVOLVE-BLOCK-END
