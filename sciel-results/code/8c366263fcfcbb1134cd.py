"""MCMF-based solver for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: Formulate as a Minimum-Cost Maximum-Flow problem.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    # Node indexing
    # Source: 0
    # Customers: 1..n_customers
    # Facilities: n_customers+1 .. n_customers+n_facilities
    # Sink: n_customers+n_facilities+1
    src = 0
    cust_start = 1
    fac_start = cust_start + n_customers
    sink = fac_start + n_facilities
    n_nodes = sink + 1

    # Graph representation: list of lists of edge indices
    adj = [[] for _ in range(n_nodes)]
    # Edge data: to, rev, cap, cost
    # We use lists of lists of lists: graph[u] = [(v, rev_index, capacity, cost)]
    graph = [[] for _ in range(n_nodes)]

    def add_edge(u, v, cap, cost):
        graph[u].append([v, len(graph[v]), cap, cost])
        graph[v].append([u, len(graph[u]) - 1, 0, -cost])

    # Add edges
    # 1. Source -> Customers (demand)
    for c in range(n_customers):
        add_edge(src, cust_start + c, demands[c], 0.0)

    # 2. Customers -> Facilities (1 unit, cost = assign_cost)
    for c in range(n_customers):
        u = cust_start + c
        for f in range(n_facilities):
            v = fac_start + f
            add_edge(u, v, 1.0, assign_cost[c, f])

    # 3. Facilities -> Sink (capacity)
    # Surrogate for fixed charge: distribute open_cost over capacity
    # This makes the flow problem approximate the CFLP objective
    for f in range(n_facilities):
        u = fac_start + f
        # If capacity is 0, skip (though problem implies valid capacities)
        if capacities[f] > 0:
            # cost per unit = open_cost / capacity
            cost_per_unit = open_cost[f] / capacities[f]
            add_edge(u, sink, capacities[f], cost_per_unit)
        else:
            add_edge(u, sink, 0.0, 0.0)

    # MCMF using Successive Shortest Path with Potentials (Dijkstra)
    # We want to send total_demand units from src to sink.
    total_demand = demands.sum()
    
    # Initial potentials (0 is fine if no negative cycles, which is true here)
    h = [0.0] * n_nodes
    flow = 0.0
    cost = 0.0
    
    # We will track the flow on edges to reconstruct the assignment
    # However, storing flow on every edge is expensive.
    # Instead, we can just run the algorithm and at the end, 
    # check which edges from Customers to Facilities have flow > 0.
    # But we need to know the flow. Let's store it.
    # To save memory/time, we only care about Customer->Facility edges.
    # We can store the flow in a dictionary or array indexed by (c, f).
    # But the graph structure mixes them.
    # Let's just keep the graph and inspect it at the end.
    
    # Optimization: Since n_customers=100, n_facilities=20, edges ~ 2000.
    # Nodes ~ 122.
    # Dijkstra is O(E log V). 
    # Max flow is ~100*20 = 2000.
    # Total ops ~ 2000 * 2000 * log(122) ~ 4e6 * 7 ~ 2.8e7. 
    # Python might be slow, but let's try.
    
    # If it's too slow, we can use a simpler approach:
    # Since capacities are small integers (5-35), and n_facilities is small,
    # maybe we can use a different method?
    # No, MCMF is the "different strategy" requested.
    
    # Let's implement a fast MCMF.
    
    INF = 1e18
    prevv = [-1] * n_nodes
    preve = [-1] * n_nodes
    dist = [0.0] * n_nodes
    
    while flow < total_demand - 1e-9:
        if time.time() > deadline - 10:
            break
            
        # Dijkstra
        dist = [INF] * n_nodes
        dist[src] = 0.0
        pq = [(0.0, src)]
        
        while pq:
            d, v = heapq.heappop(pq)
            if dist[v] < d:
                continue
            for i, e in enumerate(graph[v]):
                if e[2] > 1e-9: # capacity > 0
                    nd = d + e[3] + h[v] - h[e[0]]
                    if dist[e[0]] > nd:
                        dist[e[0]] = nd
                        prevv[e[0]] = v
                        preve[e[0]] = i
                        heapq.heappush(pq, (nd, e[0]))
        
        if dist[sink] == INF:
            # No more augmenting path
            break
            
        # Update potentials
        for v in range(n_nodes):
            if dist[v] < INF:
                h[v] += dist[v]
                
        # Augment flow
        d = total_demand - flow
        v = sink
        while v != src:
            e = graph[prevv[v]][preve[v]]
            e[2] -= d
            graph[v][e[1]][2] += d
            v = prevv[v]
            
        flow += d
        cost += d * h[sink] # h[sink] is the shortest path cost
        
    # Reconstruct assignment
    assignment = np.full(n_customers, -1, dtype=int)
    
    # Iterate through Customer -> Facility edges
    for c in range(n_customers):
        u = cust_start + c
        for i, e in enumerate(graph[u]):
            # Check if this is a forward edge to a facility
            # Forward edges have index i, reverse edges have index e[1] in graph[e[0]]
            # We added forward edges first.
            # The edge is from u to v.
            # If flow was sent, the reverse edge (in graph[v]) will have capacity > 0
            # and the forward edge will have capacity < 1 (since initial cap was 1).
            # Actually, we can just check if the reverse edge has capacity > 0.
            # The reverse edge is graph[e[0]][e[1]].
            if e[0] >= fac_start and e[0] < sink:
                # This is a customer -> facility edge
                # Check reverse edge capacity
                rev_edge = graph[e[0]][e[1]]
                if rev_edge[2] > 1e-9:
                    # Flow exists
                    f_idx = e[0] - fac_start
                    assignment[c] = f_idx
                    break

    # Verify feasibility and repair if needed
    # MCMF should guarantee feasibility if total capacity >= total demand.
    # But floating point issues or logic errors might cause issues.
    load = np.zeros(n_facilities)
    for c in range(n_customers):
        f = assignment[c]
        if f >= 0 and f < n_facilities:
            load[f] += demands[c]
        else:
            # Unassigned customer, assign to any facility with space
            # This should not happen
            for f in range(n_facilities):
                if load[f] + demands[c] <= capacities[f]:
                    assignment[c] = f
                    load[f] += demands[c]
                    break

    # Check capacity violations
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            # Repair: move customers
            # This is a fallback, shouldn't trigger often
            excess = load[f] - capacities[f]
            customers_at_f = [c for c in range(n_customers) if assignment[c] == f]
            customers_at_f.sort(key=lambda c: assign_cost[c].min() - assign_cost[c, f])
            for c in customers_at_f:
                if excess <= 1e-9:
                    break
                # Find best alternative
                best_alt_f = -1
                best_alt_cost = np.inf
                for f2 in range(n_facilities):
                    if f2 == f:
                        continue
                    if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                        alt_cost = assign_cost[c, f2]
                        if alt_cost < best_alt_cost:
                            best_alt_cost = alt_cost
                            best_alt_f = f2
                if best_alt_f >= 0:
                    load[f] -= demands[c]
                    load[best_alt_f] += demands[c]
                    assignment[c] = best_alt_f
                    excess -= demands[c]

    # Local search to improve cost (since MCMF uses surrogate cost)
    # We can do a quick local search to fix the "fixed charge" approximation.
    if time.time() < deadline - 5:
        # Simple hill climbing
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for c in range(n_customers):
                old_f = assignment[c]
                load[old_f] -= demands[c]
                best_delta = 0.0
                best_new_f = old_f
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    improved = True
                else:
                    load[old_f] += demands[c]

    return assignment.astype(int)
# EVOLVE-BLOCK-END
