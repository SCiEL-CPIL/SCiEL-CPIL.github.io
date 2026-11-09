"""Rewritten program: min-cost flow via successive shortest path with potentials."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    n_cust = len(demands)
    n_fac = len(capacities)

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    total_demand = int(demands.sum())
    total_cap = int(capacities.sum())

    # If total capacity < total demand, we cannot satisfy all; just assign greedily
    # to avoid invalid output (but problem states it's feasible).
    # We'll proceed with min-cost flow assuming feasibility.

    # Nodes:
    # 0: source
    # 1..n_cust: customers
    # n_cust+1..n_cust+n_fac: facilities
    # n_cust+n_fac+1: sink

    SRC = 0
    CUST_START = 1
    FAC_START = CUST_START + n_cust
    SNK = FAC_START + n_fac

    N = SNK + 1

    # We build a min-cost flow network:
    # Source -> each customer with capacity = demands[i], cost = 0
    # Each customer -> each facility with capacity = demands[i] (or INF), cost = assign_cost[i][f]
    # Each facility -> sink with capacity = capacities[f], cost = 0
    # Additionally, to model open_cost: we add a fixed cost for opening a facility.
    # Standard trick: add an edge from source to facility with capacity 1, cost = open_cost[f],
    # and from that "open" node... actually simpler: 
    # Alternative: treat open cost as a fixed charge. Min-cost flow doesn't handle fixed charges directly.
    # 
    # Better approach: enumerate which facilities are open? Too expensive.
    # 
    # Instead, use a different modeling:
    # We can model this as a min-cost flow where:
    # - Source connects to each customer (cap=demand, cost=0)
    # - Customer i connects to facility f (cap=INF, cost=assign_cost[i][f])
    # - Facility f connects to sink (cap=capacity[f], cost=0)
    # - Source connects to a "dummy" node for each facility? No.
    #
    # For the open cost, a common approach is to add an edge from source to facility with 
    # capacity 1 and cost open_cost[f], and then the facility's capacity edge has cost 0.
    # But that doesn't quite work because the open cost should be paid once if any flow goes through.
    #
    # Let's use a different strategy: since n_fac is only 20, we can try all subsets? No, 2^20 is too much.
    #
    # Alternative: Use min-cost flow without open cost, then evaluate. But we need to incorporate open cost.
    #
    # Practical approach: Use a heuristic that considers open cost. 
    # Let's try: min-cost flow where we add open_cost[f] to the first unit of flow through facility f.
    # We can model this by having two edges from facility to sink:
    #   Edge 1: capacity 1, cost = open_cost[f]
    #   Edge 2: capacity = capacities[f] - 1, cost = 0
    # This way, the first unit pays the open cost, subsequent units don't. But this is only correct if
    # the facility is used at all. If no flow goes through, no cost. This works!

    # Build adjacency list for min-cost max-flow
    # adj[u] = list of [v, capacity, cost, rev_index]
    adj = [[] for _ in range(N)]

    def add_edge(u, v, cap, cost):
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])

    # Source to customers
    for i in range(n_cust):
        add_edge(SRC, CUST_START + i, int(demands[i]), 0)

    # Customers to facilities
    INF_CAP = total_demand
    for i in range(n_cust):
        for f in range(n_fac):
            add_edge(CUST_START + i, FAC_START + f, INF_CAP, float(assign_cost[i][f]))

    # Facilities to sink with open cost on first unit
    for f in range(n_fac):
        cap_f = int(capacities[f])
        if cap_f >= 1:
            # First unit pays open cost
            add_edge(FAC_START + f, SNK, 1, float(open_cost[f]))
            # Remaining units pay 0
            if cap_f > 1:
                add_edge(FAC_START + f, SNK, cap_f - 1, 0)

    # Successive shortest path with potentials (Dijkstra with reduced costs)
    # We need to send total_demand units of flow from SRC to SNK.

    # Initialize potentials to 0
    potential = np.zeros(N, dtype=np.float64)

    # Flow tracking: we need to track how much flow goes through each customer-facility edge
    # to reconstruct the assignment.
    # Instead, let's track the flow on each edge.

    # For reconstruction, we'll store the flow on customer->facility edges.
    # Let's keep a 2D array flow[i][f] for i in customers, f in facilities.
    flow = np.zeros((n_cust, n_fac), dtype=np.int64)

    # We also need to map edge indices to (i, f) pairs.
    # When we add edges from customer i to facility f, we can record the edge index.

    # Let's restructure: instead of a generic MCMF, let's do a specialized version
    # that tracks flows on customer-facility edges.

    # Actually, let's just do the MCMF and then extract flows by checking residual capacities.

    # For each customer i and facility f, the edge from CUST_START+i to FAC_START+f
    # has a reverse edge. The flow is the reverse edge's capacity.

    # Let's store the index of each customer-facility edge for easy lookup.
    cust_fac_edge_idx = {}
    for i in range(n_cust):
        for f in range(n_fac):
            u = CUST_START + i
            v = FAC_START + f
            # Find the edge in adj[u] that goes to v
            for idx, edge in enumerate(adj[u]):
                if edge[0] == v:
                    cust_fac_edge_idx[(i, f)] = idx
                    break

    # Now run successive shortest path
    total_sent = 0
    while total_sent < total_demand:
        if time.time() > deadline:
            break

        # Dijkstra with potentials
        dist = np.full(N, np.inf, dtype=np.float64)
        prev_node = np.full(N, -1, dtype=np.int64)
        prev_edge = np.full(N, -1, dtype=np.int64)
        dist[SRC] = 0.0
        pq = [(0.0, SRC)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for ei, edge in enumerate(adj[u]):
                v, cap, cost, rev = edge
                if cap > 0:
                    reduced_cost = cost + potential[u] - potential[v]
                    nd = d + reduced_cost
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_node[v] = u
                        prev_edge[v] = ei
                        heapq.heappush(pq, (nd, v))

        if dist[SNK] == np.inf:
            break  # No more augmenting paths

        # Update potentials
        for v in range(N):
            if dist[v] < np.inf:
                potential[v] += dist[v]

        # Find bottleneck on the path
        min_cap = total_demand - total_sent
        v = SNK
        while v != SRC:
            u = prev_node[v]
            ei = prev_edge[v]
            edge = adj[u][ei]
            if edge[1] < min_cap:
                min_cap = edge[1]
            v = u

        # Augment flow
        v = SNK
        while v != SRC:
            u = prev_node[v]
            ei = prev_edge[v]
            edge = adj[u][ei]
            edge[1] -= min_cap
            rev = edge[3]
            adj[v][rev][1] += min_cap
            v = u

        total_sent += min_cap

    # Extract assignment from flows
    result = np.zeros(n_cust, dtype=int)
    assigned = np.zeros(n_cust, dtype=bool)
    for i in range(n_cust):
        for f in range(n_fac):
            if (i, f) in cust_fac_edge_idx:
                ei = cust_fac_edge_idx[(i, f)]
                edge = adj[CUST_START + i][ei]
                rev = edge[3]
                flow_val = adj[FAC_START + f][rev][1]  # reverse edge capacity = flow
                if flow_val > 0:
                    result[i] = f
                    assigned[i] = True
                    break
        if not assigned[i]:
            # Fallback: assign to cheapest feasible facility
            for f in np.argsort(assign_cost[i]):
                # Check capacity
                # This should not happen if flow is feasible
                result[i] = f
                break

    # Verify capacity constraints and fix if needed
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[result[i]] += demands[i]

    # If any facility exceeds capacity, reassign
    for f in range(n_fac):
        if load[f] > capacities[f]:
            excess = load[f] - capacities[f]
            # Find customers assigned to f and move them
            custs_at_f = [i for i in range(n_cust) if result[i] == f]
            # Sort by assign_cost to f (cheapest to move first? or most expensive to keep?)
            # Let's move the ones with highest assign_cost to f first
            custs_at_f.sort(key=lambda i: assign_cost[i][f], reverse=True)
            for i in custs_at_f:
                if excess <= 0:
                    break
                # Find best alternative facility
                best_f = -1
                best_cost = np.inf
                for f2 in np.argsort(assign_cost[i]):
                    if f2 == f:
                        continue
                    if load[f2] + demands[i] <= capacities[f2]:
                        best_f = f2
                        break
                if best_f >= 0:
                    load[f] -= demands[i]
                    load[best_f] += demands[i]
                    result[i] = best_f
                    excess -= demands[i]

    return result
# EVOLVE-BLOCK-END
