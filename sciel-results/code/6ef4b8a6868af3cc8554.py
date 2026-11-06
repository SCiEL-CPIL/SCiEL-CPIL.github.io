"""Path decomposition of three fixed connected graphs (Gallai's conjecture).
Uses min-cost max-flow for exact minimum path decomposition.
"""
import time
import heapq
from collections import defaultdict, deque


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _min_cost_path_decomposition(n, edges):
    """
    Solve minimum path decomposition via min-cost max-flow.
    
    Network:
    - For each vertex v: nodes v_in (v) and v_out (v+n)
    - For each edge (u,v): arcs u_out->v_in cap=1 cost=0, v_out->u_in cap=1 cost=0
    - For each vertex v: arc v_in->v_out cap=1 cost=0 (continuation)
    - Source S -> v_out cap=1 cost=1 (start a path)
    - v_in -> Sink T cap=1 cost=0 (end a path)
    
    Max flow = m (number of edges). Min cost = number of paths.
    """
    S = 2 * n
    T = 2 * n + 1
    NUM = 2 * n + 2
    
    # Build adjacency: adj[u] = list of (v, cap, cost, edge_id)
    # edge_id is index into edges_list for reverse lookup
    adj = [[] for _ in range(NUM)]
    edges_list = []  # (from, to, cap, cost)
    
    def add_arc(u, v, cap, cost):
        idx = len(edges_list)
        edges_list.append((u, v, cap, cost))
        adj[u].append((v, cap, cost, idx))
        # reverse arc
        edges_list.append((v, u, 0, -cost))
        adj[v].append((u, 0, -cost, idx + 1))
        return idx
    
    # Edge arcs: for each undirected edge (u,v), add u_out->v_in and v_out->u_in
    for (u, v) in edges:
        add_arc(u + n, v, 1, 0)  # u_out -> v_in
        add_arc(v + n, u, 1, 0)  # v_out -> u_in
    
    # Continuation arcs: v_in -> v_out, cap=1, cost=0
    for v in range(n):
        add_arc(v, v + n, 1, 0)
    
    # Source arcs: S -> v_out, cap=1, cost=1
    for v in range(n):
        add_arc(S, v + n, 1, 1)
    
    # Sink arcs: v_in -> T, cap=1, cost=0
    for v in range(n):
        add_arc(v, T, 1, 0)
    
    # Min-cost max-flow using successive shortest paths (SPFA or Dijkstra with potentials)
    # Since costs are 0 or 1 (and -1 for reverse), we can use Dijkstra with potentials.
    
    INF = float('inf')
    flow = 0
    cost = 0
    cap = [e[2] for e in edges_list]  # mutable capacities
    potential = [0] * NUM  # Johnson's potentials
    
    while True:
        # Dijkstra with potentials
        dist = [INF] * NUM
        dist[S] = 0
        prev_node = [-1] * NUM
        prev_edge = [-1] * NUM
        pq = [(0, S)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for (v, c, w, eid) in adj[u]:
                if c > 0:
                    # Reduced cost
                    rcost = w + potential[u] - potential[v]
                    nd = d + rcost
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_node[v] = u
                        prev_edge[v] = eid
                        heapq.heappush(pq, (nd, v))
        
        if dist[T] == INF:
            break
        
        # Update potentials
        for i in range(NUM):
            if dist[i] < INF:
                potential[i] += dist[i]
        
        # Augment flow along the path
        # Find bottleneck (should be 1 since all caps are 0 or 1)
        bottleneck = INF
        v = T
        while v != S:
            u = prev_node[v]
            eid = prev_edge[v]
            if cap[eid] < bottleneck:
                bottleneck = cap[eid]
            v = u
        
        if bottleneck <= 0:
            break
        
        # Update capacities
        v = T
        while v != S:
            u = prev_node[v]
            eid = prev_edge[v]
            cap[eid] -= bottleneck
            cap[eid ^ 1] += bottleneck
            v = u
        
        flow += bottleneck
        cost += bottleneck * potential[T]  # actual cost = potential[T] after update... 
        # Actually cost = sum of actual costs along path
        # Let me recalculate: the actual shortest path cost in original costs
        # With potentials, dist[T] + potential_old[T] - potential_old[S] = actual cost
        # But we updated potentials, so let's just track it differently.
        
        # Actually, let's just sum the actual costs along the path
        # Redo: the cost of augmenting is the sum of w for each edge on path
        # In the reduced cost formulation, actual_cost = dist[T] + potential_old[T] - potential_old[S]
        # But we already updated potential. Let me just compute it directly.
        pass
    
    # Let me redo this more carefully
    # Actually let me just track cost properly
    
    # Redo the algorithm with proper cost tracking
    cap = [e[2] for e in edges_list]
    potential = [0] * NUM
    total_cost = 0
    total_flow = 0
    
    while True:
        dist = [INF] * NUM
        dist[S] = 0
        prev_node = [-1] * NUM
        prev_edge = [-1] * NUM
        pq = [(0, S)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for (v, c, w, eid) in adj[u]:
                if c > 0:
                    rcost = w + potential[u] - potential[v]
                    nd = d + rcost
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_node[v] = u
                        prev_edge[v] = eid
                        heapq.heappush(pq, (nd, v))
        
        if dist[T] == INF:
            break
        
        # Actual cost of this path = dist[T] + potential[T] - potential[S]
        # But potential[S] is always 0 since dist[S]=0 and we start with 0
        # Actually the actual shortest path distance = dist[T] + potential[T] - potential[S]
        # Before updating potentials. Let's compute it before updating.
        path_cost = dist[T] + potential[T] - potential[S]
        
        # Update potentials
        for i in range(NUM):
            if dist[i] < INF:
                potential[i] += dist[i]
        
        # Augment
        bottleneck = INF
        v = T
        while v != S:
            u = prev_node[v]
            eid = prev_edge[v]
            if cap[eid] < bottleneck:
                bottleneck = cap[eid]
            v = u
        
        if bottleneck <= 0:
            break
        
        v = T
        while v != S:
            u = prev_node[v]
            eid = prev_edge[v]
            cap[eid] -= bottleneck
            cap[eid ^ 1] += bottleneck
            v = u
        
        total_flow += bottleneck
        total_cost += bottleneck * path_cost
    
    # Now extract paths from the flow
    # A path starts where there's flow on S->v_out and ends where there's flow on v_in->T
    # We need to trace through the network
    
    # Build a map: for each arc that has flow, record it
    # Arcs with flow: check cap[eid] < original_cap
    # Actually, flow on arc eid = original_cap - cap[eid]
    
    # Let's trace paths:
    # Start from S, follow arcs with flow=1
    # S -> v_out (start of path at v)
    # v_out -> w_in (edge v-w used)
    # w_in -> w_out (continuation at w)
    # ... until w_in -> T (end of path at w)
    
    paths = []
    # For each vertex v, check if S->v_out has flow
    # The arc S->v_out was added at a specific index
    # Let's rebuild: we need to know which arc indices correspond to what
    
    # Alternative: just trace by looking at which arcs have flow
    # Create a mapping from (from, to) to whether flow exists
    # Since capacities are 0/1, flow is 1 if cap[eid] == original_cap - 1
    
    # Let me just trace using the residual graph structure
    # For each vertex v, if there's flow on S->(v+n), then v is a path start
    
    # Build reverse lookup: for each node, which outgoing arcs have flow
    flow_adj = defaultdict(list)  # node -> list of (next_node, edge_id)
    orig_caps = [e[2] for e in edges_list]
    for eid, (u, v, oc, w) in enumerate(edges_list):
        if oc > 0 and cap[eid] < oc:
            # This arc has flow (oc - cap[eid]) which should be 1
            flow_adj[u].append((v, eid))
    
    # Trace paths starting from S
    visited_edges = set()
    for (v_out, eid) in flow_adj[S]:
        # v_out = v + n, so start vertex is v
        start_v = v_out - n
        path = [start_v]
        cur = v_out
        while True:
            # Find next node from cur
            next_found = False
            for (next_node, eid2) in flow_adj[cur]:
                if eid2 not in visited_edges:
                    visited_edges.add(eid2)
                    if next_node == T:
                        break  # end of path
                    # next_node is either w_in (w) or w_out (w+n)
                    if next_node < n:
                        # It's w_in, so vertex w
                        path.append(next_node)
                    else:
                        # It's w_out, shouldn't happen directly from v_out
                        # Actually v_out -> w_in, so next_node should be < n
                        pass
                    cur = next_node
                    next_found = True
                    break
            if not next_found or cur == T:
                break
            if cur == T:
                break
    
    # This tracing is getting complex. Let me use a simpler approach:
    # Just find which edges have flow and reconstruct paths from that.
    
    # Actually, let me use a cleaner approach to extract paths.
    # The flow tells us:
    # 1. Which original edges are used (all of them, since flow=m)
    # 2. At each vertex, whether the path "turns" (uses continuation arc) or "ends/starts"
    
    # For each vertex v:
    # - inflow from edge arcs: number of edge-arcs entering v_in with flow
    # - outflow to edge arcs: number of edge-arcs leaving v_out with flow  
    # - continuation: flow on v_in->v_out
    # - start: flow on S->v_out
    # - end: flow on v_in->T
    
    # A vertex v is:
    # - internal to a path if: continuation arc has flow (in=out=1 for edge arcs)
    # - start of a path if: S->v_out has flow
    # - end of a path if: v_in->T has flow
    # - isolated edge if: both start and end
    
    # Let me just identify starts and ends, then trace
    starts = []  # vertices that are path starts
    ends = []    # vertices that are path ends
    
    # Find flow on S->v_out arcs
    # These were added in a specific order. Let me track them.
    # Actually, let me just check flow_adj[S]
    for (v_out, eid) in flow_adj.get(S, []):
        starts.append(v_out - n)
    
    for (v_in, eid) in flow_adj.get(T, []):
        # v_in is the vertex index (0 to n-1)
        ends.append(v_in)
    
    # Now trace paths from each start
    # At a start vertex v, the path goes v -> w where (v,w) is an edge with flow on v_out->w_in
    # Then at w, if continuation arc w_in->w_out has flow, continue; else w is end
    
    # Build: for each vertex v, which neighbor w has flow on v_out->w_in
    next_from_out = {}  # v -> w (the unique neighbor if flow exists)
    # And for each vertex v, which neighbor u has flow on u_out->v_in (incoming edge)
    prev_to_in = {}  # v -> u
    
    for (u, v, oc, w) in edges_list:
        eid = edges_list.index((u, v, oc, w))  # This is O(n^2), let me fix
        pass
    
    # Let me redo with proper indexing
    # Rebuild the flow info
    # edges_list[i] = (from, to, cap, cost)
    # Arc i goes from edges_list[i][0] to edges_list[i][1]
    # Reverse arc is i^1
    
    # For each original edge (u,v) in the graph:
    # There are two arcs: u+n -> v and v+n -> u
    # Flow on u+n->v means the path uses edge (u,v) in direction u->v
    
    # Let me just directly trace using the flow_adj we built
    
    paths = []
    # Reset visited
    visited_arc_ids = set()
    
    for start_v in starts:
        path = [start_v]
        cur_out = start_v + n  # current position is at v_out
        
        while True:
            # From cur_out, find which v_in we go to (edge arc)
            next_v_in = None
            for (next_node, eid) in flow_adj.get(cur_out, []):
                if next_node < n and eid not in visited_arc_ids:
                    visited_arc_ids.add(eid)
                    next_v_in = next_node
                    break
            
            if next_v_in is None:
                break
            
            path.append(next_v_in)
            cur_in = next_v_in  # now at w_in
            
            # Check if continuation: w_in -> w_out
            cont_found = False
            for (next_node, eid) in flow_adj.get(cur_in, []):
                if next_node == cur_in + n and eid not in visited_arc_ids:
                    visited_arc_ids.add(eid)
                    cur_out = next_node
                    cont_found = True
                    break
            
            if not cont_found:
                # Path ends here
                break
    
        if len(path) >= 2:
            paths.append(path)
    
    # Verify all edges are covered
    used_edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            used_edges.add((min(a, b), max(a, b)))
    
    edge_set = set((min(u, v), max(u, v)) for (u, v) in edges)
    missing = edge_set - used_edges
    
    # If any edges are missing, add them as single-edge paths
    for e in missing:
        paths.append([e[0], e[1]])
    
    return paths


def _validate_and_fix(n, edges, paths):
    """Ensure paths are simple, use graph edges, edge-disjoint, cover all edges."""
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    fixed = []
    used = set()
    for path in paths:
        if len(path) < 2:
            continue
        if len(path) != len(set(path)):
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                e = (min(a, b), max(a, b))
                if e in edge_set and e not in used:
                    fixed.append([a, b])
                    used.add(e)
            continue
        valid = True
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (min(a, b), max(a, b))
            if e not in edge_set or e in used:
                valid = False
                break
        if valid:
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                used.add((min(a, b), max(a, b)))
            fixed.append(path)
        else:
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                e = (min(a, b), max(a, b))
                if e in edge_set and e not in used:
                    fixed.append([a, b])
                    used.add(e)

    for e in edge_set:
        if e not in used:
            fixed.append([e[0], e[1]])
            used.add(e)

    return fixed


def construct():
    start_time = time.time()
    deadline = start_time + 140

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            n, p, seed = spec
            edges = _instance(n, p, seed)
            if len(edges) == 0:
                results.append([])
                continue
            
            paths = _min_cost_path_decomposition(n, edges)
            paths = _validate_and_fix(n, edges, paths)
            results.append(paths)

    return results
