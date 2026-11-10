"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        """Return the set of edges in the 2-core of the graph defined by edge_set."""
        if not edge_set:
            return set()
        
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        queue = deque([v for v in deg if deg[v] < 2])
        in_core = set(deg.keys())
        
        while queue:
            v = queue.popleft()
            if v not in in_core:
                continue
            in_core.discard(v)
            for nb in adj[v]:
                if nb in in_core:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        queue.append(nb)
                        
        core_edges = set()
        for u, v in edge_set:
            if u in in_core and v in in_core:
                core_edges.add(tuple(sorted((u, v))))
        return core_edges

    def find_cycle_via_flow(rem_edges, n_vertices):
        """
        Use max-flow to find a cycle in the graph.
        For each edge (u,v), create two directed arcs u->v and v->u with capacity 1.
        Add super-source S and super-sink T.
        For each vertex v, add arc S->v_in (cap 1) and v_out->T (cap 1).
        For each original edge (u,v), add arcs u_out->v_in (cap 1) and v_out->u_in (cap 1).
        
        A cycle corresponds to a unit flow that goes S -> u_in -> u_out -> v_in -> v_out -> T
        but we need the flow to form a cycle. 
        
        Better approach: To find ANY cycle, we can try each edge as a "forced" edge.
        For edge (u,v), remove it from consideration temporarily, then check if there's
        a path from v to u in the remaining graph. If yes, we have a cycle.
        
        Actually, let's use a simpler max-flow approach:
        - Split each vertex v into v_in and v_out with capacity 1 on the v_in->v_out arc.
        - For each undirected edge (u,v), add directed arcs u_out->v_in and v_out->u_in with cap 1.
        - Add super-source S connected to all v_in with cap 1.
        - Add super-sink T connected from all v_out with cap 1.
        
        Wait, that doesn't directly give cycles. Let me think differently.
        
        To find a simple cycle: pick an edge (u,v). Check if there's a path from v to u 
        in G - {(u,v)}. If so, that path plus the edge forms a cycle.
        
        For efficiency, we can use BFS/DFS for this. Let me implement a hybrid:
        1. Use BFS to find cycles (fast for sparse graphs).
        2. Use a smarter heuristic: find the longest cycle first.
        """
        if not rem_edges:
            return None
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try to find a cycle using BFS from each edge
        # For each edge (u,v), check if there's a path from v to u avoiding that edge
        edge_list = list(rem_edges)
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            # BFS from v to u, not using edge (u,v)
            visited = {v}
            parent = {v: None}
            queue = deque([v])
            found = False
            
            while queue and not found:
                curr = queue.popleft()
                for nb in adj[curr]:
                    if (min(curr, nb), max(curr, nb)) == (min(u, v), max(u, v)):
                        continue  # Skip the edge we're trying to close
                    if nb == u:
                        # Found path from v to u
                        # Reconstruct path
                        path = [u]
                        node = curr
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        # path is v -> ... -> curr -> u, plus edge u-v closes the cycle
                        # Actually path goes from v to curr, then curr to u
                        # The cycle is: u, curr, ..., v, u (closing with edge v-u)
                        # Let me reconstruct properly
                        cycle = [u]
                        node = curr
                        while node != v:
                            cycle.append(node)
                            node = parent[node]
                        cycle.append(v)
                        # Verify: cycle should be u, ..., v and edge (u,v) closes it
                        return cycle
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = curr
                        queue.append(nb)
        
        return None

    def find_longest_cycle_greedy(rem_edges, n_vertices, max_iter=50):
        """
        Try to find a long cycle by repeatedly extending a partial path.
        """
        if not rem_edges:
            return None
        
        adj = defaultdict(set)
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_cycle = None
        best_len = 0
        
        vertices = set()
        for u, v in rem_edges:
            vertices.add(u)
            vertices.add(v)
        
        if len(vertices) < 3:
            return None
        
        # Try multiple starting points
        start_candidates = list(vertices)
        random.shuffle(start_candidates)
        
        for start in start_candidates[:10]:
            if len(adj[start]) < 2:
                continue
            
            # Try to build a long path starting and ending near start
            # Use a heuristic: always go to the neighbor with highest degree
            visited = {start}
            path = [start]
            
            for _ in range(max_iter):
                curr = path[-1]
                # Get unvisited neighbors
                unvisited = [nb for nb in adj[curr] if nb not in visited]
                
                if not unvisited:
                    # Check if we can close a cycle
                    if len(path) >= 3 and path[0] in adj[curr]:
                        cycle = path[:]
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle[:]
                    break
                
                # Choose neighbor with highest degree (greedy)
                unvisited.sort(key=lambda x: len(adj[x]), reverse=True)
                next_v = unvisited[0]
                visited.add(next_v)
                path.append(next_v)
            
            # Check if we can close the cycle
            if len(path) >= 3 and path[0] in adj[path[-1]]:
                cycle = path[:]
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle[:]
        
        return best_cycle

    def decompose(edges, n, time_budget):
        """
        Decompose the edge set into cycles and single edges.
        Strategy:
          1. Separate 2-core from forest.
          2. Repeatedly extract cycles from the core.
          3. Use longest-cycle-first heuristic to maximize savings.
        """
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        best_pieces = None
        best_count = len(edge_set) + 1

        def decompose_once(rem_set):
            """One decomposition attempt."""
            rem = set(rem_set)
            pieces = []
            
            core_edges = get_2_core(rem)
            forest_edges = rem - core_edges
            
            for e in forest_edges:
                pieces.append(list(e))
            
            rem = core_edges
            if not rem:
                return pieces

            while rem:
                # Try to find a long cycle first
                cycle = find_longest_cycle_greedy(rem, n, max_iter=30)
                
                if not cycle or len(cycle) < 3:
                    # Fall back to any cycle
                    cycle = find_cycle_via_flow(rem, n)
                
                if not cycle or len(cycle) < 3:
                    # No cycle; take one edge
                    e = tuple(next(iter(rem)))
                    rem.discard(e)
                    pieces.append(list(e))
                else:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        rem.discard(e)
                    pieces.append(cycle[:])
            
            return pieces

        # Multiple restarts
        max_restarts = 40
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            pieces = decompose_once(edge_set)
            cnt = len(pieces)
            if cnt < best_count:
                best_count = cnt
                best_pieces = [p[:] for p in pieces]

        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
