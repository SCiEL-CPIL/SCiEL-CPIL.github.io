"""Rewritten program: minimum edge-path cover via optimal odd-vertex pairing."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135  # safety margin before 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def bfs_distances(adj, src, n):
        """BFS from src, return distance array of length n."""
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            v = q.popleft()
            for nb in adj[v]:
                if dist[nb] < 0:
                    dist[nb] = dist[v] + 1
                    q.append(nb)
        return dist

    def min_path_cover(edges, n):
        """Minimum edge-disjoint path cover.
        
        Key insight: the minimum number of edge-disjoint paths covering all edges
        equals the number of odd-degree vertices divided by 2. We achieve this by:
        1. Finding shortest paths between all pairs of odd-degree vertices.
        2. Finding a minimum-weight perfect matching on odd vertices (via bitmask DP).
        3. Adding dummy edges along those shortest paths.
        4. Finding an Eulerian circuit in the augmented graph.
        5. Splitting at dummy edges to get the path decomposition.
        """
        if not edges:
            return []

        # Compute degrees
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        k = len(odd_verts)

        if k == 0:
            # All even degrees: single Eulerian circuit
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Find start
            start = 0
            while start < n and not adj[start]:
                start += 1
            if start >= n:
                return []
            
            circuit = eulerian_circuit(adj, start)
            if circuit and len(circuit) >= 2:
                return [circuit]
            return []

        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # BFS from each odd vertex
        dist_from_odd = []
        for src in odd_verts:
            dist = bfs_distances(adj, src, n)
            dist_from_odd.append(dist)

        # Build cost matrix for matching
        # cost[i][j] = distance between odd_verts[i] and odd_verts[j]
        cost = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                cost[i][j] = dist_from_odd[i][odd_verts[j]]
                if cost[i][j] < 0:
                    cost[i][j] = float('inf')  # shouldn't happen in connected graph

        # Bitmask DP for minimum weight perfect matching
        # dp[mask] = min cost to match the set of vertices indicated by mask
        full_mask = (1 << k) - 1
        dp = [float('inf')] * (1 << k)
        dp[0] = 0
        parent = [(-1, -1)] * (1 << k)  # parent[mask] = (prev_mask, partner_index)

        for mask in range(1 << k):
            if dp[mask] == float('inf'):
                continue
            # Find first unmatched vertex
            first = -1
            for i in range(k):
                if not (mask & (1 << i)):
                    first = i
                    break
            if first == -1:
                continue
            # Try pairing first with any other unmatched vertex
            for j in range(first + 1, k):
                if not (mask & (1 << j)):
                    new_mask = mask | (1 << first) | (1 << j)
                    c = cost[first][j]
                    if c == float('inf'):
                        continue
                    new_cost = dp[mask] + c
                    if new_cost < dp[new_mask]:
                        dp[new_mask] = new_cost
                        parent[new_mask] = (mask, j)

        total_cost = dp[full_mask]
        if total_cost == float('inf'):
            # Fallback: greedy
            pairs = [(odd_verts[i], odd_verts[i+1]) for i in range(0, k, 2)]
        else:
            # Recover pairs by backtracking
            pairs = []
            mask = full_mask
            while mask != 0:
                prev_mask, partner = parent[mask]
                # Find first unmatched in prev_mask
                first = -1
                for i in range(k):
                    if not (prev_mask & (1 << i)):
                        first = i
                        break
                pairs.append((odd_verts[first], odd_verts[partner]))
                mask = prev_mask

        # Build shortest paths for each pair (for dummy edges)
        # We need the actual path, not just the distance
        # Re-do BFS with parent tracking for each pair
        dummy_edges = []
        for (a, b) in pairs:
            # BFS from a to find shortest path to b
            dist = [-1] * n
            par = [-1] * n
            dist[a] = 0
            q = deque([a])
            while q:
                v = q.popleft()
                if v == b:
                    break
                for nb in adj[v]:
                    if dist[nb] < 0:
                        dist[nb] = dist[v] + 1
                        par[nb] = v
                        q.append(nb)
            # Reconstruct path
            path = []
            cur = b
            while cur != -1:
                path.append(cur)
                cur = par[cur]
            path.reverse()
            # Add edges along this path as dummy edges
            for i in range(len(path) - 1):
                dummy_edges.append((path[i], path[i+1]))

        # Build augmented graph for Eulerian circuit
        # We need to track which edges are real vs dummy
        # Use edge IDs: 0..len(edges)-1 are real, len(edges).. are dummy
        all_edges = list(edges) + dummy_edges
        m = len(all_edges)

        # Build adjacency with edge tracking
        # adj_e[v] = list of (neighbor, edge_id)
        adj_e = [[] for _ in range(n)]
        for eid, (u, v) in enumerate(all_edges):
            adj_e[u].append((v, eid))
            adj_e[v].append((u, eid))

        # Find start vertex
        start = 0
        while start < n and not adj_e[start]:
            start += 1
        if start >= n:
            return [list(e) for e in edges]  # fallback: each edge as a path

        # Hierholzer's algorithm
        used = [False] * m
        stack = [start]
        circuit = []

        while stack:
            v = stack[-1]
            found = False
            while adj_e[v]:
                nb, eid = adj_e[v].pop()
                if not used[eid]:
                    used[eid] = True
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())

        circuit.reverse()

        # Now split the circuit at dummy edges
        # A dummy edge has eid >= len(edges)
        real_edge_count = len(edges)
        paths = []
        current_path = [circuit[0]]
        
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            # Find the edge ID used
            # We need to determine if this transition is a dummy edge
            # Since we used Hierholzer, we need to track which edge was used
            # Let's re-do with edge tracking
            
            current_path.append(v)

        # The above approach loses edge identity. Let's redo properly.
        # Re-implement Hierholzer tracking edges in the circuit.
        
        # Actually, let's use a different approach: 
        # Build the circuit as a list of (vertex, edge_id) pairs
        used = [False] * m
        stack = [(start, -1)]  # (vertex, edge_id used to arrive)
        circuit_verts = []
        circuit_edges = []  # edge IDs in order

        while stack:
            v, in_eid = stack[-1]
            found = False
            while adj_e[v]:
                nb, eid = adj_e[v].pop()
                if not used[eid]:
                    used[eid] = True
                    if in_eid >= 0:
                        circuit_edges.append(in_eid)
                    circuit_verts.append(v)
                    stack.append((nb, eid))
                    found = True
                    break
            if not found:
                stack.pop()

        # This is getting complicated. Let me use a cleaner approach.
        # Re-do: standard Hierholzer that returns the sequence of vertices and edge IDs.
        
        # Clean reimplementation
        adj_e2 = [[] for _ in range(n)]
        for eid, (u, v) in enumerate(all_edges):
            adj_e2[u].append((v, eid))
            adj_e2[v].append((u, eid))

        used2 = [False] * m
        stack2 = [start]
        vert_seq = []
        edge_seq = []  # edge_seq[i] is the edge between vert_seq[i] and vert_seq[i+1]

        while stack2:
            v = stack2[-1]
            found = False
            while adj_e2[v]:
                nb, eid = adj_e2[v].pop()
                if not used2[eid]:
                    used2[eid] = True
                    stack2.append(nb)
                    found = True
                    break
            if not found:
                vert = stack2.pop()
                if vert_seq:
                    # The edge we're backtracking over
                    # Actually in Hierholzer, when we pop, we add to circuit
                    vert_seq.append(vert)
        
        # This standard approach gives vertices but not edge IDs directly.
        # Let me use the approach where we record edges as we traverse.
        
        # Simplest correct approach:
        # After finding the Eulerian circuit as a vertex sequence,
        # for each consecutive pair, determine if it's a real or dummy edge.
        # Since the graph may have multiple edges between same pair (real + dummy),
        # we need to be careful. But in our case, dummy edges are shortest paths
        # which may overlap with real edges.
        
        # Actually, the cleanest approach:
        # 1. Find Eulerian circuit in augmented multigraph.
        # 2. The circuit visits vertices in sequence.
        # 3. For each step in the circuit, we know which edge (real or dummy) was used.
        # 4. Split at dummy edges.
        
        # Let me just use a straightforward implementation:
        
        # Build adjacency as list of lists, each entry is (neighbor, edge_id)
        # Use a pointer for each vertex to track position in adjacency list
        adj_ptr = [0] * n
        for v in range(n):
            adj_e2[v] = sorted(adj_e2[v], key=lambda x: x[1])  # deterministic
        
        used3 = [False] * m
        stack3 = [start]
        verts = []
        
        while stack3:
            v = stack3[-1]
            found = False
            while adj_ptr[v] < len(adj_e2[v]):
                nb, eid = adj_e2[v][adj_ptr[v]]
                adj_ptr[v] += 1
                if not used3[eid]:
                    used3[eid] = True
                    stack3.append(nb)
                    found = True
                    break
            if not found:
                verts.append(stack3.pop())
        
        verts.reverse()
        # Now verts is the Eulerian circuit vertex sequence
        # verts[i] to verts[i+1] is connected by some edge
        # We need to figure out which edge ID connects consecutive vertices
        
        # Build a mapping: for each (u,v) pair, list of edge IDs
        edge_lookup = {}
        for eid, (u, v) in enumerate(all_edges):
            key = (min(u,v), max(u,v))
            if key not in edge_lookup:
                edge_lookup[key] = []
            edge_lookup[key].append(eid)
        
        # For each consecutive pair in the circuit, find the edge
        # and determine if it's dummy
        dummy_edge_ids = set(range(len(edges), m))
        
        # Track which edge IDs have been used in the circuit
        # Since it's an Eulerian circuit, each edge is used exactly once
        # We need to match them in order
        
        # This is tricky because multiple edges can connect the same pair.
        # Let's use a different strategy: just check if the edge between
        # consecutive vertices is a dummy edge by checking if that specific
        # (u,v) connection is in our dummy set.
        
        # Actually, since we added dummy edges as specific (u,v) pairs,
        # and real edges are also (u,v) pairs, there could be conflicts.
        # The safest approach: during circuit construction, record edge IDs.
        
        # Let me restart with a clean implementation that records edge IDs:
        
        adj_final = [[] for _ in range(n)]
        for eid, (u, v) in enumerate(all_edges):
            adj_final[u].append((v, eid))
            adj_final[v].append((u, eid))
        
        ptr = [0] * n
        used_f = [False] * m
        st = [start]
        vseq = []
        esequence = []  # edge IDs in order of traversal
        
        while st:
            v = st[-1]
            found = False
            while ptr[v] < len(adj_final[v]):
                nb, eid = adj_final[v][ptr[v]]
                ptr[v] += 1
                if not used_f[eid]:
                    used_f[eid] = True
                    st.append(nb)
                    found = True
                    break
            if not found:
                vseq.append(st.pop())
        
        vseq.reverse()
        
        # Now I need the edge sequence. In Hierholzer's algorithm,
        # the edge sequence is derived from the vertex sequence.
        # Between vseq[i] and vseq[i+1], there's exactly one edge used.
        # Let me just determine for each step whether it's a dummy edge
        # by checking against our dummy edge set.
        
        dummy_set = set()
        for (a, b) in dummy_edges:
            dummy_set.add((min(a,b), max(a,b)))
        
        # But this doesn't work if a real edge has the same endpoints as a dummy edge.
        # We need to track edge IDs properly.
        
        # Alternative cleaner approach: 
        # Instead of adding dummy edges and finding Eulerian circuit,
        # directly construct the paths from the matching.
        # 
        # For each pair (a, b) in the matching, the shortest path P(a,b) 
        # tells us where to "cut" the Eulerian structure.
        #
        # Actually, the standard result is:
        # - Add dummy edges along shortest paths between matched odd vertices.
        # - The augmented graph has all even degrees.
        # - Find Eulerian circuit.
        # - Remove dummy edges; the remaining segments are the paths.
        #
        # The issue is tracking which edges are dummy in the circuit.
        # Let me use edge IDs properly.
        
        # Rebuild with explicit edge ID tracking in the circuit
        adj_id = [[] for _ in range(n)]
        for eid, (u, v) in enumerate(all_edges):
            adj_id[u].append((v, eid))
            adj_id[v].append((u, eid))
        
        ptr_id = [0] * n
        used_id = [False] * m
        st_id = [start]
        vseq_id = []
        
        while st_id:
            v = st_id[-1]
            found = False
            while ptr_id[v] < len(adj_id[v]):
                nb, eid = adj_id[v][ptr_id[v]]
                ptr_id[v] += 1
                if not used_id[eid]:
                    used_id[eid] = True
                    st_id.append(nb)
                    found = True
                    break
            if not found:
                vseq_id.append(st_id.pop())
        
        vseq_id.reverse()
        
        # Now determine edge IDs for each step
        # For each consecutive pair (vseq_id[i], vseq_id[i+1]),
        # find which edge ID was used. Since each edge is used exactly once,
        # we can maintain a count.
        
        # Build lookup: (u,v) -> list of edge IDs (sorted)
        lookup = {}
        for eid, (u, v) in enumerate(all_edges):
            key = (min(u,v), max(u,v))
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(eid)
        for key in lookup:
            lookup[key].sort()
        
        # Pointer into lookup for each key
        lookup_ptr = {key: 0 for key in lookup}
        
        edge_seq = []
        for i in range(len(vseq_id) - 1):
            u, v = vseq_id[i], vseq_id[i+1]
            key = (min(u,v), max(u,v))
            eid = lookup[key][lookup_ptr[key]]
            lookup_ptr[key] += 1
            edge_seq.append(eid)
        
        # Now split at dummy edges
        real_count = len(edges)
        paths = []
        current = [vseq_id[0]]
        
        for i in range(len(vseq_id) - 1):
            current.append(vseq_id[i+1])
            if edge_seq[i] >= real_count:
                # This is a dummy edge: end current path
                if len(current) >= 2:
                    paths.append(current)
                current = [vseq_id[i+1]]
        
        if len(current) >= 2:
            paths.append(current)
        
        # Validate
        if not paths:
            return [list(e) for e in edges]
        
        edge_set = set(edges)
        covered = set()
        valid = True
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edge_set:
                    valid = False
                    break
                if e in covered:
                    valid = False
                    break
                covered.add(e)
            if not valid:
                break
        
        if not valid or covered != edge_set:
            # Fallback: greedy
            return greedy_path_cover(edges, n)
        
        return paths

    def eulerian_circuit(adj, start):
        """Find Eulerian circuit using Hierholzer's."""
        # adj[v] is list of neighbors (with duplicates for multi-edges)
        ptr = [0] * len(adj)
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            if ptr[v] < len(adj[v]):
                nb = adj[v][ptr[v]]
                ptr[v] += 1
                stack.append(nb)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def greedy_path_cover(edges, n):
        """Greedy fallback: repeatedly find longest path in remaining edges."""
        if not edges:
            return []
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Find longest simple path in remaining subgraph
            best_path = None
            best_len = 0
            
            # Try from each vertex with remaining edges
            for start in range(n):
                if not any((start, nb) in remaining or (nb, start) in remaining for nb in range(n) if nb != start):
                    continue
                
                # DFS to find longest path
                def dfs(v, visited, path):
                    nonlocal best_path, best_len
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = path[:]
                    
                    for nb in adj[v]:
                        if nb not in visited:
                            e = tuple(sorted((v, nb)))
                            if e in remaining:
                                visited.add(nb)
                                path.append(nb)
                                dfs(nb, visited, path)
                                path.pop()
                                visited.discard(nb)
                
                dfs(start, {start}, [start])
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                best_path = list(e)
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
            
            paths.append(best_path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        
        try:
            paths = min_path_cover(edges, n)
        except Exception:
            paths = greedy_path_cover(edges, n)
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
