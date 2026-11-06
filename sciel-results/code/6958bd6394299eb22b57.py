"""Combined path-cover program: min-cost flow formulation with local polish."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_cost_flow_path_cover(edges, n, time_budget):
        """
        Model edge path cover as min-cost flow:
        - Each original edge (u,v) is a unit-capacity arc from u_out to v_in (and v_out to u_in for undirected) with cost 0.
        - For each vertex v, an arc v_in -> v_out with capacity 1, cost 0 (continuation).
        - Source S connects to every vertex_out with capacity 1, cost 1 (starting a path).
        - Every vertex_in connects to sink T with capacity 1, cost 0 (ending a path).
        We need to send |E| units of flow. The min cost equals the number of paths.
        We implement successive shortest paths with potentials (Johnson) using Dijkstra.
        """
        if not edges:
            return []

        edge_set = set(edges)
        m = len(edge_set)

        # Build the flow network
        # Nodes: S, T, v_in, v_out for v in 0..n-1
        # We'll index: 0 = S, 1 = T, 2..2n-1 = v_in, 2n..3n-1 = v_out
        S = 0
        T = 1
        def in_node(v): return 2 + v
        def out_node(v): return 2 + n + v
        N = 3 * n

        # adjacency list for residual graph: adj[node] = list of (neighbor, capacity, cost, reverse_index)
        adj = [[] for _ in range(N)]
        # We'll store edges as [to, cap, cost, rev_idx]
        def add_edge(u, v, cap, cost):
            adj[u].append([v, cap, cost, len(adj[v])])
            adj[v].append([u, 0, -cost, len(adj[u]) - 1])

        # Source to each vertex_out: cost 1, cap 1
        for v in range(n):
            add_edge(S, out_node(v), 1, 1)
        # Each vertex_in to sink: cost 0, cap 1
        for v in range(n):
            add_edge(in_node(v), T, 1, 0)
        # Vertex continuation: v_in -> v_out, cap 1, cost 0
        for v in range(n):
            add_edge(in_node(v), out_node(v), 1, 0)
        # For each original undirected edge (u,v):
        #   u_out -> v_in, cap 1, cost 0
        #   v_out -> u_in, cap 1, cost 0
        for u, v in edge_set:
            add_edge(out_node(u), in_node(v), 1, 0)
            add_edge(out_node(v), in_node(u), 1, 0)

        # Successive shortest paths with potentials
        INF = float('inf')
        flow = 0
        cost_total = 0
        potentials = [0] * N
        t0 = time.time()

        while flow < m:
            if time.time() - t0 > time_budget:
                break
            # Dijkstra with potentials
            dist = [INF] * N
            prev_node = [-1] * N
            prev_edge = [-1] * N
            dist[S] = 0
            heap = [(0, S)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for i, (v, cap, c, rev) in enumerate(adj[u]):
                    if cap <= 0:
                        continue
                    nd = d + c + potentials[u] - potentials[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_node[v] = u
                        prev_edge[v] = i
                        heapq.heappush(heap, (nd, v))
            if dist[T] == INF:
                break
            # Update potentials
            for v in range(N):
                if dist[v] < INF:
                    potentials[v] += dist[v]
            # Augment 1 unit along shortest path
            v = T
            while v != S:
                u = prev_node[v]
                i = prev_edge[v]
                adj[u][i][1] -= 1
                adj[v][adj[u][i][3]][1] += 1
                v = u
            flow += 1
            cost_total += potentials[T] - potentials[S]

        if flow < m:
            # Fallback: not all edges covered; use trivial
            return [[u, v] for u, v in edge_set]

        # Extract paths from flow
        # For each original edge, find which flow used it.
        # We need to trace: for each edge (u,v), we used either u_out->v_in or v_out->u_in.
        # Build a directed graph of used arcs among out/in nodes.
        # A path starts at S -> v_out, then follows out->in edges (original edges) and in->out (continuation), ends at v_in -> T.
        
        # Let's reconstruct by following flow.
        # Build used arcs:
        used_arcs = []  # list of (from, to) in the original graph nodes (out/in)
        # Check original edge arcs
        for u, v in edge_set:
            # Check u_out -> v_in
            idx = out_node(u)
            for i, (to, cap, c, rev) in enumerate(adj[idx]):
                if to == in_node(v) and adj[in_node(v)][rev][1] > 0:
                    # This arc has residual cap 0, meaning it was used
                    # Actually, we need to check if the reverse has cap > 0
                    if adj[in_node(v)][rev][1] == 1:
                        used_arcs.append((out_node(u), in_node(v)))
                        break
            # Check v_out -> u_in
            idx2 = out_node(v)
            for i, (to, cap, c, rev) in enumerate(adj[idx2]):
                if to == in_node(u) and adj[out_node(v)][i][1] == 0 and adj[in_node(u)][rev][1] == 1:
                    used_arcs.append((out_node(v), in_node(u)))
                    break

        # Now build adjacency from out_nodes: out_node -> in_node (via original edge)
        # And in_node -> out_node (via continuation, if used)
        # Let's build a directed graph on the n vertices:
        # For each used arc out_node(u) -> in_node(v), we have a directed edge u -> v.
        # For each used continuation in_node(v) -> out_node(v), we have v -> v (self-loop, but in path it's just continuation).
        
        # Actually, simpler: the flow decomposes into paths from S to T.
        # Each path is: S -> out_node(a) -> in_node(b) -> out_node(b) -> in_node(c) -> ... -> in_node(z) -> T
        # The corresponding vertex path is: a, b, c, ..., z
        # The edges are (a,b), (b,c), ..., (y,z)
        
        # Let's extract by following from S.
        paths = []
        # Find all out_nodes that have flow from S
        # adj[S] has arcs to out_nodes
        for i, (to, cap, c, rev) in enumerate(adj[S]):
            # Check if this arc was used (reverse has cap 1)
            if adj[to][rev][1] == 1:
                # Start a path from vertex 'to' (which is out_node(v), so v = to - 2 - n)
                v_start = to - 2 - n
                path = [v_start]
                cur = to  # current node is out_node(v_start)
                while True:
                    # From cur (an out_node), find the used arc to an in_node
                    found = False
                    for j, (to2, cap2, c2, rev2) in enumerate(adj[cur]):
                        if cap2 == 0 and adj[to2][rev2][1] == 1:
                            # This arc was used
                            # to2 should be an in_node
                            if 2 <= to2 < 2 + n:
                                v_next = to2 - 2
                                path.append(v_next)
                                cur = to2  # now at in_node(v_next)
                                found = True
                                break
                    if not found:
                        # Check if cur is an in_node with flow to T
                        if 2 <= cur < 2 + n:
                            # Check in_node -> T
                            for j, (to2, cap2, c2, rev2) in enumerate(adj[cur]):
                                if to2 == T and cap2 == 0 and adj[T][rev2][1] == 1:
                                    break
                        break
                    # From in_node, check continuation to out_node
                    if 2 <= cur < 2 + n:
                        v_cur = cur - 2
                        target_out = out_node(v_cur)
                        for j, (to2, cap2, c2, rev2) in enumerate(adj[cur]):
                            if to2 == target_out and cap2 == 0 and adj[target_out][rev2][1] == 1:
                                cur = target_out
                                break
                        else:
                            # No continuation, so this path ends here
                            break
                if len(path) >= 2:
                    paths.append(path)
        
        if not paths:
            # Fallback
            return [[u, v] for u, v in edge_set]
        
        return paths

    def greedy_path_cover(edges, n, time_budget):
        """Greedy fallback: repeatedly extract long paths."""
        if not edges:
            return []
        edge_set = set(edges)
        remaining = set(edge_set)
        paths = []
        t0 = time.time()
        
        while remaining and (time.time() - t0) < time_budget:
            # Build adjacency of remaining
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            if not adj:
                break
            
            # Find a long path using Warnsdorff
            nodes = list(adj.keys())
            random.shuffle(nodes)
            best_path = []
            for start in nodes[:min(len(nodes), 20)]:
                path = [start]
                visited = {start}
                cur = start
                while True:
                    nbrs = [nb for nb in adj[cur] if nb not in visited and (min(cur,nb), max(cur,nb)) in remaining]
                    if not nbrs:
                        break
                    # Warnsdorff: pick neighbor with fewest onward moves
                    def onward_count(nb):
                        cnt = 0
                        for nn in adj[nb]:
                            if nn not in visited and nn != cur and (min(nb,nn), max(nb,nn)) in remaining:
                                cnt += 1
                        return cnt
                    nbrs.sort(key=onward_count)
                    cur = nbrs[0]
                    path.append(cur)
                    visited.add(cur)
                if len(path) > len(best_path):
                    best_path = path[:]
            
            if not best_path or len(best_path) < 2:
                # Add remaining as single edges
                for e in remaining:
                    paths.append(list(e))
                break
            
            # Remove edges used in best_path
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                remaining.discard((min(u,v), max(u,v)))
            paths.append(best_path)
        
        for e in remaining:
            paths.append(list(e))
        return paths

    def validate_and_fix(paths, edges):
        """Ensure validity; fix by adding missing edges as singletons."""
        edge_set = set(edges)
        used = set()
        valid_paths = []
        for p in paths:
            if len(p) < 2:
                continue
            if len(set(p)) != len(p):
                # Not simple; break into single edges
                for i in range(len(p)-1):
                    e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                    if e in edge_set and e not in used:
                        used.add(e)
                        valid_paths.append([p[i], p[i+1]])
                continue
            ok = True
            for i in range(len(p)-1):
                e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                if e not in edge_set or e in used:
                    ok = False
                    break
                used.add(e)
            if ok:
                valid_paths.append(p)
        
        # Add missing edges
        for e in edge_set - used:
            valid_paths.append(list(e))
        return valid_paths

    def try_merge(paths, edge_set, time_budget):
        """Try to merge paths that can be concatenated."""
        t0 = time.time()
        changed = True
        while changed and (time.time() - t0) < time_budget:
            changed = False
            for i in range(len(paths)):
                if (time.time() - t0) > time_budget:
                    break
                for j in range(i + 1, len(paths)):
                    pi = paths[i]
                    pj = paths[j]
                    for perm_i in [pi, pi[::-1]]:
                        for perm_j in [pj, pj[::-1]]:
                            if perm_i[-1] == perm_j[0]:
                                new_path = perm_i + perm_j[1:]
                                if len(set(new_path)) != len(new_path):
                                    continue
                                valid = True
                                for k in range(len(new_path)-1):
                                    u, v = new_path[k], new_path[k+1]
                                    e = (min(u,v), max(u,v))
                                    if e not in edge_set:
                                        valid = False
                                        break
                                if valid:
                                    paths[i] = new_path
                                    del paths[j]
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        n, p, seed = spec
        g_edges = instance(n, p, seed)
        m = len(g_edges)
        time_left = deadline - time.time()
        
        if time_left < 2:
            paths = [[u, v] for u, v in g_edges]
        else:
            # Try min-cost flow first with a time budget
            flow_budget = min(15.0, time_left / 3.0 * 0.6)
            paths = min_cost_flow_path_cover(g_edges, n, flow_budget)
            
            if not paths:
                # Fallback to greedy
                greedy_budget = min(10.0, time_left / 3.0 * 0.3)
                paths = greedy_path_cover(g_edges, n, greedy_budget)
            
            # Validate and fix
            paths = validate_and_fix(paths, g_edges)
            
            # Try to merge for further improvement
            merge_budget = min(5.0, time_left / 3.0 * 0.1)
            if merge_budget > 0.5:
                paths = try_merge(paths, set(g_edges), merge_budget)
                paths = validate_and_fix(paths, g_edges)
        
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
