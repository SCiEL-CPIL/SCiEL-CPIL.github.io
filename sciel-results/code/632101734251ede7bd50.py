import time
import heapq
from collections import deque
from random import Random
import sys

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def odd_vertices(adj, n):
        deg = [len(a) for a in adj]
        return [v for v in range(n) if deg[v] % 2 == 1]

    def all_pairs_dists(adj, n, sources):
        res = {}
        for s in sources:
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            res[s] = (dist, prev)
        return res

    def min_cost_matching(costs, k):
        if k == 0:
            return []
        INF = 10**9
        n = k
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        pairs = []
        for j in range(1, n + 1):
            i = p[j]
            if i > 0 and i < j:
                pairs.append((i - 1, j - 1))
        return pairs

    def get_eulerian_trails(adj_mult, n):
        # adj_mult: list of lists of (neighbor, edge_id)
        # Returns list of trails (list of vertices)
        used = [False] * len(adj_mult[0]) if n > 0 else []
        # Note: adj_mult structure needs to be consistent. 
        # Let's use a global edge list and adjacency of edge IDs.
        pass

    def decompose_trail_to_paths(trail):
        # trail is a list of vertices
        # Returns list of simple paths (lists of vertices)
        if not trail:
            return []
        paths = []
        remaining = trail[:]
        while remaining:
            path = [remaining[0]]
            visited_in_path = {remaining[0]}
            i = 1
            while i < len(remaining):
                next_v = remaining[i]
                if next_v in visited_in_path:
                    break
                path.append(next_v)
                visited_in_path.add(next_v)
                i += 1
            paths.append(path)
            # The rest of the trail starts from the last vertex added?
            # No, the trail is a sequence. 
            # If we break at i, the next path should start from remaining[i-1]?
            # Actually, the trail is connected.
            # If we take path P1 = v0...vk, the next part of the trail starts at vk?
            # Yes, because the trail is continuous.
            # So we remove v0...vk-1 from remaining?
            # remaining = remaining[k-1:]
            # But we need to ensure edge disjointness.
            # The edges in path P1 are (v0,v1)...(vk-1,vk).
            # The next path should start with edge (vk, v_{k+1})?
            # So the next path starts at vk.
            # So remaining = remaining[k-1:]
            remaining = remaining[len(path)-1:]
            if not remaining:
                break
        return paths

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        adj = build_adj(edges, n)
        odds = odd_vertices(adj, n)
        k = len(odds)
        
        # Map original edges to IDs
        edge_id = {}
        for idx, (u, v) in enumerate(edges):
            edge_id[(u, v)] = idx
            edge_id[(v, u)] = idx

        if k == 0:
            # Eulerian. Add dummy edge?
            # If we add a dummy edge between two vertices, we get 1 trail.
            # Then remove dummy edge -> 1 path?
            # If we pick u, v, add (u,v).
            # Trail starts at u, ends at v? No, Eulerian circuit.
            # If we add (u,v), u and v become odd.
            # Eulerian trail from u to v.
            # Remove (u,v). The trail is broken?
            # No, the trail uses (u,v) at some point.
            # If we remove it, we get a path from u to v?
            # Yes, if (u,v) is at the start or end?
            # Not necessarily.
            # If (u,v) is in the middle, removing it splits the trail into two paths?
            # Or one path if we consider the circuit?
            # Actually, for Eulerian graph, min paths is 1? No, 2?
            # Let's just add a dummy edge, find trail, remove dummy edge.
            # If the dummy edge is used, we split.
            pass

        # Standard case: k > 0
        dists = all_pairs_dists(adj, n, odds)
        INF = 10**9
        costs = [[INF] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j: continue
                d = dists[odds[i]][0][odds[j]]
                if d >= 0:
                    costs[i][j] = d
                else:
                    costs[i][j] = INF

        pairs = min_cost_matching(costs, k)
        
        # Build augmented multigraph
        # We need to track edges to ensure edge-disjointness in the final output.
        # The final output must be simple paths in the ORIGINAL graph.
        # So we cannot use the added edges in the output.
        
        # Let's build the multigraph adjacency with edge IDs.
        # Original edges: 0 to len(edges)-1
        # Added edges: len(edges) to ...
        
        mult_adj = [[] for _ in range(n)]
        all_edges = [] # list of (u, v, is_original, orig_id)
        
        for idx, (u, v) in enumerate(edges):
            all_edges.append((u, v, True, idx))
            mult_adj[u].append((v, len(all_edges)-1))
            all_edges.append((v, u, True, idx))
            mult_adj[v].append((u, len(all_edges)-1))

        for (i, j) in pairs:
            u, v = odds[i], odds[j]
            dist, prev = dists[u]
            path = []
            cur = v
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            for a, b in zip(path, path[1:]):
                eid = len(all_edges)
                all_edges.append((a, b, False, -1))
                mult_adj[a].append((b, eid))
                all_edges.append((b, a, False, -1))
                mult_adj[b].append((a, eid+1))

        # Find Eulerian trails in mult_adj
        # We need to find trails that cover all edges in the multigraph.
        # Since the multigraph is Eulerian (all degrees even), we can find Eulerian circuits.
        # But we want trails?
        # If we start at an odd vertex of the ORIGINAL graph (which is now even in multigraph?),
        # Wait. Original odd vertices become even after adding edges.
        # So all vertices are even.
        # So we have Eulerian circuits.
        # We can start anywhere.
        # Let's find all Eulerian circuits.
        
        used = [False] * len(all_edges)
        trails = []
        
        for start in range(n):
            if not any(not used[eid] for (v, eid) in mult_adj[start]):
                continue
            # Start a new circuit
            stack = [start]
            trail = []
            while stack:
                u = stack[-1]
                found = None
                for (v, eid) in mult_adj[u]:
                    if not used[eid]:
                        found = (v, eid)
                        break
                if found is None:
                    trail.append(u)
                    stack.pop()
                else:
                    v, eid = found
                    used[eid] = True
                    used[eid ^ 1] = True
                    stack.append(v)
            trail.reverse()
            trails.append(trail)

        # Now we have trails in the multigraph.
        # Each trail uses some original edges and some added edges.
        # We need to convert these to simple paths in the original graph.
        # We iterate through the trail.
        # We maintain a current simple path.
        # If the next edge is an original edge and the next vertex is not in the current path, extend.
        # If the next edge is an added edge, we must break?
        # No, added edges are "virtual".
        # The trail is a sequence of vertices.
        # The edges between them are either original or added.
        # We want to output paths that use ONLY original edges.
        # So we can only traverse original edges.
        # If we encounter an added edge, we cannot include it in the output path.
        # So we must split the trail at the added edges.
        
        final_paths = []
        
        for trail in trails:
            # trail is a list of vertices
            # We need to know which edges were used.
            # The trail is a sequence v0, v1, ..., vk.
            # The edge between vi and vi+1 is either original or added.
            # We need to reconstruct the edge IDs used.
            # This is tricky with Hierholzer which doesn't store edge IDs in the trail directly.
            # Let's modify Hierholzer to store edge IDs.
            
            pass

        # Alternative:
        # Instead of extracting trails and then splitting,
        # we can simulate the traversal and split whenever we hit an added edge or a repeated vertex.
        
        # Let's redo the extraction to keep track of edges.
        
        used2 = [False] * len(all_edges)
        final_paths = []
        
        for start in range(n):
            if not any(not used2[eid] for (v, eid) in mult_adj[start]):
                continue
            stack = [(start, -1)] # (vertex, incoming_edge_id)
            trail_edges = [] # list of edge_ids
            trail_vertices = [start]
            
            while stack:
                u, in_eid = stack[-1]
                found = None
                for (v, eid) in mult_adj[u]:
                    if not used2[eid]:
                        found = (v, eid)
                        break
                if found is None:
                    # backtrack
                    stack.pop()
                    if in_eid != -1:
                        # This edge is part of the circuit
                        # We need to insert it into the circuit
                        # Standard Hierholzer insertion is complex.
                        pass
                else:
                    v, eid = found
                    used2[eid] = True
                    used2[eid ^ 1] = True
                    stack.append((v, eid))
                    trail_vertices.append(v)
                    trail_edges.append(eid)

            # This simple stack approach doesn't correctly build the circuit if there are sub-cycles.
            # Let's use a different method:
            # Since n is small, we can just use a recursive DFS or similar?
            # Or use the standard algorithm and record edges.
            
            pass

        # Given the complexity, let's use a simpler heuristic:
        # 1. Find the matching.
        # 2. Add edges.
        # 3. Find Eulerian trails.
        # 4. For each trail, greedily extract simple paths using ONLY original edges.
        #    When we hit an added edge, we stop the current path and start a new one from the next vertex?
        #    No, the added edge connects two vertices.
        #    If we stop, the next path starts from the other end of the added edge?
        #    Yes.
        
        # Let's implement `extract_simple_paths_from_trail`
        
        def extract_simple_paths(trail, edge_ids):
            # trail: list of vertices
            # edge_ids: list of edge ids corresponding to edges in trail
            # Returns list of simple paths (vertex lists)
            paths = []
            i = 0
            n_trail = len(trail)
            while i < n_trail:
                path = [trail[i]]
                visited = {trail[i]}
                j = i
                while j < n_trail - 1:
                    eid = edge_ids[j]
                    u, v, is_orig, _ = all_edges[eid]
                    if not is_orig:
                        break
                    next_v = trail[j+1]
                    if next_v in visited:
                        break
                    path.append(next_v)
                    visited.add(next_v)
                    j += 1
                paths.append(path)
                # Next path starts from trail[j+1]?
                # No, the edge j was (trail[j], trail[j+1]).
                # If we broke because of added edge or repeated vertex,
                # The next path should start from trail[j+1]?
                # If we broke because of added edge, the added edge is (trail[j], trail[j+1]).
                # We skip it. Next path starts at trail[j+1].
                # If we broke because of repeated vertex, we are at trail[j+1] which is in visited.
                # We start a new path at trail[j+1].
                i = j + 1
            return paths

        # We need to get edge_ids from the trail.
        # This requires a Hierholzer variant that outputs edge IDs.
        
        # Let's implement a proper Hierholzer that returns the sequence of edges.
        
        def get_eulerian_trails_with_edges(mult_adj, n, all_edges):
            used = [False] * len(all_edges)
            trails = []
            for start in range(n):
                if not any(not used[eid] for (v, eid) in mult_adj[start]):
                    continue
                stack = [start]
                trail_verts = []
                trail_eids = []
                while stack:
                    u = stack[-1]
                    found = None
                    for (v, eid) in mult_adj[u]:
                        if not used[eid]:
                            found = (v, eid)
                            break
                    if found is None:
                        trail_verts.append(stack.pop())
                    else:
                        v, eid = found
                        used[eid] = True
                        used[eid ^ 1] = True
                        stack.append(v)
                        trail_eids.append(eid)
                trail_verts.reverse()
                trail_eids.reverse()
                # The length of trail_eids should be len(trail_verts) - 1
                trails.append((trail_verts, trail_eids))
            return trails

        trails = get_eulerian_trails_with_edges(mult_adj, n, all_edges)
        
        for tv, te in trails:
            paths = extract_simple_paths(tv, te)
            final_paths.extend(paths)
            
        return final_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            results.append([[u, v] for u, v in instance(*spec)])
            continue
        paths = solve_one(*spec)
        results.append(paths)
    return results

if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
