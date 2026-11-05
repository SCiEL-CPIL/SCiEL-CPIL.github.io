"""Path cover via odd-vertex pairing + Eulerian trail extraction with iterative refinement."""
import time
import heapq
from collections import deque
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

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
        # BFS from each source; return dict source -> (dist, prev)
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
        # Hungarian algorithm for min cost perfect matching, k even, k small
        # costs: k x k symmetric with inf on diagonal
        # Returns list of k//2 pairs (i, j) with i < j
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
        # p[j] = row assigned to column j
        pairs = []
        for j in range(1, n + 1):
            i = p[j]
            if i > 0 and i < j:
                pairs.append((i - 1, j - 1))
        return pairs

    def extract_paths_from_multigraph(n, edge_list):
        """
        Given a list of undirected edges (u,v), find Eulerian trails/circuits.
        Returns a list of paths (each path is a list of vertices).
        The multigraph may have multiple edges between same vertices.
        """
        if not edge_list:
            return []
        
        # Build adjacency with edge indices
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edge_list):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        used = [False] * len(edge_list)
        paths = []
        
        # Find all vertices with odd degree
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        
        odd_verts = [v for v in range(n) if deg[v] % 2 == 1]
        
        if not odd_verts:
            # All even: single Eulerian circuit
            # Find any vertex with edges
            start = None
            for v in range(n):
                if deg[v] > 0:
                    start = v
                    break
            if start is None:
                return []
            # Hierholzer
            stack = [start]
            path = []
            while stack:
                u = stack[-1]
                found = None
                for (v, idx) in adj[u]:
                    if not used[idx]:
                        found = (v, idx)
                        break
                if found is None:
                    path.append(u)
                    stack.pop()
                else:
                    v, idx = found
                    used[idx] = True
                    stack.append(v)
            path.reverse()
            paths.append(path)
        else:
            # Odd degree vertices: need Eulerian trails
            # Pair them up (we'll do this via the matching above)
            # For now, just do Hierholzer starting from each odd vertex
            # Actually, for a connected multigraph with exactly 2 odd vertices,
            # there's a single Eulerian trail
            # For more odd vertices, we need to decompose
            
            # Since we're adding edges to make all degrees even (via matching),
            # this shouldn't happen. But let's handle it.
            # Find connected components and do Eulerian trails in each
            
            # For simplicity, assume the graph is connected (it should be)
            # and do a single Eulerian trail from one odd vertex to another
            if len(odd_verts) == 2:
                start = odd_verts[0]
                stack = [start]
                path = []
                while stack:
                    u = stack[-1]
                    found = None
                    for (v, idx) in adj[u]:
                        if not used[idx]:
                            found = (v, idx)
                            break
                    if found is None:
                        path.append(u)
                        stack.pop()
                    else:
                        v, idx = found
                        used[idx] = True
                        stack.append(v)
                path.reverse()
                paths.append(path)
            else:
                # More than 2 odd vertices: this means our matching didn't work
                # Fall back to individual edges
                for u, v in edge_list:
                    paths.append([u, v])
        
        return paths

    def can_merge(p1, p2, adj):
        """
        Check if p2 can be appended to p1 to form a simple path.
        Returns the new path if valid, else None.
        p1 and p2 are lists of vertices.
        """
        # The connection must be between an endpoint of p1 and an endpoint of p2
        # Case 1: p1[-1] connected to p2[0]
        # Case 2: p1[-1] connected to p2[-1] (reverse p2)
        # Case 3: p1[0] connected to p2[0] (reverse p1)
        # Case 4: p1[0] connected to p2[-1] (reverse both)
        
        # We can just check if there's an edge between any endpoint of p1 and any endpoint of p2
        # and if the resulting path is simple.
        
        # Optimization: Pre-check if sets of vertices overlap. If they do, we need to be careful.
        # Actually, if they overlap, they MUST share vertices, so the union is not a simple path
        # UNLESS the overlap is exactly at the connection point? No, simple path means all vertices distinct.
        # So if set(p1) & set(p2) is non-empty, can we merge?
        # Only if the intersection is exactly the connecting vertex? No, if v is in p1 and p2, and v is not the connecting vertex, then v appears twice.
        # If v IS the connecting vertex, e.g., p1 ends at v and p2 starts at v, then v appears once in the merged path.
        # So:
        # 1. Check if p1[-1] is in p2. If p1[-1] == p2[0], check if rest of p2 is disjoint from p1.
        # 2. Check if p1[-1] is in p2. If p1[-1] == p2[-1], check if rest of p2 (reversed) is disjoint from p1.
        # 3. Check if p1[0] is in p2. If p1[0] == p2[0], check if rest of p1 (reversed) is disjoint from p2.
        # 4. Check if p1[0] is in p2. If p1[0] == p2[-1], check if rest of p1 (reversed) is disjoint from p2 (reversed).
        
        # Simpler: Just try the 4 combinations and check simplicity.
        
        # To speed up, use sets.
        set1 = set(p1)
        set2 = set(p2)
        
        # If intersection size > 1, impossible (unless they are the same path? No, distinct paths in cover).
        # If intersection size == 1, let v be the common vertex.
        # v must be an endpoint of p1 AND an endpoint of p2.
        inter = set1 & set2
        if len(inter) > 1:
            return None
        if len(inter) == 1:
            v = next(iter(inter))
            # v must be p1[0] or p1[-1]
            if v != p1[0] and v != p1[-1]:
                return None
            # v must be p2[0] or p2[-1]
            if v != p2[0] and v != p2[-1]:
                return None
            
            # Now try the valid connection
            # Determine orientation
            # If v == p1[-1] and v == p2[0]: merge p1 + p2[1:]
            # If v == p1[-1] and v == p2[-1]: merge p1 + p2[-2::-1]
            # If v == p1[0] and v == p2[0]: merge p1[-2::-1] + p2
            # If v == p1[0] and v == p2[-1]: merge p1[-1::-1] + p2[-1::-1]
            
            if v == p1[-1] and v == p2[0]:
                # Check edge p1[-1]-p2[0] is trivial (same vertex), but we need to check the NEXT vertex?
                # No, p1 ends at v, p2 starts at v. The edge is (v, p2[1]).
                # We need (v, p2[1]) to be in adj.
                if len(p2) > 1:
                    if p2[1] not in adj[v]:
                        return None
                # Construct new path
                new_path = p1 + p2[1:]
                # It should be simple by construction if intersection was only v
                return new_path
            elif v == p1[-1] and v == p2[-1]:
                if len(p2) > 1:
                    if p2[-2] not in adj[v]:
                        return None
                new_path = p1 + p2[-2::-1]
                return new_path
            elif v == p1[0] and v == p2[0]:
                if len(p1) > 1:
                    if p1[-1] not in adj[v]: # Wait, p1 is reversed. The new start is p1[-1]. The edge is (v, p1[-1]).
                        # Actually, if we reverse p1, it becomes p1[-1], ..., v.
                        # The connection is v to p2[1].
                        pass
                # Let's just construct and check edge
                # Reversed p1: p1[-1::-1]
                # p2: p2
                # Connection is v (end of rev p1) to p2[1].
                if len(p2) > 1:
                    if p2[1] not in adj[v]:
                        return None
                new_path = p1[-1::-1] + p2[1:]
                return new_path
            elif v == p1[0] and v == p2[-1]:
                # Rev p1, Rev p2
                # End of rev p1 is v. Start of rev p2 is p2[-2].
                if len(p2) > 1:
                    if p2[-2] not in adj[v]:
                        return None
                new_path = p1[-1::-1] + p2[-1::-1][1:] # p2[-1::-1] starts with v. We want to append rest.
                # p2[-1::-1] is [v, p2[-2], ...]. We want [v] + [p2[-2]...]? No.
                # p1_rev ends at v. p2_rev starts at v.
                # Merged: p1_rev + p2_rev[1:]
                new_path = p1[-1::-1] + p2[-2::-1]
                return new_path
            else:
                return None
        else:
            # No common vertices.
            # Check 4 endpoint pairs
            # 1. p1[-1] - p2[0]
            if p2[0] in adj[p1[-1]]:
                return p1 + p2
            # 2. p1[-1] - p2[-1]
            if p2[-1] in adj[p1[-1]]:
                return p1 + p2[::-1]
            # 3. p1[0] - p2[0]
            if p2[0] in adj[p1[0]]:
                return p1[::-1] + p2
            # 4. p1[0] - p2[-1]
            if p2[-1] in adj[p1[0]]:
                return p1[::-1] + p2[::-1]
            return None

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        adj = build_adj(edges, n)
        
        # Initialize paths as individual edges
        paths = [[u, v] for u, v in edges]
        
        # Greedy merging
        # To ensure determinism and consistent improvement, we can sort or just iterate.
        # Since we want to minimize count, any merge is good.
        # We'll repeat until no more merges are possible.
        
        changed = True
        while changed:
            changed = False
            # We iterate over all pairs. O(N^2 * L).
            # N is number of paths. L is length.
            # To speed up, we can maintain a list of active paths.
            # When a merge happens, we remove the two and add the one.
            
            new_paths = []
            # This simple O(N^2) approach might be slow if N is large (e.g. 100 edges -> 5000 checks).
            # 5000 * 120s is fine.
            
            # However, modifying list while iterating is tricky.
            # Let's use a flag or a new list.
            # Actually, a better way:
            # Keep a list `paths`.
            # Iterate i from 0 to len(paths)-1.
            # If paths[i] is None, skip.
            # Iterate j from i+1 to len(paths)-1.
            # If paths[j] is None, skip.
            # Try merge. If success, replace paths[i] with new, paths[j] with None, changed=True.
            # Break inner loop (since paths[i] changed, need to re-check it against others? Or just continue to next i?)
            # If we break inner loop, we might miss merges with earlier j? No, j > i.
            # But the new path might merge with a k < i? 
            # To be safe and simple, we can just do passes.
            
            # Let's do a pass-based approach.
            # Pass 1: try to merge path i with path j (j > i).
            # If merged, mark j as dead, update i.
            # After pass, remove dead paths.
            # Repeat until no changes.
            
            n_paths = len(paths)
            dead = [False] * n_paths
            # We need to be careful if we update paths[i], it might merge with paths[j] where j < i?
            # No, we only check j > i.
            # But what if paths[i] merges with paths[j], and then the NEW paths[i] could merge with paths[k] where k < i?
            # That will be caught in the NEXT outer while loop iteration.
            
            for i in range(n_paths):
                if dead[i]:
                    continue
                for j in range(i + 1, n_paths):
                    if dead[j]:
                        continue
                    merged = can_merge(paths[i], paths[j], adj)
                    if merged is not None:
                        paths[i] = merged
                        dead[j] = True
                        changed = True
                        # We don't break inner loop immediately because the new paths[i] might merge with another j' > j?
                        # Yes, it's possible. So we continue.
            
            if changed:
                # Remove dead paths
                paths = [p for i, p in enumerate(paths) if not dead[i]]
        
        return paths

    def verify_and_fix(n, edges, paths):
        """Verify paths are valid and fix if needed."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        valid_paths = []
        
        for path in paths:
            # Check if path is simple
            if len(set(path)) != len(path):
                # Not simple, skip or fix
                continue
            # Check if all edges are in graph and not used
            valid = True
            path_edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                path_edges.append(e)
            if valid:
                valid_paths.append(path)
                used_edges.update(path_edges)
        
        # Check coverage
        missing = edge_set - used_edges
        if missing:
            # Add missing edges as individual paths
            for e in missing:
                valid_paths.append(list(e))
        
        return valid_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE - 5:
            # Fallback: trivial edge-as-path cover
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_one(n, p, seed)
        
        # Verify and fix
        paths = verify_and_fix(n, edges, paths)
        results.append(paths)
    
    return results


if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
    print("Total:", sum(len(lst) for lst in r))
