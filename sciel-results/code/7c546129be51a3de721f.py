"""Rewritten program: Bipartite matching for path cover + stitching via local search."""
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    def maximum_bipartite_matching(adj, n):
        """Find maximum matching in bipartite graph where left=right=vertices, edge (u,v) exists if u-v is an edge in original graph.
        Returns dict mapping left node to right node for matched pairs."""
        # Use Hopcroft-Karp algorithm for O(E*sqrt(V)) matching
        from collections import defaultdict
        
        # Build bipartite adjacency: left[i] -> list of right nodes
        left_adj = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                left_adj[u].append(v)
        
        # Hopcroft-Karp
        pair_u = [-1] * n  # pair_u[u] = v means u is matched to v
        pair_v = [-1] * n  # pair_v[v] = u means v is matched to u
        dist = [0] * n
        
        def bfs():
            """BFS to find augmenting paths. Returns True if there's an augmenting path."""
            from collections import deque
            q = deque()
            for u in range(n):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in left_adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found
        
        def dfs(u):
            """DFS to find augmenting path from u."""
            for v in left_adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching_size = 0
        while bfs():
            for u in range(n):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        return pair_u, pair_v, matching_size
    
    def extract_paths_from_matching(pair_u, pair_v, adj, n):
        """Given a matching, extract vertex-disjoint paths.
        Each vertex has at most one outgoing edge (pair_u) and at most one incoming edge (pair_v).
        This forms a collection of vertex-disjoint paths and cycles."""
        # Build successor and predecessor maps
        succ = {}  # succ[u] = v if u->v is in matching
        pred = {}  # pred[v] = u if u->v is in matching
        
        for u in range(n):
            if pair_u[u] != -1:
                succ[u] = pair_u[u]
                pred[pair_u[u]] = u
        
        # Find path starts (vertices with no predecessor)
        starts = []
        for v in range(n):
            if v not in pred:
                starts.append(v)
        
        # Trace paths from each start
        paths = []
        visited = set()
        for start in starts:
            if start in visited:
                continue
            path = [start]
            visited.add(start)
            current = start
            while current in succ:
                current = succ[current]
                if current in visited:
                    break  # Should not happen in a proper matching, but safety
                path.append(current)
                visited.add(current)
            if len(path) > 1:
                paths.append(path)
        
        # Handle isolated vertices (no edges in matching)
        # They are single-vertex paths
        for v in range(n):
            if v not in visited:
                paths.append([v])
                visited.add(v)
        
        # Remove single-vertex paths for now, keep track of them
        multi_paths = [p for p in paths if len(p) > 1]
        single_paths = [p for p in paths if len(p) == 1]
        
        return multi_paths, single_paths
    
    def stitch_paths(paths, adj, deadline):
        """Try to stitch multiple paths into one long path by finding connections between endpoints."""
        if not paths:
            return []
        
        # Sort paths by length descending
        paths = sorted(paths, key=len, reverse=True)
        
        best_path = paths[0]
        remaining = paths[1:]
        used_vertices = set(best_path)
        
        # Try to attach each remaining path to the main path
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(remaining)):
                rp = remaining[i]
                # Try to attach rp to the end of best_path
                # Check if last of best_path connects to first of rp
                if best_path[-1] in adj[rp[0]]:
                    best_path = best_path + rp
                    used_vertices.update(rp)
                    remaining.pop(i)
                    improved = True
                    break
                # Try to attach rp to the front of best_path
                if best_path[0] in adj[rp[-1]]:
                    best_path = rp[::-1] + best_path
                    used_vertices.update(rp)
                    remaining.pop(i)
                    improved = True
                    break
                # Try reversing rp and attaching to end
                rp_rev = rp[::-1]
                if best_path[-1] in adj[rp_rev[0]]:
                    best_path = best_path + rp_rev
                    used_vertices.update(rp)
                    remaining.pop(i)
                    improved = True
                    break
                # Try reversing rp and attaching to front
                if best_path[0] in adj[rp_rev[-1]]:
                    best_path = rp_rev + best_path
                    used_vertices.update(rp)
                    remaining.pop(i)
                    improved = True
                    break
        
        # Now try to insert remaining paths into the middle of best_path
        for rp in remaining:
            if time.time() > deadline:
                break
            for i in range(len(best_path) - 1):
                u = best_path[i]
                w = best_path[i + 1]
                # Can we insert rp between u and w?
                # Need: u connects to rp[0] or rp[-1], and rp[-1] or rp[0] connects to w
                if u in adj[rp[0]] and rp[-1] in adj[w]:
                    best_path = best_path[:i+1] + rp + best_path[i+1:]
                    used_vertices.update(rp)
                    improved = True
                    break
                if u in adj[rp[-1]] and rp[0] in adj[w]:
                    best_path = best_path[:i+1] + rp[::-1] + best_path[i+1:]
                    used_vertices.update(rp)
                    improved = True
                    break
                if improved:
                    break
        
        return best_path
    
    def local_search_improve(path, adj, deadline):
        """Improve path via insertion of unvisited vertices and extension."""
        path = list(path)
        visited = set(path)
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to extend at ends
            changed = True
            while changed:
                changed = False
                # Extend front
                front = path[0]
                for v in sorted(adj[front]):
                    if v not in visited:
                        path.insert(0, v)
                        visited.add(v)
                        changed = True
                        improved = True
                        break
                # Extend back
                if not changed:
                    back = path[-1]
                    for v in sorted(adj[back]):
                        if v not in visited:
                            path.append(v)
                            visited.add(v)
                            changed = True
                            improved = True
                            break
            
            if not improved:
                break
            
            # 2. Try to insert unvisited vertices
            found_insertion = False
            for i in range(len(path) - 1):
                u = path[i]
                w = path[i + 1]
                common = adj[u] & adj[w]
                candidates = [v for v in common if v not in visited]
                if candidates:
                    v = candidates[0]
                    path.insert(i + 1, v)
                    visited.add(v)
                    found_insertion = True
                    improved = True
                    break
            
            if not found_insertion:
                break
        
        return path
    
    def warnsdorff_dfs(start, adj, deadline):
        """DFS with Warnsdorff heuristic."""
        best = [start]
        path = [start]
        visited = {start}
        
        def dfs():
            nonlocal best, path, visited
            if time.time() > deadline:
                return
            if len(path) > len(best):
                best = list(path)
            
            current = path[-1]
            candidates = [v for v in adj[current] if v not in visited]
            if not candidates:
                return
            
            candidates.sort(key=lambda v: (len(adj[v] - visited), v))
            
            for v in candidates:
                if time.time() > deadline:
                    return
                path.append(v)
                visited.add(v)
                dfs()
                path.pop()
                visited.discard(v)
        
        dfs()
        return best
    
    # Strategy:
    # 1. Use bipartite matching to find maximum vertex-disjoint path cover
    # 2. Stitch the paths together
    # 3. Improve via local search
    # 4. If time remains, try Warnsdorff DFS from multiple starts and compare
    
    best_path = []
    
    # Step 1: Bipartite matching approach
    pair_u, pair_v, match_size = maximum_bipartite_matching(adj, n)
    multi_paths, single_paths = extract_paths_from_matching(pair_u, pair_v, adj, n)
    
    # Stitch all paths
    all_paths = multi_paths + single_paths
    stitched = stitch_paths(all_paths, adj, deadline)
    
    if stitched and len(stitched) > len(best_path):
        best_path = stitched
    
    # Step 2: Local search improvement
    if best_path:
        best_path = local_search_improve(best_path, adj, deadline)
    
    # Step 3: If time remains, try Warnsdorff DFS from multiple starts
    rng = random.Random(42)
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline - 5.0:
            break
        if len(adj[start]) == 0:
            continue
        
        path = warnsdorff_dfs(start, adj, deadline)
        path = local_search_improve(path, adj, deadline)
        
        if len(path) > len(best_path):
            best_path = path
    
    # Final polish
    if best_path:
        best_path = local_search_improve(best_path, adj, deadline)
    
    if not best_path:
        best_path = [0]
    
    return best_path
