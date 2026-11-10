import time
import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_tree(adj, n):
        """
        Decompose a tree (or forest) into minimum number of simple paths.
        adj: dict of sets of neighbors
        Returns list of paths
        """
        paths = []
        # Work on a copy of adj to remove edges
        # Actually, we need to track edges.
        # Let's use a set of edges for the current component.
        
        # This function assumes adj represents a tree.
        # We find the diameter, remove it, and recurse on components.
        
        def find_diameter(adj):
            if not adj:
                return []
            
            # BFS from arbitrary node to find farthest
            start = next(iter(adj))
            # If isolated node, return [start]
            if not adj[start]:
                return [start]
                
            def bfs(start):
                visited = {start}
                parent = {start: None}
                queue = [start]
                farthest = start
                while queue:
                    u = queue.pop(0)
                    if len(parent[u]) > len(parent[farthest]) if 'len' in dir() else True: # Placeholder
                        pass
                # Proper BFS
                dist = {start: 0}
                par = {start: -1}
                q = [start]
                while q:
                    u = q.pop(0)
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            par[v] = u
                            q.append(v)
                # Find farthest
                farthest = max(dist, key=dist.get)
                return farthest, par, dist
            
            far1, _, _ = bfs(start)
            far2, par, _ = bfs(far1)
            
            # Reconstruct path from far2 to far1
            path = []
            curr = far2
            while curr != -1:
                path.append(curr)
                curr = par[curr]
            path.reverse()
            return path

        # We need to handle the graph as a set of edges to remove them properly
        # This recursive tree decomposition is tricky with just adjacency lists because
        # removing edges changes the structure.
        
        # Let's stick to the general DFS approach for simplicity, but optimize it.
        # The tree case can be handled by the general approach too, likely efficiently.
        
        return paths # Fallback, will use general method

    def decompose_general(edges, n):
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        start_time = time.time()
        time_limit = 10.0 # 10 seconds per graph
        
        while remaining:
            if time.time() - start_time > time_limit:
                # Fallback: add remaining edges as single-edge paths
                for u, v in remaining:
                    paths.append([u, v])
                break
                
            # Pick start vertex
            # Heuristic: highest degree
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            start = max(candidates, key=lambda x: len(adj[x]))
            
            best_path = [start]
            best_edges = set()
            
            # DFS to find longest simple path
            # Use iterative or recursive with limit
            # Recursive is fine for N=30
            visited = {start}
            
            def dfs(path, current_edges):
                nonlocal best_path, best_edges
                
                # Update best if longer
                if len(path) > len(best_path):
                    best_path = path[:]
                    best_edges = set(current_edges)
                    
                # Pruning: if we can't beat best, return
                # Max possible length is n
                if len(path) + (n - len(visited)) <= len(best_path):
                    return
                
                last = path[-1]
                # Shuffle neighbors to randomize search
                neighbors = list(adj[last])
                random.shuffle(neighbors)
                
                for nb in neighbors:
                    if nb in visited:
                        continue
                    e = (min(last, nb), max(last, nb))
                    if e in remaining:
                        visited.add(nb)
                        path.append(nb)
                        current_edges.add(e)
                        dfs(path, current_edges)
                        path.pop()
                        current_edges.remove(e)
                        visited.remove(nb)
                        
            dfs(best_path, set())
            
            # Add path
            if best_path:
                paths.append(best_path)
                for e in best_edges:
                    remaining.remove(e)
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
            else:
                # Should not happen if remaining is not empty
                # But if start has no neighbors in remaining?
                # adj[start] should be empty then, so it wouldn't be in candidates.
                break

        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        # If graph is empty, return []
        if not edges:
            results.append([])
            continue
            
        paths = decompose_general(edges, n)
        results.append(paths)
        
    return results
