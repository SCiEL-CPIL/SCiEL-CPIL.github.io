import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if time.time() > deadline:
            # Fallback for remaining graphs
            # Generate graph and return all edges as singletons
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        
        # Build adjacency list and edge set
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        pieces = []

        def find_bridges(adj, n):
            """
            Returns a set of bridges (edges) in the graph defined by adj.
            Uses DFS with low-link values.
            """
            visited = [False] * n
            disc = [-1] * n
            low = [-1] * n
            parent = [-1] * n
            bridges = set()
            timer = [0]

            def dfs(u):
                visited[u] = True
                disc[u] = low[u] = timer[0]
                timer[0] += 1
                child_count = 0
                for v in list(adj[u]):
                    if not visited[v]:
                        child_count += 1
                        parent[v] = u
                        dfs(v)
                        low[u] = min(low[u], low[v])
                        if low[v] > disc[u]:
                            bridges.add((u, v) if u < v else (v, u))
                    elif v != parent[u]:
                        low[u] = min(low[u], disc[v])

            for i in range(n):
                if not visited[i]:
                    dfs(i)
            return bridges

        def find_longest_path(u, v, adj, n):
            """
            Finds the longest simple path from u to v in the graph adj.
            Returns the list of vertices in the path, or None if no path.
            """
            best_path = []
            visited = [False] * n
            current_path = [u]
            visited[u] = True

            def dfs(curr):
                nonlocal best_path
                if curr == v:
                    if len(current_path) > len(best_path):
                        best_path = current_path[:]
                    return

                # Pruning: if current path length + remaining nodes < best path length, stop
                # This is a weak bound, but helps.
                
                for nxt in list(adj[curr]):
                    if not visited[nxt]:
                        visited[nxt] = True
                        current_path.append(nxt)
                        dfs(nxt)
                        current_path.pop()
                        visited[nxt] = False

            dfs(u)
            return best_path

        while edge_set:
            if time.time() > deadline:
                for e in edge_set:
                    pieces.append(list(e))
                break

            # 1. Find and remove bridges
            bridges = find_bridges(adj, n)
            if bridges:
                for e in bridges:
                    u, v = e
                    if e in edge_set:
                        edge_set.remove(e)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        pieces.append([u, v])
                continue # Restart loop to re-evaluate bridges on the modified graph

            # 2. No bridges. Pick a random edge
            if not edge_set:
                break
                
            # To be deterministic-ish or just pick one, we can pick the first one
            # or randomize. Let's pick the first one in the set for determinism in debugging,
            # but random might help escape local minima? 
            # Actually, for "longest path", the choice of edge matters.
            # Let's pick a random edge to vary the search.
            import random
            edge = random.choice(list(edge_set))
            u, v = edge
            
            # 3. Find longest path from u to v in graph without (u,v)
            # Temporarily remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            path = find_longest_path(u, v, adj, n)
            
            # Restore (u,v) temporarily? No, we will remove the whole cycle.
            # If path is empty, it means no path was found, which shouldn't happen
            # if there are no bridges. But if it does, we restore and treat as bridge?
            if not path:
                # This implies (u,v) was a bridge, but we just checked for bridges.
                # This can happen if the bridge detection missed something or graph changed.
                # Just restore and add as singleton to be safe.
                adj[u].add(v)
                adj[v].add(u)
                edge_set.remove(edge)
                pieces.append([u, v])
                continue

            # Construct cycle
            cycle_nodes = path + [u] # path ends at v, so v -> u closes it? 
            # Path is u -> ... -> v. 
            # Cycle is u -> ... -> v -> u.
            # So nodes are path (which starts with u and ends with v) and we close to u.
            # The list of vertices for the piece is just `path`. 
            # Wait, the piece is a cycle. "cycle given by at least three distinct vertices (the closing edge is implied)".
            # So if path is [u, a, b, v], the cycle is u-a-b-v-u.
            # The vertices are [u, a, b, v].
            
            cycle_edges = set()
            for i in range(len(path)):
                a = path[i]
                b = path[(i + 1) % len(path)]
                e = tuple(sorted((a, b)))
                cycle_edges.add(e)
                
            # The edge (u,v) is the closing edge. 
            # Is (u,v) in cycle_edges?
            # path[0]=u, path[-1]=v.
            # The loop adds (path[i], path[i+1]).
            # The last pair added is (path[-2], path[-1]) = (x, v).
            # The closing edge is (v, u).
            # We need to add (u,v) to cycle_edges.
            e_close = tuple(sorted((u, v)))
            cycle_edges.add(e_close)

            # Remove cycle edges
            for e in cycle_edges:
                if e in edge_set:
                    edge_set.remove(e)
                    a, b = e
                    adj[a].remove(b)
                    adj[b].remove(a)
            
            pieces.append(path) # The list of vertices in the path defines the cycle

        results.append(pieces)

    return results
