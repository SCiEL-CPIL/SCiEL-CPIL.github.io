"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85.0

    # Dinic's Max Flow for vertex-disjoint paths (unit capacities on vertices)
    class Dinic:
        def __init__(self, n_nodes):
            self.n = n_nodes
            self.graph = [[] for _ in range(n_nodes)]
        
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
        
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, cap, _ in self.graph[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        if v == t:
                            return True
                        q.append(v)
            return self.level[t] >= 0
        
        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(len(self.graph[u])):
                e = self.graph[u][i]
                v, cap, rev = e
                if cap > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        e[1] -= d
                        self.graph[v][rev][1] += d
                        return d
            return 0
        
        def max_flow(self, s, t):
            flow = 0
            INF = 10**9
            while self.bfs(s, t):
                f = INF
                while f:
                    d = self.dfs(s, t, f)
                    if not d:
                        break
                    flow += d
                    f -= d
            return flow

    def max_vertex_disjoint_paths(src, dst, forbidden, adj_list):
        """
        Returns the maximum number of vertex-disjoint paths from src to dst
        avoiding forbidden vertices.
        Uses vertex splitting: v -> v_in, v_out.
        Capacity 1 on (v_in, v_out).
        src and dst have infinite capacity (or just 1 if we only care about existence, 
        but for counting, we need to allow multiple paths through src/dst if they are 
        sources/sinks. Here src/dst are just endpoints, so we don't split them or 
        give them high capacity).
        Actually, for vertex disjoint paths, internal vertices have cap 1.
        Src and Dst are not internal, so they don't constrain the count in the same way.
        """
        if src == dst:
            return 1
        if dst in forbidden or src in forbidden:
            return 0
            
        # Map original nodes to flow nodes
        # 0..n-1: in-nodes
        # n..2n-1: out-nodes
        N = 2 * n
        dinic = Dinic(N)
        
        # Add edges for original graph
        for u in range(n):
            if u in forbidden and u not in (src, dst):
                continue
            for v in adj_list[u]:
                if v in forbidden and v not in (src, dst):
                    continue
                # Edge from u_out to v_in
                dinic.add_edge(u + n, v, 1)
                dinic.add_edge(v + n, u, 1)
        
        # Add vertex capacity edges
        for u in range(n):
            if u in forbidden and u not in (src, dst):
                continue
            if u == src or u == dst:
                # Infinite capacity for source/sink
                dinic.add_edge(u, u + n, 100) 
            else:
                dinic.add_edge(u, u + n, 1)
        
        return dinic.max_flow(src, src + n) if False else dinic.max_flow(src, dst + n)

    def find_path_bfs(src, dst, forbidden, adj_list):
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nb in sorted(adj_list[node]):
                if nb == dst:
                    path = [dst]
                    cur = node
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    queue.append(nb)
        return None

    def try_build_subdivision(branches, deadline):
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        paths = {}
        used_internal = set()
        remaining_pairs = list(pairs)
        
        while remaining_pairs:
            if time.time() > deadline:
                return None
            
            # Find the most constrained pair
            min_flow = float('inf')
            best_pair = None
            for u, v in remaining_pairs:
                forbidden = branch_set - {u, v} | used_internal
                flow = max_vertex_disjoint_paths(u, v, forbidden, adj)
                if flow < min_flow:
                    min_flow = flow
                    best_pair = (u, v)
                
                if min_flow == 0:
                    return None
            
            u, v = best_pair
            forbidden = branch_set - {u, v} | used_internal
            
            # Find shortest path
            path = find_path_bfs(u, v, forbidden, adj)
            if path is None:
                return None
            
            # Validate path
            if len(path) < 2:
                return None
            if path[0] != u or path[-1] != v:
                return None
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return None
            for x in path[1:-1]:
                if x in branch_set or x in used_internal:
                    return None
            
            paths[(u, v)] = path
            for x in path[1:-1]:
                used_internal.add(x)
            
            remaining_pairs.remove((u, v))
        
        return [paths[(u, v)] for (u, v) in pairs]

    def find_max_clique(deadline):
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        cliques.sort(key=len, reverse=True)
        return cliques

    cliques = find_max_clique(deadline)
    best_result = None
    best_size = 0
    
    for clique in cliques:
        if time.time() > deadline:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        result = try_build_subdivision(clique, deadline)
        if result is not None:
            best_result = {'branches': list(clique), 'paths': result}
            best_size = len(clique)

    if best_size >= 2:
        base_branches = list(best_result['branches'])
    else:
        u, v = P['edges'][0]
        base_branches = [u, v]
        best_size = 2
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    current_branches = list(base_branches)
    
    for restart in range(5):
        if time.time() > deadline - 3:
            break
        
        if restart == 0:
            cur = list(current_branches)
        else:
            if cliques:
                idx = min(random.randint(0, len(cliques)-1), 20)
                cur = list(cliques[idx])
            else:
                cur = list(P['edges'][0])
        
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            
            candidates = [v for v in range(n) if v not in set(cur)]
            candidates.sort(key=lambda v: len(adj[v]))
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if time.time() > deadline - 3:
                    break
                new_branches = cur + [v]
                result = try_build_subdivision(new_branches, deadline)
                if result is not None:
                    cur = new_branches
                    if len(cur) > best_size:
                        best_size = len(cur)
                        best_result = {'branches': list(cur), 'paths': result}
                    added = True
                    break
            
            if not added:
                if len(cur) >= 3:
                    swapped = False
                    for i in range(len(cur)):
                        if time.time() > deadline - 3:
                            break
                        temp = cur[:i] + cur[i+1:]
                        cands2 = [v for v in range(n) if v not in set(temp)]
                        cands2.sort(key=lambda v: len(adj[v]))
                        random.shuffle(cands2)
                        for v in cands2[:20]:
                            if time.time() > deadline - 3:
                                break
                            new_b = temp + [v]
                            result = try_build_subdivision(new_b, deadline)
                            if result is not None:
                                cur = new_b
                                if len(cur) > best_size:
                                    best_size = len(cur)
                                    best_result = {'branches': list(cur), 'paths': result}
                                swapped = True
                                break
                        if swapped:
                            break

    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
