import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 82.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # Dinic's max flow implementation
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.graph = [[] for _ in range(N)]

        def add_edge(self, u, v, c):
            self.graph[u].append([v, c, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])

        def bfs(self, s, t):
            self.level = [-1] * self.N
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, c, rev in self.graph[u]:
                    if c > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            while self.it[u] < len(self.graph[u]):
                v, c, rev = self.graph[u][self.it[u]]
                if c > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, t, min(f, c))
                    if d > 0:
                        self.graph[u][self.it[u]][1] -= d
                        self.graph[v][rev][1] += d
                        return d
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = float('inf')
            while self.bfs(s, t):
                self.it = [0] * self.N
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    def check_feasibility(branches):
        """
        Check if branch set is feasible using max-flow.
        Returns (feasible, paths_dict) where paths_dict maps (i,j) -> path.
        
        Flow network:
        - For each vertex v: v_in, v_out
        - For non-branch v: edge v_in -> v_out with cap 1
        - For branch v: edge v_in -> v_out with cap INF
        - For each edge (u,v) in original graph:
            * If u,v both branches: u_out -> v_in cap 1, v_out -> u_in cap 1
            * If u branch, v non-branch: u_out -> v_in cap 1
            * If u non-branch, v branch: u_out -> v_in cap 1
            * If both non-branch: u_out -> v_in cap 1
        - Super source S, super sink T
        - For each branch a: S -> a_in cap (k-1), a_out -> T cap (k-1)
        
        If max_flow >= C(k,2), feasible.
        """
        k = len(branches)
        if k < 2:
            return False, None
        need = k * (k - 1) // 2
        
        # Vertex splitting: v_in = 2*v, v_out = 2*v+1
        # S = 2*n, T = 2*n+1
        N = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        INF = 10**9
        
        dinic = Dinic(N)
        branch_set = set(branches)
        
        # Add split edges
        for v in range(n):
            if v in branch_set:
                dinic.add_edge(2*v, 2*v+1, INF)
            else:
                dinic.add_edge(2*v, 2*v+1, 1)
        
        # Add original edges (directed both ways)
        for u, v in edges:
            dinic.add_edge(2*u+1, 2*v, 1)
            dinic.add_edge(2*v+1, 2*u, 1)
        
        # Add source/sink edges for branches
        for a in branches:
            dinic.add_edge(S, 2*a, k-1)
            dinic.add_edge(2*a+1, T, k-1)
        
        flow = dinic.max_flow(S, T)
        
        if flow < need:
            return False, None
        
        # Extract paths from the flow
        # We need to find which paths carry flow
        # For each pair (i,j), find a path from branches[i] to branches[j] in the flow
        paths_dict = {}
        branch_index = {b: idx for idx, b in enumerate(branches)}
        
        # Build residual graph for path extraction
        # We need to find actual paths in the original graph
        
        # Strategy: for each pair, do a DFS in the residual graph
        # following edges with positive flow (original capacity - residual capacity > 0)
        
        # Actually, let's use a different approach:
        # Reconstruct by finding paths in the flow network
        # For each branch pair, find a path through the flow
        
        # Simpler approach: use the flow to guide BFS in original graph
        # Mark which edges have flow > 0
        edge_flow = {}
        for u in range(N):
            for v, c, rev in dinic.graph[u]:
                orig_cap = dinic.graph[v][rev][1]  # This is the reverse edge's current cap = original forward flow
                # Actually, the reverse edge's capacity equals the flow on the forward edge
                # Forward edge: u->v with current cap c
                # Reverse edge: v->u with current cap = original_cap - c = flow
                # So flow on u->v = original_cap - c
                # But we don't easily have original_cap stored separately...
                
        # Let's use a cleaner approach: rebuild the flow network and track flows
        # Actually, let's just do BFS in the original graph guided by the flow
        
        # For each pair (i, j), find a path from branches[i] to branches[j]
        # that uses only edges with positive flow in the direction of travel
        # and only uses non-branch vertices that have flow through them
        
        # Build a directed graph of "flow edges"
        # An edge u->v has flow if the reverse edge v->u has capacity > 0
        # (since reverse cap = original flow)
        
        flow_graph = [set() for _ in range(n)]
        for u in range(N):
            for v, c, rev in dinic.graph[u]:
                # v is the reverse edge's target
                # The reverse edge is graph[v][rev]
                rev_cap = dinic.graph[v][rev][1]
                if rev_cap > 0:
                    # There is flow on edge u->v
                    # Map back to original vertices
                    if u < 2*n and v < 2*n:
                        u_orig = u // 2
                        v_orig = v // 2
                        # Determine direction in original graph
                        # u_out -> v_in means flow from u to v
                        # u_in -> v_out means... shouldn't happen for original edges
                        if u % 2 == 1 and v % 2 == 0:
                            # u_out -> v_in: flow from u_orig to v_orig
                            flow_graph[u_orig].add(v_orig)
        
        # Now for each pair, find a path in flow_graph
        for i in range(k):
            for j in range(i+1, k):
                u, v = branches[i], branches[j]
                # BFS in flow_graph from u to v
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue:
                    cur = queue.popleft()
                    if cur == v:
                        found = True
                        break
                    for nb in flow_graph[cur]:
                        if nb not in parent:
                            parent[nb] = cur
                            queue.append(nb)
                if not found:
                    # Try reverse direction
                    parent = {v: None}
                    queue = deque([v])
                    found = False
                    while queue:
                        cur = queue.popleft()
                        if cur == u:
                            found = True
                            break
                        for nb in flow_graph[cur]:
                            if nb not in parent:
                                parent[nb] = cur
                                queue.append(nb)
                    if not found:
                        return False, None
                    # Path from v to u, reverse it
                    path = []
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                else:
                    path = []
                    cur = v
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                
                # Verify path
                if path[0] != u or path[-1] != v:
                    return False, None
                for idx in range(len(path)-1):
                    if path[idx+1] not in adj[path[idx]]:
                        return False, None
                # Check no branches in interior
                for vertex in path[1:-1]:
                    if vertex in branch_set:
                        return False, None
                
                paths_dict[(i, j)] = path
        
        # Verify all interiors are disjoint
        used_interiors = set()
        for (i, j), path in paths_dict.items():
            interior = set(path[1:-1])
            if interior & used_interiors:
                return False, None
            used_interiors |= interior
        
        return True, paths_dict

    def find_max_clique():
        """Bron-Kerbosch with pivoting to find maximum clique."""
        max_clique = []
        
        def bron_kerbosch(R, P_set, X):
            nonlocal max_clique
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            P_X = P_set | X
            if not P_X:
                return
            pivot = max(P_X, key=lambda v: len(adj[v] & P_set))
            candidates = [v for v in P_set if v not in adj[pivot]]
            for v in candidates:
                if time_up():
                    return
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
        
        bron_kerbosch(set(), set(range(n)), set())
        return max_clique

    def try_greedy_growth(initial_branches):
        """
        Start with initial_branches, greedily add vertices using max-flow feasibility check.
        Returns (branches, paths_dict) if successful, else None.
        """
        branches = list(initial_branches)
        if len(branches) < 2:
            return None
        
        feasible, paths = check_feasibility(branches)
        if not feasible:
            return None
        
        candidates = [v for v in range(n) if v not in set(branches)]
        
        while not time_up():
            added = False
            # Sort candidates by heuristic: prefer vertices adjacent to many existing branches
            candidates.sort(key=lambda v: -len(adj[v] & set(branches)))
            
            for cand in candidates:
                if time_up():
                    break
                new_branches = branches + [cand]
                new_feasible, new_paths = check_feasibility(new_branches)
                if new_feasible:
                    branches = new_branches
                    paths = new_paths
                    candidates = [v for v in range(n) if v not in set(branches)]
                    added = True
                    break
            
            if not added:
                break
        
        if len(branches) >= 2:
            return (branches, paths)
        return None

    best_result = None
    best_size = 0
    
    # Strategy 1: Start from max clique and grow
    clique = find_max_clique()
    if clique:
        result = try_greedy_growth(clique)
        if result and len(result[0]) > best_size:
            best_result = result
            best_size = len(result[0])
    
    # Strategy 2: Try multiple random seeds and grow
    random.seed(42)
    num_attempts = 30
    
    for attempt in range(num_attempts):
        if time_up():
            break
        
        # Random seed of size 2-4
        seed_size = random.randint(2, min(4, n-1))
        seed = random.sample(range(n), seed_size)
        
        result = try_greedy_growth(seed)
        if result and len(result[0]) > best_size:
            best_result = result
            best_size = len(result[0])
    
    # Strategy 3: Try systematic pairs as seeds (limited)
    if not time_up():
        pairs_tried = 0
        for i in range(n):
            if time_up() or pairs_tried >= 20:
                break
            for j in range(i+1, n):
                if time_up() or pairs_tried >= 20:
                    break
                result = try_greedy_growth([i, j])
                if result and len(result[0]) > best_size:
                    best_result = result
                    best_size = len(result[0])
                pairs_tried += 1
    
    # If no result found, fallback to max clique
    if best_result is None:
        if not clique:
            clique = [0, 1]
        branches = list(clique)
        paths_dict = {}
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u, v = branches[i], branches[j]
                paths_dict[(i, j)] = [u, v]
        best_result = (branches, paths_dict)
    
    branches, paths_dict = best_result
    
    # Convert paths_dict to list format
    paths_list = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            paths_list.append(paths_dict[(i, j)])
    
    # Ensure types are correct
    branches = [int(b) for b in branches]
    paths_list = [[int(v) for v in p] for p in paths_list]
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
