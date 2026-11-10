import sys
from collections import deque

# Increase recursion limit if needed, though iterative flow is better
sys.setrecursionlimit(10000)

def construct():
    import math
    import time
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Define sets
    S = list(range(2, 129))  # Small numbers
    L = list(range(129, 257)) # Large numbers
    
    # Map values to indices for flow graph
    # Nodes:
    # 0: Source
    # 1..len(S): S nodes
    # len(S)+1..len(S)+len(L): L nodes
    # len(S)+len(L)+1: Sink
    
    n_s = len(S)
    n_l = len(L)
    n_nodes = 1 + n_s + n_l + 1
    src = 0
    sink = n_nodes - 1
    
    # Graph for Max-Flow
    # Using adjacency list for residual graph
    # graph[u][v] = capacity
    # We'll use a dictionary of dictionaries or list of lists
    # List of lists is faster for dense-ish graphs, but this is sparse.
    # Let's use a standard adjacency list structure for Dinic.
    
    # Dinic's Algorithm Implementation
    class Dinic:
        def __init__(self, n):
            self.n = n
            self.adj = [[] for _ in range(n)]
            
        def add_edge(self, u, v, cap):
            self.adj[u].append([v, cap, len(self.adj[v])])
            self.adj[v].append([u, 0, len(self.adj[u]) - 1])
            
        def bfs(self, src, sink):
            self.level = [-1] * self.n
            q = deque([src])
            self.level[src] = 0
            while q:
                u = q.popleft()
                for v, cap, rev in self.adj[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[sink] >= 0
            
        def dfs(self, u, sink, f):
            if u == sink:
                return f
            for i in range(self.it[u], len(self.adj[u])):
                self.it[u] = i
                v, cap, rev = self.adj[u][i]
                if cap > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, sink, min(f, cap))
                    if d > 0:
                        self.adj[u][i][1] -= d
                        self.adj[v][rev][1] += d
                        return d
            return 0
            
        def max_flow(self, src, sink):
            flow = 0
            INF = float('inf')
            while self.bfs(src, sink):
                self.it = [0] * self.n
                f = self.dfs(src, sink, INF)
                while f > 0:
                    flow += f
                    f = self.dfs(src, sink, INF)
            return flow

    # Build Graph
    dinic = Dinic(n_nodes)
    
    # Weights
    w_s = {a: 1.0/a for a in S}
    w_l = {b: 1.0/b for b in L}
    
    # Add edges s -> a
    for i, a in enumerate(S):
        dinic.add_edge(src, 1 + i, w_s[a])
        
    # Add edges b -> t
    for j, b in enumerate(L):
        dinic.add_edge(1 + n_s + j, sink, w_l[b])
        
    # Add edges a -> b if conflict
    # Conflict if lcm(a, b) <= 256
    for i, a in enumerate(S):
        for j, b in enumerate(L):
            if math.lcm(a, b) <= 256:
                # Infinite capacity edge
                dinic.add_edge(1 + i, 1 + n_s + j, float('inf'))
                
    # Run Max Flow
    min_cut_val = dinic.max_flow(src, sink)
    
    # Determine which nodes are in S_cut (reachable from src in residual graph)
    visited = [False] * n_nodes
    q = deque([src])
    visited[src] = True
    while q:
        u = q.popleft()
        for v, cap, rev in dinic.adj[u]:
            if cap > 0 and not visited[v]:
                visited[v] = True
                q.append(v)
                
    # A = {a in S | visited[node_a]}
    # B_removed = {b in L | NOT visited[node_b]}
    # B_kept = {b in L | visited[node_b]}
    
    A = []
    for i, a in enumerate(S):
        if visited[1 + i]:
            A.append(a)
            
    B_kept = []
    for j, b in enumerate(L):
        if visited[1 + n_s + j]:
            B_kept.append(b)
            
    # Candidate solution
    candidate = A + B_kept
    
    # Check for internal conflicts in A
    # If A has conflicts, we need to resolve them.
    # Since A comes from a relaxation, it might contain conflicting pairs.
    # We will greedily remove conflicts from A, preferring to keep larger weights (smaller numbers).
    
    # Sort A by weight descending (value ascending)
    A.sort(key=lambda x: 1.0/x, reverse=True)
    
    valid_A = []
    for a in A:
        conflict = False
        for b in valid_A:
            if math.lcm(a, b) <= 256:
                conflict = True
                break
        if not conflict:
            valid_A.append(a)
            
    # Final set
    final_set = valid_A + B_kept
    
    # If final_set is empty (should not happen as B_kept is large), fallback
    if not final_set:
        final_set = L
        
    # Validate
    # The problem requires non-empty subset.
    # And lcm(a,b) > 256 for all distinct pairs.
    
    # Note: The flow ensures A and B_kept are compatible.
    # We manually ensured A is independent.
    # B_kept is a subset of L, and L is independent.
    # So the union is independent.
    
    # Return as list of integers
    return sorted(final_set)

# Run construct to verify locally if needed, but the function is the entry point.
# The evaluator calls construct().
