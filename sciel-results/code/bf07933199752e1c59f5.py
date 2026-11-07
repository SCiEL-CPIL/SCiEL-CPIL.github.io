"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(n, m) is the
    least positive L such that (m, m+L] contains distinct b_1,...,b_n with
    i dividing b_i.
    
    We use a max-flow (Dinic's algorithm) approach to check feasibility for
    a given L, and exploit the periodicity of the divisibility conditions
    to limit the search space to one period of LCM(1..12) = 27720.
    """
    n = 12
    max_m = 1000000
    max_L = 144  # Upper bound from task description
    period = 27720  # LCM(1..12)
    
    # Dinic's algorithm implementation
    class Dinic:
        def __init__(self, num_nodes):
            self.n = num_nodes
            self.graph = [[] for _ in range(num_nodes)]
            # Each edge is [to, cap, rev_index]
            
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
        
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.graph[u]:
                    v, cap, rev = e
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0
        
        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(self.it[u], len(self.graph[u])):
                self.it[u] = i
                v, cap, rev = self.graph[u][i]
                if cap > 0 and self.level[u] < self.level[v]:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        self.graph[u][i][1] -= d
                        self.graph[v][rev][1] += d
                        return d
            return 0
        
        def max_flow(self, s, t):
            flow = 0
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, 10**9)
                    if f == 0:
                        break
                    flow += f
            return flow
    
    def feasible(m, L):
        """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
        # Build bipartite matching: left side = indices 1..n, right side = values in (m, m+L]
        # Source -> index_i -> value_v -> Sink
        # We need to find if there's a perfect matching.
        
        # Collect all values in (m, m+L]
        values = list(range(m + 1, m + L + 1))
        
        # Left nodes: 0..n-1 (indices 1..n)
        # Right nodes: n..n+len(values)-1
        # Source: n + len(values)
        # Sink: n + len(values) + 1
        num_nodes = n + len(values) + 2
        source = n + len(values)
        sink = n + len(values) + 1
        
        dinic = Dinic(num_nodes)
        
        # Source to each index
        for i in range(n):
            dinic.add_edge(source, i, 1)
        
        # Each index to values it divides
        # Optimization: Pre-compute divisibility or just check inline
        for i in range(1, n + 1):
            for v in values:
                if v % i == 0:
                    # Map value v to node index
                    # v is in [m+1, m+L], so index in values is v - (m+1)
                    v_idx = v - (m + 1)
                    dinic.add_edge(i - 1, n + v_idx, 1)
        
        # Each value to sink
        for j in range(len(values)):
            dinic.add_edge(n + j, sink, 1)
        
        return dinic.max_flow(source, sink) >= n
    
    best_m = 0
    best_L = 0
    
    # Search m in one period [0, period-1]
    # f(12, m) is periodic with period 27720
    for m in range(period):
        # Binary search for the smallest L such that feasible(m, L)
        lo, hi = 1, max_L
        result_L = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(m, mid):
                result_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if result_L > best_L:
            best_L = result_L
            best_m = m
            
    # The optimal m found is in [0, 27719]. This is a valid m in [0, 1000000].
    return int(best_m)

# EVOLVE-BLOCK-END
