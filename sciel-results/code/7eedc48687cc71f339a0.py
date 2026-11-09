import time
import sys
from random import Random
from collections import deque

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        
    def add_edge(self, u, v, cap):
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])
        
    def bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, rev in self.graph[u]:
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
        INF = 10**9
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self.dfs(s, t, INF)
                if f == 0:
                    break
                flow += f
        return flow

def _find_max_2_matching(adj, n):
    """
    Finds a maximum 2-matching in the graph defined by adj.
    Returns a set of edges (u, v) in the matching.
    """
    # Node mapping:
    # v_in = 2*v
    # v_out = 2*v + 1
    # S = 2*n
    # T = 2*n + 1
    N = 2 * n + 2
    S = 2 * n
    T = 2 * n + 1
    
    dinic = Dinic(N)
    
    # Add vertex split edges
    for v in range(n):
        dinic.add_edge(2*v, 2*v+1, 2)
        
    # Add source/sink connections
    for v in range(n):
        dinic.add_edge(S, 2*v, 2) # S to v_in
        dinic.add_edge(2*v+1, T, 2) # v_out to T
        
    # Add graph edges
    # We need to iterate over edges in adj
    # adj is dict of sets
    seen = set()
    for u in range(n):
        for v in adj[u]:
            if (u, v) in seen or (v, u) in seen:
                continue
            seen.add((u, v))
            seen.add((v, u))
            # Edge u->v
            # u_out -> v_in
            dinic.add_edge(2*u+1, 2*v, 1)
            # v_out -> u_in
            dinic.add_edge(2*v+1, 2*u, 1)
            
    dinic.max_flow(S, T)
    
    # Extract edges
    matching_edges = set()
    for u in range(n):
        for v, cap, rev in dinic.graph[2*u+1]:
            if v < 2*n: # v is a v_in node
                # Check if flow was sent
                # The reverse edge is in dinic.graph[v][rev]
                # If cap < 1 (original cap was 1), then flow was used?
                # Original cap was 1. If current cap is 0, flow 1 was used.
                if cap == 0:
                    v_node = v
                    v_idx = v_node // 2
                    matching_edges.add(tuple(sorted((u, v_idx))))
                    
    return matching_edges

def _extract_cycles(adj, matching_edges, n):
    """
    Extracts cycles from the 2-matching edges.
    Returns a list of cycles (list of vertices).
    """
    if not matching_edges:
        return []
        
    # Build subgraph
    sub_adj = {i: set() for i in range(n)}
    for u, v in matching_edges:
        sub_adj[u].add(v)
        sub_adj[v].add(u)
        
    visited = set()
    cycles = []
    
    for i in range(n):
        if i in visited or len(sub_adj[i]) != 2:
            continue
            
        # Start DFS
        path = [i]
        visited.add(i)
        curr = i
        prev = -1
        found_cycle = False
        cycle_nodes = []
        
        while True:
            neighbors = list(sub_adj[curr])
            # Filter out prev
            nexts = [x for x in neighbors if x != prev]
            
            if not nexts:
                break # Dead end, shouldn't happen in 2-matching components unless path
                
            nxt = nexts[0]
            
            if nxt == i:
                # Closed cycle
                found_cycle = True
                cycle_nodes = path[:]
                break
            else:
                if nxt in visited:
                    # Found a cycle, but not necessarily starting at i?
                    # In a 2-matching, components are paths or cycles.
                    # If we hit a visited node that is not i, it's a cycle within a larger structure?
                    # No, degree is 2. So it must be the start of the cycle.
                    # If we started at i, and hit i, it's a cycle.
                    # If we hit some other visited node, it implies we entered a cycle already visited?
                    # But we mark visited globally.
                    # So if we hit a visited node, we are in a cycle that was already processed?
                    # Or we are closing a cycle.
                    # Since degree is 2, the only way to hit a visited node is if it's the start of the current component.
                    # So nxt must be i.
                    pass
                    
                path.append(nxt)
                visited.add(nxt)
                prev = curr
                curr = nxt
                
        if found_cycle and len(cycle_nodes) >= 3:
            cycles.append(cycle_nodes)
            
    return cycles

def _decompose(edges, n, rng=None):
    if rng is None:
        rng = Random(42)
        
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    
    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    # Iterative peeling
    while remaining:
        # Check time
        if time.time() > 140:
            break
            
        matching_edges = _find_max_2_matching(adj, n)
        cycles = _extract_cycles(adj, matching_edges, n)
        
        if not cycles:
            # No more cycles can be found via 2-matching
            break
            
        # Pick the longest cycle? Or all?
        # We can add all disjoint cycles found.
        # But we must ensure they are edge-disjoint from each other.
        # In a 2-matching, cycles are vertex-disjoint, hence edge-disjoint.
        # So we can add all of them.
        
        for cycle in cycles:
            pieces.append(cycle)
            
            # Remove edges from remaining and adj
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    remaining.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
    # Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
        
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                s = specs[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        rng = Random(42 + spec[2])
        pieces = _decompose(edges, n, rng)
        results.append(pieces)
        
    return results
