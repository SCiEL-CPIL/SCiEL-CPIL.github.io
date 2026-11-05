import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit
    
    def moore_bound(k, g):
        if k <= 1: return 0
        if g == 1: return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)
    
    def validate_graph(n, edges, k, g):
        """Validate that the graph is simple, k-regular, connected, and has girth >= g."""
        if n < 1:
            return False
        adj = [set() for _ in range(n)]
        seen_edges = set()
        for i, j in edges:
            if i < 0 or i >= n or j < 0 or j >= n or i == j:
                return False
            a, b = min(i, j), max(i, j)
            if (a, b) in seen_edges:
                return False
            seen_edges.add((a, b))
            adj[i].add(j)
            adj[j].add(i)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check connectivity
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if count != n:
            return False
        
        # Check girth using BFS from each node (slower but robust for small N)
        # For N ~ 500, BFS from each node is O(N * (N+E)) ~ 500 * 500 * 5 = 1.25M ops, which is fine.
        for u in range(n):
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= g - 1:
                    continue
                for nxt in adj[curr]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[curr] + 1
                        q.append(nxt)
                    elif nxt != u and dist[nxt] + dist[curr] + 1 < g:
                        # Found a cycle shorter than g
                        # Note: This simple check might miss some cycles if not careful, 
                        # but for "no cycle shorter than g", if we find a path back to u 
                        # with length < g, it's a cycle.
                        # Actually, standard girth check: 
                        # If we encounter a visited node 'nxt' that is not the parent, 
                        # and dist[curr] + 1 + dist[nxt] (from u) ... no, this is tricky.
                        # Better: Non-backtracking walk check or simple BFS cycle detection.
                        pass
                        
        # Let's use the non-backtracking check from the original code, it was correct.
        def check_girth_nonbacktracking(adj, n, g):
            for u in range(n):
                visited_states = set()
                queue = deque()
                for v in adj[u]:
                    state = (v, u, 1)
                    visited_states.add((v, u))
                    queue.append(state)
                while queue:
                    curr, prev, length = queue.popleft()
                    if length >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == u and length + 1 < g:
                            return False
                        new_state = (nxt, curr)
                        if new_state not in visited_states:
                            visited_states.add(new_state)
                            queue.append((nxt, curr, length + 1))
            return True

        if not check_girth_nonbacktracking(adj, n, g):
            return False
            
        return True

    def random_regular_graph(n, k):
        """Generate a random k-regular graph on n vertices."""
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Self loop, retry
            if v in adj[u]:
                return None # Multi-edge, retry
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_shortest_cycle_length(adj, n, g):
        """Return length of shortest cycle, or inf if none < g."""
        min_len = float('inf')
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= g:
                    continue
                for nxt in adj[curr]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[curr] + 1
                        parent[nxt] = curr
                        q.append(nxt)
                    elif nxt != parent[curr] and parent[nxt] != curr:
                        # Found a cycle
                        # Length is dist[curr] + dist[nxt] + 1
                        # But we need to ensure it's a simple cycle.
                        # This BFS approach finds the shortest cycle containing u?
                        # Not exactly. It finds the shortest cycle in the graph if we do it for all u?
                        # Actually, the shortest cycle in a graph can be found by BFS from each node.
                        cycle_len = dist[curr] + dist[nxt] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
        return min_len

    # Main Construction Loop
    best_n = None
    best_edges = None
    
    # Try increasing N
    # Moore bound for k=5, g=9 is 426.
    # Known smallest is 1116.
    # We try to find a valid graph.
    # Start from a reasonable N.
    
    for N in range(450, 1200, 50):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple random restarts for this N
        attempts = 10
        for _ in range(attempts):
            if time.time() - start_time > timeout:
                break
                
            adj = random_regular_graph(N, k)
            if adj is None:
                continue
                
            # Check girth
            # Use the non-backtracking check
            def check_girth_nb(adj, n, g):
                for u in range(n):
                    visited_states = set()
                    queue = deque()
                    for v in adj[u]:
                        state = (v, u, 1)
                        visited_states.add((v, u))
                        queue.append(state)
                    while queue:
                        curr, prev, length = queue.popleft()
                        if length >= g - 1:
                            continue
                        for nxt in adj[curr]:
                            if nxt == prev:
                                continue
                            if nxt == u and length + 1 < g:
                                return False
                            new_state = (nxt, curr)
                            if new_state not in visited_states:
                                visited_states.add(new_state)
                                queue.append((nxt, curr, length + 1))
                return True

            if check_girth_nb(adj, N, g):
                # Found a valid graph!
                edges = []
                for i in range(N):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return N, edges

    # If we didn't find one, raise error
    raise ValueError("Could not construct a valid graph")

# EVOLVE-BLOCK-END
