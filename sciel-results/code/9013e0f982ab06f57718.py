"""Construct a 4-regular graph of girth 10 using random 4-regular graph + local search."""
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    random.seed(42)
    
    # For k=4, g=10: target n around 320 (known best)
    # Moore bound for (4,10): 1 + 4*(1+3+3^2+3^3) = 1 + 4*91 = 365... wait
    # Actually Moore bound: for even girth g, n >= 1 + k*(1 + (k-1) + (k-1)^2 + ... + (k-1)^{g/2 - 1})
    # For k=4, g=10: n >= 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161
    # Wait, let me recalculate. For girth g (even), the Moore bound is:
    # n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{g/2-1})
    # For k=4, g=10: n >= 2 * (1 + 3 + 9 + 27) = 2*40 = 80
    # Hmm, that seems low. Let me think again.
    # Actually the Moore bound for (k,g) with g even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # For k=4, g=10: g/2-1 = 4, so sum_{i=0}^{4} 3^i = 1+3+9+27+81 = 121, n >= 2*121 = 242
    # Wait, that's 242. The best lower bound mentioned is 243.
    
    # Let's use n=320 as starting point (known best) and try to find a valid graph
    # Strategy: generate random 4-regular graph, then use local search to eliminate short cycles
    
    n = 320
    
    def generate_random_4reg(n):
        """Generate a random 4-regular simple graph on n vertices."""
        # Use configuration model
        for attempt in range(100):
            half_edges = list(range(n))
            for _ in range(4):
                half_edges.extend(range(n))
            random.shuffle(half_edges)
            adj = {}
            for i in range(n):
                adj[i] = set()
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                # Verify all degrees are exactly 4
                if all(len(adj[i]) == 4 for i in range(n)):
                    return adj
        raise Exception("Failed to generate random 4-regular graph")
    
    def find_short_cycles(adj, max_len):
        """Find all non-backtracking closed walks of length < max_len."""
        # Use BFS from each vertex
        cycles = set()
        for start in range(n):
            # State: (current_vertex, prev_vertex, length)
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                cur, prev, length = queue.popleft()
                if length >= max_len:
                    continue
                if cur == start and length >= 3:
                    cycles.add(frozenset([start, prev, cur]))
                    continue
                for nb in adj[cur]:
                    if nb == prev:
                        continue  # no backtracking
                    if (nb, cur) not in visited:
                        visited.add((nb, cur))
                        queue.append((nb, cur, length + 1))
        return cycles
    
    def count_short_walks(adj, max_len):
        """Count non-backtracking closed walks of length < max_len."""
        count = 0
        for start in range(n):
            # State: (current_vertex, prev_vertex, length)
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                cur, prev, length = queue.popleft()
                if length >= max_len:
                    continue
                if cur == start and length >= 3:
                    count += 1
                    continue
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if (nb, cur) not in visited:
                        visited.add((nb, cur))
                        queue.append((nb, cur, length + 1))
        return count
    
    def try_edge_swap(adj, u1, v1, u2, v2):
        """Try to swap edges (u1,v1) and (u2,v2) with (u1,u2) and (v1,v2)."""
        if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
            return False
        # Check that new edges don't already exist
        if u2 in adj[u1] or v2 in adj[v1]:
            return False
        # Check that old edges exist
        if v1 not in adj[u1] or v2 not in adj[u2]:
            return False
        # Perform swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        adj[u1].add(u2)
        adj[u2].add(u1)
        adj[v1].add(v2)
        adj[v2].add(v1)
        return True
    
    def undo_edge_swap(adj, u1, v1, u2, v2):
        """Undo an edge swap."""
        adj[u1].remove(u2)
        adj[u2].remove(u1)
        adj[v1].remove(v2)
        adj[v2].remove(v1)
        adj[u1].add(v1)
        adj[v1].add(u1)
        adj[u2].add(v2)
        adj[v2].add(u2)
    
    # Generate initial graph
    adj = generate_random_4reg(n)
    
    # Local search: minimize count of short cycles
    best_adj = [set(adj[i]) for i in range(n)]
    best_score = count_short_walks(adj, g)
    
    for iteration in range(10000):
        if time.time() > deadline - 5.0:
            break
        
        # Try random edge swaps
        improved = False
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            # Pick two random edges
            u1 = random.randint(0, n-1)
            v1 = random.choice(list(adj[u1]))
            u2 = random.randint(0, n-1)
            v2 = random.choice(list(adj[u2]))
            
            if try_edge_swap(adj, u1, v1, u2, v2):
                new_score = count_short_walks(adj, g)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [set(adj[i]) for i in range(n)]
                    improved = True
                else:
                    undo_edge_swap(adj, u1, v1, u2, v2)
        
        if not improved and best_score == 0:
            break
    
    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append((i, j))
    
    # Verify
    assert len(edges) == n * k // 2
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    assert all(d == k for d in deg)
    
    return n, edges
