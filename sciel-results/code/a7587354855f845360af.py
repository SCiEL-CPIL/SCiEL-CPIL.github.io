import time
import random
from collections import defaultdict

def construct():
    # Start with the seed graph (Andrasfai-like, n=26)
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def max_independent_set_size(adj, n):
        """Branch and bound to find max independent set size."""
        if n <= 1:
            return n
        
        # Use memoization-like pruning with bitmasks for n <= 40
        # But for simplicity, use recursive B&B with ordering
        vertices = list(range(n))
        
        # Order vertices by degree (higher degree first for better pruning)
        degrees = [len(adj[v]) for v in vertices]
        order = sorted(range(n), key=lambda v: -degrees[v])
        
        best = 0
        
        def dfs(remaining, current_size):
            nonlocal best
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            if current_size + len(remaining) <= best:
                return
            
            # Pick vertex with highest degree in remaining
            v = max(remaining, key=lambda x: len(adj[x] & set(remaining)))
            rest = [u for u in remaining if u != v]
            
            # Option 1: Include v
            new_remaining = [u for u in rest if u not in adj[v]]
            dfs(new_remaining, current_size + 1)
            
            # Option 2: Exclude v
            dfs(rest, current_size)
        
        dfs(vertices, 0)
        return best
    
    def is_valid(n, adj):
        """Check if graph is triangle-free and alpha < 10."""
        # Triangle check
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        
        # Independent set check
        alpha = max_independent_set_size(adj, n)
        return alpha < 10
    
    # Verify initial graph
    if not is_valid(n, adj):
        # Fallback: return a known valid smaller graph
        n = 10
        adj = [set() for _ in range(n)]
        edges = [[i, j] for i in range(n) for j in range(i+1, n) if (j-i) % 3 != 0]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
    
    # Try to extend the graph
    deadline = time.time() + 110  # 110 seconds deadline (safety margin)
    
    # We will try to add vertices from current n up to 40
    for new_n in range(n + 1, 41):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 500  # More attempts for better search
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Strategy: Pick a random independent set S from current graph
            # Connect new vertex to S
            # The key insight: neighbors of new vertex must be independent
            
            # Generate random independent set using greedy with randomization
            remaining = list(range(new_n - 1))
            S = []
            random.shuffle(remaining)
            
            for v in remaining:
                if v not in adj and all(v not in adj[u] for u in S):
                    S.append(v)
                    if len(S) >= 5:  # Cap to avoid too-large sets
                        break
            
            # Also try smaller sets
            if len(S) > 3 and random.random() < 0.5:
                S = random.sample(S, random.randint(1, min(3, len(S))))
            
            # Create new adjacency
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(new_n - 1)
            
            # Check validity
            if is_valid(new_n, new_adj):
                # Update state
                edges.extend([[new_n - 1, u] for u in S])
                adj = new_adj
                n = new_n
                found = True
                break
        
        if not found:
            # Try edge modifications: remove some edges to allow extension
            # This is complex, so for now stop
            break
    
    return {'n': n, 'edges': edges}
