import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    def find_violation(adj, n, k):
        """Find the first induced subgraph of size k that is regular. Returns subset or None."""
        if k > n:
            return None
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            # Calculate degrees, short-circuit if not regular
            first_deg = (adj[subset[0]] & mask).bit_count()
            is_reg = True
            for idx in range(1, k):
                v = subset[idx]
                d = (adj[v] & mask).bit_count()
                if d != first_deg:
                    is_reg = False
                    break
            if is_reg:
                return subset
        return None

    def is_valid(adj, n):
        """Check if graph has no regular induced subgraph of size >= 6."""
        for k in range(6, n + 1):
            if find_violation(adj, n, k) is not None:
                return False
        return True

    def local_search(adj, n, time_limit):
        """Local search to eliminate regular induced subgraphs."""
        deadline_local = time.time() + time_limit
        while time.time() < deadline_local:
            # Check for violations starting from smallest k
            violation = None
            for k in range(6, n + 1):
                violation = find_violation(adj, n, k)
                if violation is not None:
                    break
            
            if violation is None:
                return True
            
            # Pick random pair in violation and flip edge
            u = random.choice(violation)
            v = random.choice([x for x in violation if x != u])
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return False

    # Try n=21 down to 16
    for n in range(21, 15, -1):
        if time.time() > deadline:
            break
        
        constructions = []
        
        # Generate a few random candidates
        for seed in range(10):
            if time.time() > deadline:
                break
            
            random.seed(seed + n)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj.copy())
            
        # Also try sparse and dense
        for p in [0.3, 0.7]:
            if time.time() > deadline:
                break
            random.seed(999)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj.copy())

        # Try each construction
        for adj in constructions:
            if time.time() > deadline:
                break
            
            # Quick check: if already valid, return
            if is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Try local search
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                if local_search(adj, n, remaining_time - 2):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
