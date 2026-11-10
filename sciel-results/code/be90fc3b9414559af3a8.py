import random
import time
from itertools import combinations

def _find_bad_subset(adj, n, max_size=None):
    """Find a regular induced subgraph of size >= 6, or return None if none."""
    if max_size is None:
        max_size = n
    for size in range(6, max_size + 1):
        # Optimization: If size is large, the number of combinations is huge.
        # But we stop at first found.
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            first = -1
            regular = True
            for v in comb:
                d = (adj[v] & mask).bit_count()
                if first == -1:
                    first = d
                elif d != first:
                    regular = False
                    break
            if regular:
                return comb
    return None

def _flip_edge(adj, u, v):
    bit_v = 1 << v
    bit_u = 1 << u
    if adj[u] & bit_v:
        adj[u] &= ~bit_v
        adj[v] &= ~bit_u
    else:
        adj[u] |= bit_v
        adj[v] |= bit_u

def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    deadline = start + 145  # safety margin
    
    rng = random.Random(12345)
    best = None  # (n, edges)
    
    def elapsed():
        return time.time() - start
    
    def try_find(n, time_budget):
        """Try to find a valid graph on n vertices within time_budget seconds."""
        target_time = time.time() + time_budget
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > target_time or elapsed() > deadline - 5:
                break
            # Random init
            adj = [0]*n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # We limit the size of the bad subset we look for to speed up checks
            # If we find a bad subset of size 6, we fix it.
            # If we don't find any up to n, it's valid.
            for it in range(200): # Reduced iterations
                if time.time() > target_time or elapsed() > deadline - 5:
                    break
                # Check for bad subset
                # To speed up, we can check smaller sizes first.
                bad = _find_bad_subset(adj, n)
                if bad is None:
                    return adj
                # Flip a random edge within bad subset
                a = rng.choice(bad)
                b = rng.choice(bad)
                if a == b:
                    # pick another
                    b = rng.choice([x for x in bad if x != a])
                _flip_edge(adj, a, b)
        return None
    
    # Try n=20 first
    # Allocate time
    remaining = deadline - 10 - elapsed()
    if remaining > 0:
        budget = min(100, remaining)
        if budget > 10:
            adj = try_find(20, budget)
            if adj is not None:
                # Verify
                if _find_bad_subset(adj, 20) is None:
                    edges = _to_edges(adj, 20)
                    best = (20, edges)
    
    if best is None:
        # Try n=19
        remaining = deadline - 10 - elapsed()
        if remaining > 5:
            budget = min(30, remaining)
            if budget > 5:
                adj = try_find(19, budget)
                if adj is not None:
                    if _find_bad_subset(adj, 19) is None:
                        edges = _to_edges(adj, 19)
                        best = (19, edges)
    
    if best is None:
        # Try n=18
        remaining = deadline - 10 - elapsed()
        if remaining > 5:
            budget = min(20, remaining)
            if budget > 5:
                adj = try_find(18, budget)
                if adj is not None:
                    if _find_bad_subset(adj, 18) is None:
                        edges = _to_edges(adj, 18)
                        best = (18, edges)
                        
    if best is None:
        # Fallback to a known small valid graph or random search for smaller n
        # A 5-vertex graph is always valid (no 6-vertex subset)
        # But we want to maximize n.
        # Let's try n=17, 16... quickly
        for n_try in [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
            if elapsed() > deadline - 5:
                break
            remaining = deadline - 5 - elapsed()
            if remaining < 2:
                break
            # For small n, search is very fast
            adj = try_find(n_try, min(5, remaining))
            if adj is not None:
                if _find_bad_subset(adj, n_try) is None:
                    edges = _to_edges(adj, n_try)
                    best = (n_try, edges)
                    break
    
    if best is None:
        # Absolute fallback: n=5 empty graph
        return {'n': 5, 'edges': []}
    
    return {'n': best[0], 'edges': best[1]}
