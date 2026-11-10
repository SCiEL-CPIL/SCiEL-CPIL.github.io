import random
import time
from itertools import combinations

def _find_bad_subset(adj, n):
    """Find a regular induced subgraph of size >= 6, or return None if none."""
    for size in range(6, n+1):
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
            for it in range(500):
                if time.time() > target_time or elapsed() > deadline - 5:
                    break
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
    
    # Try n=20 first (known achievable)
    # Allocate time based on remaining
    for n_try in [20, 21]:
        if elapsed() > deadline - 10:
            break
        remaining = deadline - 10 - elapsed()
        if n_try == 20:
            budget = min(100, remaining)
        else:
            budget = min(35, remaining)
        if budget < 5:
            break
        adj = try_find(n_try, budget)
        if adj is not None:
            edges = _to_edges(adj, n_try)
            if best is None or n_try > best[0]:
                best = (n_try, edges)
            # If found 20, try 21 only if time permits
            if n_try == 20:
                continue
            else:
                break
        else:
            # Failed on this n; for n=20 failure, try smaller
            if n_try == 20:
                # Try smaller n
                for n_small in [19, 18, 17, 16, 15]:
                    if elapsed() > deadline - 10:
                        break
                    remaining = deadline - 10 - elapsed()
                    if remaining < 5:
                        break
                    adj = try_find(n_small, min(20, remaining))
                    if adj is not None:
                        edges = _to_edges(adj, n_small)
                        if best is None or n_small > best[0]:
                            best = (n_small, edges)
                        break
            break
    
    if best is None:
        return {'n': 5, 'edges': []}
    
    return {'n': best[0], 'edges': best[1]}
