"""C6-free graph on 40 vertices: fast C6 check via BFS + greedy + local search."""
import time
import random
from collections import deque

N = 40
DEADLINE = 118.0  # safety margin before 150s kill


def _creates_c6(adj, u, v):
    """Check if adding edge (u,v) creates a C6.
    A C6 containing (u,v) is a path of length 5 from u to v in the current graph.
    We do BFS from u up to depth 5, avoiding v, and check if we can reach any neighbor of v.
    """
    if v in adj[u]:
        return True  # already adjacent, shouldn't happen for non-edge
    
    # BFS from u, tracking (node, depth, path_set for cycle detection)
    # We need a simple path of length exactly 5 from u to some neighbor of v.
    # Equivalently: is there a simple path u -> ... -> w where w is adjacent to v,
    # and the path has 5 edges (so 6 vertices: u, a, b, c, d, w where w~v).
    
    # BFS with depth limit 5, avoiding v
    # State: (node, depth, frozenset_of_visited)
    # This is expensive. Better: use the fact that we just need ANY simple path of length 5.
    
    # Alternative: enumerate all paths of length 5 from u (avoiding v) and check if endpoint is in N(v).
    # But that's the same as the original. Let's use a smarter approach.
    
    # Key insight: A C6 through (u,v) exists iff there's a path of length 5 from u to v.
    # We can do BFS from u with depth limit 5, but we need to ensure the path is simple
    # (no repeated vertices) AND doesn't pass through v.
    
    # Since N=40 is small, let's use a more efficient method:
    # For each pair (a, d) where a ~ u and d ~ v, check if there's a path of length 3 from a to d
    # avoiding u, v, a, d.
    # Path: u - a - b - c - d - v
    # So we need: a in N(u), d in N(v), and a path a-b-c-d of length 3 with b,c distinct from u,v,a,d.
    
    nu = adj[u]
    nv = adj[v]
    
    for a in nu:
        if a == v:
            continue
        for d in nv:
            if d == u or d == a:
                continue
            # Check if there's a path a -> b -> c -> d with b,c not in {u,v,a,d}
            # b in N(a) \ {u, v, d}, c in N(b) \ {a, u, v, d}, and c in N(d)
            for b in adj[a]:
                if b == u or b == v or b == d:
                    continue
                for c in adj[b]:
                    if c == a or c == u or c == v or c == d:
                        continue
                    if c in adj[d]:
                        return True
    return False


def _edge_list(adj):
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges


def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_adj = [set() for _ in range(N)]
    best_count = 0
    
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Phase 1: Greedy build with multiple random restarts
    for trial in range(100):
        if time.time() - start > DEADLINE - 40:
            break
        adj = [set() for _ in range(N)]
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        for (u, v) in pairs:
            if time.time() - start > DEADLINE - 40:
                break
            if not _creates_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        cnt = sum(len(s) for s in adj) // 2
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
    
    # Phase 2: Local improvement
    adj = [set(s) for s in best_adj]
    improved = True
    while improved and time.time() - start < DEADLINE - 5:
        improved = False
        # Try adding any missing edge
        for u in range(N):
            if time.time() - start > DEADLINE - 5:
                break
            for v in range(u + 1, N):
                if v in adj[u]:
                    continue
                if not _creates_c6(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
        
        # Try swaps: remove one edge, add two
        if time.time() - start < DEADLINE - 5:
            for _ in range(5000):
                if time.time() - start > DEADLINE - 5:
                    break
                u = rng.randrange(N)
                if not adj[u]:
                    continue
                v = rng.choice(list(adj[u]))
                if u > v:
                    u, v = v, u
                adj[u].discard(v)
                adj[v].discard(u)
                added = 0
                for _ in range(3):
                    c = rng.randrange(N)
                    d = rng.randrange(N)
                    if c == d or d in adj[c]:
                        continue
                    if not _creates_c6(adj, c, d):
                        adj[c].add(d)
                        adj[d].add(c)
                        added += 1
                if added >= 1:
                    improved = True
                else:
                    adj[u].add(v)
                    adj[v].add(u)
    
    cnt = sum(len(s) for s in adj) // 2
    if cnt > best_count:
        best_count = cnt
        best_adj = adj
    
    edges = _edge_list(best_adj)
    return edges


if __name__ == "__main__":
    e = construct()
    print(len(e))
