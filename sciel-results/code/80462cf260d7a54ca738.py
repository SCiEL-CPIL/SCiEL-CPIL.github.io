"""Decompose three fixed random graphs into few cycles/edges (Erdos-Gallai task).

Strategy: repeatedly extract long cycles via randomized DFS with backtracking,
and use a maximum matching based 2-factor extraction when possible.
Restart with fresh random edge orderings within a wall-clock deadline, keeping
the best (fewest-piece) valid result.
"""
import time
import random
import sys
from collections import deque

sys.setrecursionlimit(1000000)

SPEC = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
DEADLINE = 110.0  # wall-clock safety margin under the 120 s evaluator limit


def instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


# ---- Hopcroft-Karp on a bipartite graph given by adjacency lists ----
def hk_match(adj, n):
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n

    def bfs():
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return pair_u, matching


def extract_2factor(n, edges):
    """Build a 2-factor of the leftover graph via HK; return (cycles, used_edges)."""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    adjL = [list(s) for s in adj]
    pair_u, matching = hk_match(adjL, n)
    if matching < n:
        return [], set()
    # build functional digraph u -> pair_u[u]
    nxt = pair_u[:]
    seen = [0] * n
    cycles = []
    for s in range(n):
        if seen[s]:
            continue
        path = []
        cur = s
        while not seen[cur]:
            seen[cur] = 1
            path.append(cur)
            cur = nxt[cur]
        # cur is already seen; if it is in this path we closed a cycle
        if cur in path:
            idx = path.index(cur)
            cyc = path[idx:]
            if len(cyc) >= 3:
                cycles.append(cyc)
    used = set()
    for cyc in cycles:
        L = len(cyc)
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            used.add((a, b) if a < b else (b, a))
    return cycles, used


def find_long_cycle(n, edges, rng, max_depth=30):
    """Try to find a long cycle using randomized DFS with backtracking."""
    if not edges:
        return None
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    # Try multiple starting edges
    edge_list = list(edges)
    rng.shuffle(edge_list)
    
    for start_edge in edge_list[:5]:
        a, b = start_edge
        # Try both directions
        for start, other in [(a, b), (b, a)]:
            # DFS to find a path from start back to start (cycle)
            # Use iterative DFS with explicit stack
            best_cycle = None
            best_len = 2  # We need at least 3 vertices
            
            # State: (current, path, visited)
            stack = [(other, [start, other], {start, other})]
            visited_global = set()
            
            while stack:
                cur, path, visited = stack.pop()
                
                # Check if we can close the cycle
                if start in adj[cur] and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path + [start]
                
                # If path is too long, skip expanding
                if len(path) >= max_depth:
                    continue
                
                # Expand to neighbors
                nbrs = list(adj[cur])
                rng.shuffle(nbrs)
                for nbr in nbrs:
                    if nbr not in visited:
                        new_path = path + [nbr]
                        new_visited = visited | {nbr}
                        stack.append((nbr, new_path, new_visited))
            
            if best_cycle and len(best_cycle) >= 4:  # cycle of at least 3 edges
                # Convert to vertex list without closing edge
                return best_cycle[:-1]
    
    return None


def decompose_graph(n, edges, rng, deadline):
    """Return list of pieces (cycles or single edges) covering all edges."""
    pieces = []
    remaining = set(edges)
    
    # Phase 1: Try 2-factor extraction
    c, used = extract_2factor(n, remaining)
    if used:
        pieces.extend(c)
        remaining -= used
    
    # Phase 2: Repeatedly extract long cycles
    while remaining and time.time() < deadline:
        # Try to find a long cycle
        cycle = find_long_cycle(n, remaining, rng, max_depth=min(len(remaining), 30))
        if cycle is None:
            # Try with shorter max depth
            cycle = find_long_cycle(n, remaining, rng, max_depth=10)
        
        if cycle is None:
            break
        
        # Remove cycle edges from remaining
        L = len(cycle)
        used = set()
        for i in range(L):
            a = cycle[i]
            b = cycle[(i + 1) % L]
            key = (a, b) if a < b else (b, a)
            if key in remaining:
                used.add(key)
        
        if used:
            pieces.append(cycle)
            remaining -= used
        else:
            break
    
    # Phase 3: Leftover single edges
    for e in remaining:
        pieces.append(list(e))
    
    return pieces


def is_valid(n, edges, pieces):
    edge_set = set((min(a, b), max(a, b)) for a, b in edges)
    seen = set()
    for p in pieces:
        if len(p) == 2:
            u, v = p
            key = (min(u, v), max(u, v))
            if key in seen or key not in edge_set:
                return False
            seen.add(key)
        else:
            if len(p) < 3 or len(set(p)) != len(p):
                return False
            L = len(p)
            for i in range(L):
                u = p[i]
                v = p[(i + 1) % L]
                key = (min(u, v), max(u, v))
                if key in seen or key not in edge_set:
                    return False
                seen.add(key)
    return seen == edge_set


def construct():
    t0 = time.time()
    results = []
    for (n, p, seed) in SPEC:
        edges = instance(n, p, seed)
        best = None
        best_count = float('inf')
        
        # Restart with fresh rng seeds until deadline
        attempt = 0
        while time.time() < DEADLINE:
            arng = random.Random(1000 + attempt * 7919)
            pieces = decompose_graph(n, edges, arng, DEADLINE)
            if is_valid(n, edges, pieces):
                if len(pieces) < best_count:
                    best = pieces
                    best_count = len(pieces)
                # Early stop if we reached a strong lower bound
                if best_count <= n:
                    break
            attempt += 1
            if attempt >= 30:
                break
        
        if best is None:
            # safe fallback: every edge as its own piece
            best = [list(e) for e in edges]
        
        results.append(best)
    
    return results


if __name__ == "__main__":
    r = construct()
    for i, spec in enumerate(SPEC):
        print(spec, "pieces:", len(r[i]))
    print("total:", sum(len(x) for x in r))
