"""Packing edge-disjoint Hamiltonian cycles on the fixed 21-vertex host graph."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute base adjacency and edge set.
N = P['n']
BASE_ADJ = [set() for _ in range(N)]
EDGE_SET = set()
for a, b in P['edges']:
    BASE_ADJ[a].add(b)
    BASE_ADJ[b].add(a)
    EDGE_SET.add((a, b) if a < b else (b, a))


def _is_connected(adj, allowed_vertices, start_vertex):
    """
    Check if the subgraph induced by allowed_vertices is connected.
    adj: list of sets of neighbors in the current remaining graph.
    allowed_vertices: set of vertices that must be connected.
    start_vertex: a vertex in allowed_vertices to start BFS from.
    """
    if not allowed_vertices:
        return True
    
    visited = set()
    queue = deque([start_vertex])
    visited.add(start_vertex)
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v in allowed_vertices and v not in visited:
                visited.add(v)
                queue.append(v)
                
    return len(visited) == len(allowed_vertices)


def _find_hamiltonian(remaining_edges, deadline, rng, max_steps=200000):
    """
    Find one Hamiltonian cycle using only edges in remaining_edges.
    Returns list of vertices (length N, distinct) or None.
    Uses Warnsdorff heuristic, degree pruning, and connectivity pruning.
    """
    n = N
    # Build adjacency for the remaining graph
    adj = [set() for _ in range(n)]
    for (u, v) in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Quick degree feasibility: every vertex needs degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None

    best_cycle = None
    steps = 0
    start = 0
    
    # State
    path = [start]
    used = [False] * n
    used[0] = True
    
    # Precompute unvisited count for Warnsdorff updates? 
    # Simpler: just iterate adj[cur] and check used.
    
    def backtrack():
        nonlocal steps, best_cycle
        if best_cycle is not None:
            return True
        if time.time() >= deadline:
            return False
        
        cur = path[-1]
        
        if len(path) == n:
            # Check if closing edge exists
            if start in adj[cur]:
                best_cycle = list(path)
                return True
            return False
        
        # Generate candidates
        candidates = [x for x in adj[cur] if not used[x]]
        if not candidates:
            return False
            
        # Pruning: Check connectivity of unvisited vertices + start
        # The unvisited vertices must be connected to each other and to the path ends?
        # Actually, the unvisited vertices must form a connected subgraph that includes
        # the start vertex (since the cycle must close to start).
        # Wait, the path is cur -> ... -> unvisited -> ... -> start.
        # So the unvisited vertices must be connected to each other.
        # Also, each unvisited vertex must be able to reach the start (eventually).
        # A strong prune: The subgraph induced by (unvisited vertices U) plus {start} 
        # must be connected.
        unvisited = set(range(n)) - set(path)
        # Note: 'start' is in path, so we remove it from unvisited set for the check,
        # but we add it back for the connectivity check because the cycle must return to it.
        check_set = unvisited | {start}
        
        # Optimization: Only do connectivity check if it's cheap or if we are deep.
        # For N=21, BFS is very fast.
        if not _is_connected(adj, check_set, start):
            return False
            
        # Degree Pruning:
        # For each unvisited vertex v, the number of neighbors in (unvisited U {start, cur}) 
        # must be at least 2? 
        # Actually, v needs 2 edges in the final cycle.
        # One edge might come from a visited vertex (only cur is the current end, but previous visited 
        # vertices are dead ends unless we backtrack).
        # In the partial path, only 'cur' is an active endpoint. 'start' is the other.
        # So for any unvisited v, it must have at least 2 neighbors in (unvisited U {start, cur}).
        # Wait, if v is adjacent to a visited vertex w (w != cur, w != start), that edge is unusable 
        # because w is already in the path and not an endpoint.
        # So v must have >= 2 neighbors in (unvisited U {start, cur}).
        
        active_neighbors_pool = unvisited | {start, cur}
        for v in unvisited:
            deg = 0
            for nb in adj[v]:
                if nb in active_neighbors_pool:
                    deg += 1
            if deg < 2:
                return False

        # Sort candidates by Warnsdorff rule: fewest unvisited neighbors first
        # Tie-break randomly
        candidates.sort(key=lambda x: sum(1 for y in adj[x] if not used[y]))
        rng.shuffle(candidates)
        
        for x in candidates:
            path.append(x)
            used[x] = True
            steps += 1
            
            if steps > max_steps:
                path.pop()
                used[x] = False
                return False
                
            if backtrack():
                return True
                
            path.pop()
            used[x] = False
            
        return False

    backtrack()
    return best_cycle


def _is_valid_cycles(cycles, all_edges):
    seen = set()
    vset = None
    for cyc in cycles:
        if len(cyc) != N:
            return False
        if len(set(cyc)) != N:
            return False
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            return False
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            if e not in all_edges:
                return False
            if e in seen:
                return False
            seen.add(e)
    return True


def construct():
    deadline = time.time() + 50.0  # Safety margin
    rng = random.Random(12345)
    remaining = set(EDGE_SET)
    cycles = []
    
    # Fallback: one known cycle if valid
    seed_cycle = list(range(N))
    seed_ok = all((a, b) if a < b else (b, a) in EDGE_SET
                  for a, b in zip(seed_cycle, seed_cycle[1:] + seed_cycle[:1]))
    fallback = [seed_cycle] if seed_ok else []

    # Try to find up to 3 cycles
    for _ in range(3):
        if time.time() >= deadline:
            break
            
        found = None
        # Multiple restarts with different random seeds for tie-breaking
        # The connectivity and degree prunes make the search deterministic-ish 
        # but Warnsdorff tie-breaking is randomized.
        for attempt in range(50):
            if time.time() >= deadline:
                break
            found = _find_hamiltonian(remaining, deadline, rng)
            if found is not None:
                break
                
        if found is None:
            break
            
        # Remove edges of the found cycle from remaining
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
            
        cycles.append(list(cyc))

    # Validate
    if cycles and _is_valid_cycles(cycles, EDGE_SET):
        return [list(c) for c in cycles]
        
    if fallback:
        return [list(c) for c in fallback]
        
    # Last resort
    remaining = set(EDGE_SET)
    for attempt in range(100):
        if time.time() >= deadline:
            break
        f = _find_hamiltonian(remaining, deadline, rng)
        if f is not None and _is_valid_cycles([f], EDGE_SET):
            return [list(f)]
            
    return [list(range(N))]
# EVOLVE-BLOCK-END
