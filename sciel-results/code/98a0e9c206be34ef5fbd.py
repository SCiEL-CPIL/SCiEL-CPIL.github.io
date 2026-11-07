"""Packing edge-disjoint Hamiltonian cycles on the fixed 21-vertex host graph."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
BASE_ADJ = [set() for _ in range(N)]
EDGE_SET = set()
for a, b in P['edges']:
    BASE_ADJ[a].add(b)
    BASE_ADJ[b].add(a)
    EDGE_SET.add((a, b) if a < b else (b, a))


def _connected(adj):
    """Check if the graph defined by adj is connected."""
    n = len(adj)
    if n == 0:
        return True
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    return count == n


def _find_hamiltonian(remaining_edges, deadline, rng, max_steps=5000000):
    """Find one Hamiltonian cycle using only edges in remaining_edges.
    Returns list of vertices (length N, distinct) or None.
    Uses Warnsdorff-guided backtracking with forced-move detection."""
    n = N
    adj = [set() for _ in range(n)]
    for (u, v) in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Quick degree feasibility: every vertex needs degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Connectivity check
    if not _connected(adj):
        return None
    
    start = 0
    path = [start]
    used = [False] * n
    used[0] = True
    
    def _warnsdorff_key(x):
        """Number of unvisited neighbors of x (forward options)."""
        cnt = 0
        for y in adj[x]:
            if not used[y]:
                cnt += 1
        return cnt
    
    def _has_forced_move(cur, path_len):
        """Check if there's an unvisited vertex with exactly one unvisited neighbor.
        If so, return that vertex if it's adjacent to cur, else None."""
        if path_len >= n:
            return None
        for v in range(n):
            if not used[v]:
                unvisited_neighbors = [u for u in adj[v] if not used[u]]
                if len(unvisited_neighbors) == 1:
                    # This vertex must be visited next if it's reachable from cur
                    if v in adj[cur]:
                        return v
                    # Otherwise we're in trouble - but let's not prune yet
        return None
    
    def backtrack():
        nonlocal path, used
        if time.time() >= deadline:
            return False
        if len(path) == n:
            cur = path[-1]
            if start in adj[cur]:
                return True
            return False
        
        cur = path[-1]
        
        # Check for forced moves
        forced = _has_forced_move(cur, len(path))
        if forced is not None:
            path.append(forced)
            used[forced] = True
            if backtrack():
                return True
            path.pop()
            used[forced] = False
            return False
        
        cand = [x for x in adj[cur] if not used[x]]
        if not cand:
            return False
        
        # Sort by Warnsdorff: fewest forward options first
        cand.sort(key=lambda x: _warnsdorff_key(x))
        # Randomize ties
        for i in range(len(cand) - 1, 0, -1):
            if _warnsdorff_key(cand[i]) == _warnsdorff_key(cand[i-1]):
                rng.swap(i, i-1) if hasattr(rng, 'swap') else None
                # Simple shuffle for ties
                if rng.random() < 0.5:
                    cand[i], cand[i-1] = cand[i-1], cand[i]
        
        for x in cand:
            path.append(x)
            used[x] = True
            if backtrack():
                return True
            path.pop()
            used[x] = False
        return False
    
    # Try multiple starting vertices and random restarts
    for attempt in range(100):
        if time.time() >= deadline:
            return None
        
        # Reset state
        path = [start]
        used = [False] * n
        used[0] = True
        
        if backtrack():
            return list(path)
    
    return None


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
    deadline = time.time() + 80.0  # safety margin under 90s limit
    rng = random.Random(42)
    remaining = set(EDGE_SET)
    cycles = []
    
    # Seed fallback: one known cycle (full range) if it is valid in the graph
    seed_cycle = list(range(N))
    seed_ok = all((a, b) if a < b else (b, a) in EDGE_SET
                  for a, b in zip(seed_cycle, seed_cycle[1:] + seed_cycle[:1]))
    fallback = [seed_cycle] if seed_ok else []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for cycle_idx in range(3):
        if time.time() >= deadline:
            break
        
        found = None
        # Multiple attempts with fresh randomness
        for attempt in range(200):
            if time.time() >= deadline:
                break
            found = _find_hamiltonian(remaining, deadline, rng)
            if found is not None:
                break
        
        if found is None:
            break
        
        # Remove its edges from remaining
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
        cycles.append(list(cyc))
    
    # Validate; if anything is off, fall back.
    if cycles and _is_valid_cycles(cycles, EDGE_SET):
        return [list(c) for c in cycles]
    if fallback:
        return [list(c) for c in fallback]
    
    # Last resort: any valid single cycle via one more attempt
    remaining = set(EDGE_SET)
    for attempt in range(500):
        if time.time() >= deadline:
            break
        f = _find_hamiltonian(remaining, deadline, rng)
        if f is not None and _is_valid_cycles([f], EDGE_SET):
            return [list(f)]
    
    return [list(range(N))]
# EVOLVE-BLOCK-END
