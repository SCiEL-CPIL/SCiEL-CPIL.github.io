"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
FULL_EDGES = set()
for u, v in P['edges']:
    if u > v:
        u, v = v, u
    FULL_EDGES.add((u, v))

ADJ = defaultdict(set)
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

class _TimeExceeded(Exception):
    pass

def _find_hamiltonian_cycle(adj_cur, start, deadline, rng):
    """Find a Hamiltonian cycle in the remaining graph using randomized DFS with Warnsdorff heuristic."""
    n2 = N
    visited = [False] * n2
    path = [start]
    visited[start] = True
    step_count = [0]
    
    def dfs():
        step_count[0] += 1
        if step_count[0] % 5000 == 0 and time.time() > deadline:
            raise _TimeExceeded()
        
        if len(path) == n2:
            last = path[-1]
            return start in adj_cur[last]
        
        cur = path[-1]
        nbrs = list(adj_cur[cur])
        # Warnsdorff: prefer neighbors with fewer unvisited neighbors
        def score(w):
            if visited[w]:
                return 1000
            return sum(1 for x in adj_cur[w] if not visited[x])
        
        nbrs.sort(key=lambda w: (score(w), rng.random()))
        
        for w in nbrs:
            if not visited[w]:
                visited[w] = True
                path.append(w)
                if dfs():
                    return True
                path.pop()
                visited[w] = False
        return False
    
    try:
        if dfs():
            return list(path)
    except _TimeExceeded:
        return None
    return None

def _verify_cycles(cycles):
    """Verify that cycles are valid: simple, on same vertex set, edge-disjoint, edges in host."""
    if not cycles:
        return False
    
    # All cycles must use exactly the same vertex set of size at least 3
    vertex_set = set(cycles[0])
    if len(vertex_set) < 3:
        return False
    
    used_edges = set()
    for cyc in cycles:
        if len(cyc) < 3:
            return False
        if len(set(cyc)) != len(cyc):
            return False
        if set(cyc) != vertex_set:
            return False
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i + 1) % len(cyc)]
            k = _edge_key(a, b)
            if k not in FULL_EDGES:
                return False
            if k in used_edges:
                return False
            used_edges.add(k)
    return True

def construct():
    """Find maximum number of edge-disjoint Hamiltonian cycles."""
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s margin
    
    best_cycles = []
    best_len = 0
    
    # Multiple randomized attempts
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        
        # Work on a copy of adjacency
        adj_cur = [set(ADJ[u]) for u in range(N)]
        cycles = []
        ok = True
        
        for c in range(3):
            if time.time() > deadline - 5:
                ok = False
                break
            
            # Pick a start vertex with max degree in remaining graph
            starts = list(range(N))
            starts.sort(key=lambda v: len(adj_cur[v]), reverse=True)
            # Try a few start vertices
            found = None
            for s in starts[:5]:
                if time.time() > deadline - 5:
                    break
                cyc = _find_hamiltonian_cycle(adj_cur, s, deadline - 5, rng)
                if cyc is not None:
                    found = cyc
                    break
            
            if found is None:
                ok = False
                break
            
            cycles.append(found)
            
            # Remove edges of this cycle
            for i in range(len(found)):
                a = found[i]
                b = found[(i + 1) % len(found)]
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)
        
        if ok and len(cycles) == 3:
            if _verify_cycles(cycles):
                return [[int(v) for v in cyc] for cyc in cycles]
        
        # Track best partial result
        if len(cycles) > best_len and len(cycles) >= 1:
            # Verify the partial set
            if _verify_cycles(cycles):
                best_cycles = list(cycles)
                best_len = len(cycles)
        
        # If we found 2 valid cycles, that's good but keep looking for 3
        if best_len >= 2 and attempt > 50 and time.time() > deadline - 10:
            break
    
    if best_cycles:
        return [[int(v) for v in cyc] for cyc in best_cycles]
    
    # Fallback: try to find at least one Hamiltonian cycle
    rng = random.Random(42)
    adj_cur = [set(ADJ[u]) for u in range(N)]
    for s in range(N):
        cyc = _find_hamiltonian_cycle(adj_cur, s, deadline - 2, rng)
        if cyc is not None:
            if _verify_cycles([cyc]):
                return [[int(v) for v in cyc]]
    
    # Last resort: return a trivial cycle if any exists
    # Find any 3-cycle
    for u in range(N):
        for v in ADJ[u]:
            if u < v:
                for w in ADJ[v]:
                    if w != u and w != v and w in ADJ[u]:
                        cyc = [u, v, w]
                        if _verify_cycles([cyc]):
                            return [[int(x) for x in cyc]]
    
    return [[0, 1, 2]]  # Should never reach here

# EVOLVE-BLOCK-END
