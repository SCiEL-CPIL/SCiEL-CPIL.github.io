"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_VERTS = frozenset(range(N))
EDGE_SET = set(frozenset(e) for e in P['edges'])
ADJ = defaultdict(set)
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Precompute edge list for quick lookup
EDGE_LOOKUP = set()
for u, v in P['edges']:
    EDGE_LOOKUP.add((u, v))
    EDGE_LOOKUP.add((v, u))


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _is_valid_cycle(cycle, used_edges):
    """Check if cycle is valid: distinct vertices, all edges exist and unused."""
    if len(cycle) < 3:
        return False
    verts = set(cycle)
    if len(verts) != len(cycle):
        return False
    if verts != ALL_VERTS:
        return False
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        ek = _edge_key(u, v)
        if ek not in EDGE_LOOKUP:
            return False
        if ek in used_edges:
            return False
    return True


def _find_hamiltonian_cycle(used_edges, start=0, deadline=None, max_steps=500000):
    """Find a Hamiltonian cycle avoiding used_edges using backtracking."""
    n = N
    # Fix start vertex to 0 to reduce symmetry
    # Try to build cycle: 0 -> ... -> 0
    
    # Precompute available neighbors
    avail = {}
    for v in range(n):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    
    # Sort neighbors by degree for heuristic (least constraining first)
    for v in range(n):
        avail[v].sort(key=lambda u: len(avail[u]))
    
    path = [0]
    visited = {0}
    
    sys.setrecursionlimit(10000)
    
    steps = [0]
    
    def dfs():
        if deadline is not None and time.time() > deadline:
            return None
        steps[0] += 1
        if steps[0] > max_steps:
            return None
        
        if len(path) == n:
            # Check if last vertex connects to start
            last = path[-1]
            if _edge_key(last, 0) not in used_edges and (last, 0) in EDGE_LOOKUP:
                return list(path)
            return None
        
        current = path[-1]
        for nxt in avail[current]:
            if nxt in visited:
                continue
            if nxt == 0:
                continue  # Can't go back to start until end
            visited.add(nxt)
            path.append(nxt)
            
            result = dfs()
            if result is not None:
                return result
            
            path.pop()
            visited.discard(nxt)
        
        return None
    
    result = dfs()
    return result


def _try_random_start(used_edges, deadline, max_steps=200000):
    """Try finding Hamiltonian cycle with random neighbor ordering."""
    n = N
    start = random.randint(0, n - 1)
    
    avail = {}
    for v in range(n):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    
    # Randomize neighbor order
    for v in range(n):
        random.shuffle(avail[v])
    
    path = [start]
    visited = {start}
    
    sys.setrecursionlimit(10000)
    steps = [0]
    
    def dfs():
        if deadline is not None and time.time() > deadline:
            return None
        steps[0] += 1
        if steps[0] > max_steps:
            return None
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_edges and (last, start) in EDGE_LOOKUP:
                return list(path)
            return None
        
        current = path[-1]
        for nxt in avail[current]:
            if nxt in visited:
                continue
            if nxt == start:
                continue
            visited.add(nxt)
            path.append(nxt)
            
            result = dfs()
            if result is not None:
                return result
            
            path.pop()
            visited.discard(nxt)
        
        return None
    
    result = dfs()
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # First, try to find one Hamiltonian cycle to establish baseline
    # If we can't find even one, return empty (but task says nonempty)
    # The seed guarantees at least one exists (0..20 is a Hamiltonian cycle)
    
    # Use the known Hamiltonian cycle from seed as fallback
    fallback_cycle = list(range(N))
    
    cycles = []
    used_edges = set()
    
    # Strategy: greedily find edge-disjoint Hamiltonian cycles
    # Try multiple strategies with different random seeds
    
    best_cycles = []
    
    # First attempt: deterministic with sorted neighbors
    deadline1 = min(deadline, start_time + 15)
    cycle = _find_hamiltonian_cycle(used_edges, start=0, deadline=deadline1)
    if cycle is not None:
        best_cycles.append(cycle)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
    
    # If we didn't find one, use the fallback
    if not best_cycles:
        best_cycles.append(fallback_cycle)
        for i in range(len(fallback_cycle)):
            u = fallback_cycle[i]
            v = fallback_cycle[(i + 1) % len(fallback_cycle)]
            used_edges.add(_edge_key(u, v))
    
    # Continue finding more cycles with random restarts
    random.seed(42)
    
    for attempt in range(200):
        if time.time() >= deadline - 5:
            break
        if len(best_cycles) >= 10:
            break
        
        deadline_attempt = min(deadline - 2, time.time() + 8)
        cycle = _try_random_start(used_edges, deadline=deadline_attempt, max_steps=100000)
        
        if cycle is not None:
            # Verify it's valid
            if _is_valid_cycle(cycle, used_edges):
                best_cycles.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    used_edges.add(_edge_key(u, v))
    
    # Return the cycles as lists of ints
    result = [list(c) for c in best_cycles[:10]]
    
    # Ensure types are correct
    for i, cycle in enumerate(result):
        result[i] = [int(v) for v in cycle]
    
    return result


# EVOLVE-BLOCK-END
