"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_all_cycles_on_vertex_set(vertices, adj, max_cycles=200):
    """Find all simple cycles that use exactly the given vertex set."""
    vset = frozenset(vertices)
    n = len(vertices)
    if n < 3:
        return []
    
    # Fix a canonical starting vertex (smallest) to avoid duplicates
    start = min(vertices)
    cycles = []
    used = [False] * P['n']
    
    def dfs(current, path, depth):
        if len(cycles) >= max_cycles:
            return
        if depth == n:
            # Check if we can close the cycle back to start
            if start in adj[current]:
                cycle = list(path)
                cycles.append(cycle)
            return
        for nxt in adj[current]:
            if not used[nxt] and nxt in vset and nxt != start:
                used[nxt] = True
                path.append(nxt)
                dfs(nxt, path, depth + 1)
                path.pop()
                used[nxt] = False
    
    used[start] = True
    dfs(start, [start], 1)
    return cycles

def _canonical_cycle(cycle):
    """Return canonical form of a cycle for deduplication."""
    # Rotate so smallest vertex is first
    min_idx = cycle.index(min(cycle))
    rotated = cycle[min_idx:] + cycle[:min_idx]
    # Two directions are equivalent
    rev = [rotated[0]] + rotated[1:][::-1]
    return tuple(min(rotated, rev))

def _edges_in_cycle(cycle):
    """Return set of edges (as frozensets) in the cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add(frozenset((u, v)))
    return edges

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    adj = _build_adjacency()
    
    # Try vertex sets of different sizes, starting from largest
    # The graph has 21 vertices. We need to find vertex sets of size >= 3
    # where we can pack many edge-disjoint cycles.
    
    # Strategy: Try all subsets of vertices of various sizes
    # For efficiency, start with size 21, then 20, etc.
    
    best_result = []
    
    n = P['n']
    all_vertices = list(range(n))
    
    # Try different subset sizes
    for size in range(n, 2, -1):
        if time.time() > deadline:
            break
        
        # Generate combinations of vertices of given size
        # For large sizes, this is infeasible, so we need to be smart
        if size > 15:
            # Only try the full vertex set and near-full sets
            if size == n:
                subsets = [all_vertices]
            else:
                # Try removing a few vertices
                from itertools import combinations
                removed = list(combinations(range(n), n - size))
                subsets = []
                for rem in removed:
                    rem_set = set(rem)
                    vs = [v for v in range(n) if v not in rem_set]
                    subsets.append(vs)
                if len(subsets) > 100:
                    # Too many, sample
                    import random
                    random.seed(42)
                    subsets = random.sample(subsets, 100)
        else:
            from itertools import combinations
            subsets = list(combinations(range(n), size))
            subsets = [list(s) for s in subsets]
            if len(subsets) > 500:
                import random
                random.seed(42)
                subsets = random.sample(subsets, 500)
        
        for vs in subsets:
            if time.time() > deadline:
                break
            
            # Find all cycles on this vertex set
            cycles = _find_all_cycles_on_vertex_set(vs, adj, max_cycles=500)
            if not cycles:
                continue
            
            # Deduplicate cycles
            canon_map = {}
            for c in cycles:
                key = _canonical_cycle(c)
                if key not in canon_map:
                    canon_map[key] = c
            unique_cycles = list(canon_map.values())
            
            if len(unique_cycles) < 1:
                continue
            
            # Greedy packing: try to pack as many edge-disjoint cycles as possible
            # Sort by length (shorter cycles use fewer edges, might allow more packing)
            unique_cycles.sort(key=len)
            
            used_edges = set()
            packed = []
            for c in unique_cycles:
                if time.time() > deadline:
                    break
                ce = _edges_in_cycle(c)
                if not ce & used_edges:
                    used_edges |= ce
                    packed.append(c)
            
            if len(packed) > len(best_result):
                best_result = packed
            
            # Early exit if we found a good number
            if len(packed) >= 10:
                break
        
        # If we found enough, stop
        if len(best_result) >= 10:
            break
    
    # Final validation and conversion
    result = []
    for c in best_result:
        # Ensure all are Python ints and list type
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
