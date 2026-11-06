"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_ilp(edges, n, seed, time_limit=30.0):
    """
    Use ILP to find a set of edge-disjoint cycles that maximize total edges covered.
    Returns list of cycles (each a list of vertices).
    """
    if not edges:
        return []
    
    m = len(edges)
    if m < 3:
        return []
    
    # Enumerate all simple cycles up to length 8 in the graph
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set(edges)
    
    # Find all cycles of length 3-8
    all_cycles = []
    seen_cycle_edges = set()
    
    # Enumerate cycles by trying all starting vertices and doing bounded DFS
    for start in range(n):
        if not adj[start]:
            continue
        # DFS to find cycles starting and ending at start
        # Limit depth to 7 (cycle length up to 8)
        stack = [(start, [start])]
        visited_in_dfs = set()
        
        while stack:
            node, path = stack.pop()
            if len(path) > 8:
                continue
            
            for nb in sorted(adj[node]):
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = list(path)
                    # Canonical form: rotate so min vertex is first
                    min_idx = cycle.index(min(cycle))
                    canonical = tuple(cycle[min_idx:] + cycle[:min_idx])
                    # Check if edges are same as another cycle (reverse)
                    rev = list(reversed(cycle))
                    rev_min_idx = rev.index(min(rev))
                    rev_canonical = tuple(rev[rev_min_idx:] + rev[:rev_min_idx])
                    
                    key = frozenset([canonical, rev_canonical])
                    if key not in seen_cycle_edges:
                        seen_cycle_edges.add(key)
                        all_cycles.append(cycle)
                elif nb not in path and len(path) < 8:
                    stack.append((nb, path + [nb]))
    
    if not all_cycles:
        return []
    
    num_cycles = len(all_cycles)
    if num_cycles == 0:
        return []
    
    # Build constraint matrix: each row is an edge, each column is a cycle
    # A[i,j] = 1 if edge i is in cycle j
    A = np.zeros((m, num_cycles))
    edge_index = {e: i for i, e in enumerate(edges)}
    
    for j, cycle in enumerate(all_cycles):
        for k in range(len(cycle)):
            u = cycle[k]
            v = cycle[(k + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in edge_index:
                A[edge_index[e], j] = 1.0
    
    # Objective: maximize sum of edges covered = sum_j (sum of edges in cycle j) * x_j
    # But we want to minimize pieces, so we want to maximize edges covered by cycles
    # c_j = number of edges in cycle j (which is len(cycle_j))
    c = np.array([len(cyc) for cyc in all_cycles], dtype=float)
    
    # ILP: maximize c^T x subject to A x <= 1 (each edge used at most once), x binary
    # scipy milp minimizes, so we minimize -c^T x
    obj = -c
    
    constraints = [LinearConstraint(A, ub=np.ones(m))]
    integrality = np.ones(num_cycles)
    bounds = Bounds(lb=np.zeros(num_cycles), ub=np.ones(num_cycles))
    
    start_time = time.time()
    try:
        result = milp(c=obj, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        return []
    
    elapsed = time.time() - start_time
    
    if result.status != 0 and result.status != 1:  # 0=optimal, 1=feasible
        return []
    
    if result.x is None:
        return []
    
    # Extract selected cycles
    selected = []
    for j in range(num_cycles):
        if result.x[j] > 0.5:
            selected.append(all_cycles[j])
    
    return selected

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Phase 1: Try ILP to find good cycles
    cycles = _find_cycles_ilp(edges, n, seed, time_limit=min(20.0, time_limit * 0.5))
    
    # Build adjacency from remaining edges
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    
    # Use ILP-found cycles first
    for cycle in cycles:
        if time.time() > deadline:
            break
        cycle_edges = set()
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e not in remaining:
                valid = False
                break
            cycle_edges.add(e)
        
        if valid:
            for e in cycle_edges:
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle)
    
    # Phase 2: Greedy cycle peeling on remaining edges
    cycle_counter = 0
    while remaining:
        if time.time() > deadline - 1.0:
            for e in remaining:
                pieces.append([e[0], e[1]])
            break
        
        # Find a cycle in remaining graph
        best_cycle = []
        vertices_with_edges = [v for v in adj if adj[v]]
        
        if not vertices_with_edges:
            break
        
        # Try multiple starting points
        rng = random.Random(seed + cycle_counter)
        rng.shuffle(vertices_with_edges)
        
        for start in vertices_with_edges[:6]:
            if time.time() > deadline - 1.0:
                break
            if not adj[start]:
                continue
            
            # Greedy longest path
            path = [start]
            visited = {start}
            
            while True:
                u = path[-1]
                candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
                if not candidates:
                    break
                candidates.sort(reverse=True)
                next_v = candidates[0][1]
                path.append(next_v)
                visited.add(next_v)
            
            # Check for cycle closure
            if len(path) >= 3 and start in adj[path[-1]]:
                if len(path) > len(best_cycle):
                    best_cycle = list(path)
            
            # Look for any cycle in the path
            for i in range(len(path) - 2):
                if time.time() > deadline - 1.0:
                    break
                for j in range(i + 2, len(path)):
                    if path[i] in adj[path[j]]:
                        cycle = path[i:j+1]
                        if len(cycle) > len(best_cycle):
                            best_cycle = list(cycle)
                        break
            
            if len(best_cycle) >= n - 1:
                break
        
        if best_cycle:
            cycle_edges = set()
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                cycle_edges.add(e)
            
            for e in cycle_edges:
                if e in remaining:
                    remaining.discard(e)
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            pieces.append(best_cycle)
            cycle_counter += 1
        else:
            # No cycle, take single edge
            e = min(remaining, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e))
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_deadline = time.time() + 140.0
    
    for idx, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        remaining_time = total_deadline - time.time()
        graphs_left = len(specs) - idx
        time_for_this = max(10.0, remaining_time / graphs_left - 1.0)
        
        pieces = _decompose(edges, n, seed, time_limit=time_for_this)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
