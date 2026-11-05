import time
from random import Random
from collections import defaultdict
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_with_ilp(edges, n):
    """
    Decomposes the graph into cycles and edges using Integer Linear Programming.
    Objective: Maximize the number of edges covered by cycles.
    This is equivalent to minimizing the number of single-edge pieces,
    since total_pieces = total_edges - edges_in_cycles + number_of_cycles.
    Wait, total pieces = (# cycles) + (# single edges).
    Let E be total edges. Let C be set of cycles. Let E_c be edges in cycles.
    # single edges = E - E_c.
    Total pieces = |C| + (E - E_c).
    To minimize this, we want to maximize E_c and minimize |C|.
    Since each cycle has at least 3 edges, maximizing E_c generally helps,
    but we must balance the cost of adding a cycle (1 piece) vs saving edges.
    Actually, if we just maximize E_c, we might end up with many small cycles
    which increases |C|.
    
    A better objective: Minimize (|C| + E - E_c) = E - (E_c - |C|).
    So we want to maximize (E_c - |C|).
    For a cycle of length k, it contributes k edges and 1 cycle.
    Value = k - 1.
    So we want to maximize sum(k_c - 1) over all cycles c.
    This is equivalent to maximizing sum(k_c) - |C|.
    
    Let x_c be 1 if cycle c is selected.
    Maximize sum_c (len(c) - 1) * x_c
    Subject to:
    1. Edge disjointness: For each edge e, sum_{c: e in c} x_c <= 1
    2. x_c in {0, 1}
    
    However, enumerating all cycles is exponential.
    For n=30, the number of cycles can be large.
    We can restrict to "simple" cycles up to a certain length or use a column generation approach,
    but with scipy.milp we need to provide the full matrix.
    
    Alternative: Use a heuristic to generate a pool of "good" cycles (e.g., long cycles)
    and solve the ILP on that pool.
    If the pool is small enough, this is fast.
    """
    if not edges:
        return []
        
    # Generate a pool of candidate cycles
    # We use a randomized heuristic to find many cycles.
    # We want a diverse set of long cycles.
    
    candidate_cycles = []
    adj = {u: set() for u in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    remaining_for_search = set(tuple(sorted(e)) for e in edges)
    
    # We will perform multiple passes of cycle extraction to build a pool.
    # But for ILP, we need a static set of candidate cycles.
    # Let's find up to 100-200 good cycles using a fast heuristic.
    
    def find_cycle(adj, n, seed_val):
        rng = Random(seed_val)
        # Start from a random vertex with degree >= 2
        starts = [u for u in range(n) if len(adj[u]) >= 2]
        if not starts:
            return None
        start = rng.choice(starts)
        path = [start]
        visited = {start}
        stack = [(start, iter(sorted(adj[start], key=lambda x: rng.random())))]
        
        while stack:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    return path[:]
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(sorted(adj[nb], key=lambda x: rng.random()))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    visited.discard(path.pop())
        return None

    # Generate pool
    pool = []
    seen_cycles = set()
    # We run this for a fixed number of iterations to build the pool
    # We don't remove edges from adj for this phase, just find cycles in the full graph
    # This ensures we have a rich set of options for the ILP.
    
    max_pool_size = 200
    attempts = 0
    while len(pool) < max_pool_size and attempts < 500:
        c = find_cycle(adj, n, attempts)
        if c:
            # Canonical form for deduplication
            # Rotate to start with min vertex
            min_idx = c.index(min(c))
            c_rot = c[min_idx:] + c[:min_idx]
            # Two directions
            c_rev = c_rot[::-1]
            key1 = tuple(c_rot)
            key2 = tuple(c_rev)
            if key1 not in seen_cycles and key2 not in seen_cycles:
                pool.append(c_rot)
                seen_cycles.add(key1)
                seen_cycles.add(key2)
        attempts += 1
        
    if not pool:
        # No cycles found, return all edges as single edges
        return [[e[0], e[1]] for e in edges]

    # Build ILP
    # Variables: x_i for each cycle i in pool
    # Objective: Maximize sum (len(c_i) - 1) * x_i
    # Constraints: For each edge e, sum_{i: e in c_i} x_i <= 1
    
    m = len(pool)
    if m == 0:
        return [[e[0], e[1]] for e in edges]
        
    # Map edges to indices
    edge_indices = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    num_edges = len(edges)
    
    # Objective coefficients
    c_obj = np.array([len(c) - 1 for c in pool], dtype=float)
    
    # Constraint matrix A: num_edges x m
    # A[e, i] = 1 if edge e is in cycle i
    A = np.zeros((num_edges, m), dtype=float)
    
    for i, c in enumerate(pool):
        for j in range(len(c)):
            u = c[j]
            v = c[(j+1)%len(c)]
            e_key = (u, v) if u < v else (v, u)
            if e_key in edge_indices:
                A[edge_indices[e_key], i] = 1.0
    
    # Constraints: A @ x <= 1
    # x >= 0, x <= 1, integer
    
    constraints = LinearConstraint(A, ub=np.ones(num_edges))
    integrality = np.ones(m, dtype=int)
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    
    # Solve
    try:
        res = milp(c=-c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        # Fallback to greedy if MILP fails
        res = None
        
    if res is None or not res.success:
        # Fallback: Greedy selection from pool
        # Sort cycles by length descending
        sorted_pool = sorted(pool, key=len, reverse=True)
        used_edges = set()
        selected_cycles = []
        for c in sorted_pool:
            c_edges = set()
            valid = True
            for j in range(len(c)):
                u = c[j]
                v = c[(j+1)%len(c)]
                e_key = (u, v) if u < v else (v, u)
                if e_key in used_edges:
                    valid = False
                    break
                c_edges.add(e_key)
            if valid:
                selected_cycles.append(c)
                used_edges.update(c_edges)
                
        pieces = [c for c in selected_cycles]
        # Add remaining edges
        for e in edges:
            e_key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            if e_key not in used_edges:
                pieces.append([e[0], e[1]])
        return pieces
        
    # Extract solution
    x = res.x
    selected_cycles = []
    used_edges = set()
    
    for i in range(m):
        if x[i] > 0.5:
            c = pool[i]
            # Verify edges are unique (should be by constraint, but double check)
            c_edges = set()
            valid = True
            for j in range(len(c)):
                u = c[j]
                v = c[(j+1)%len(c)]
                e_key = (u, v) if u < v else (v, u)
                if e_key in used_edges:
                    valid = False
                    break
                c_edges.add(e_key)
            if valid:
                selected_cycles.append(c)
                used_edges.update(c_edges)
                
    pieces = [c for c in selected_cycles]
    
    # Add remaining edges as single pieces
    for e in edges:
        e_key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        if e_key not in used_edges:
            pieces.append([e[0], e[1]])
            
    return pieces

def decompose_greedy_fallback(edges, n):
    """
    Fast greedy fallback if ILP is too slow or fails.
    """
    if not edges:
        return []
        
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    adj = {u: set() for u in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    rng = Random(42)
    
    while remaining:
        start = None
        for u in range(n):
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
            
        # Find a cycle
        best_cycle = None
        for _ in range(5):
            visited = {start}
            path = [start]
            stack = [(start, iter(sorted(adj[start], key=lambda x: rng.random())))]
            found = False
            while stack and not found:
                node, neighbors = stack[-1]
                advanced = False
                for nb in neighbors:
                    e = (node, nb) if node < nb else (nb, node)
                    if e not in remaining:
                        continue
                    if nb == start and len(path) >= 3:
                        if not best_cycle or len(path) > len(best_cycle):
                            best_cycle = path[:]
                        found = True
                        break
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: rng.random()))))
                        advanced = True
                        break
                if found:
                    break
                if not advanced:
                    stack.pop()
                    if path:
                        visited.discard(path.pop())
                        
        if best_cycle:
            pieces.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i+1)%len(best_cycle)]
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            break
            
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > 100:
            # Use fast fallback
            pieces = decompose_greedy_fallback(edges, n)
        else:
            # Try ILP
            try:
                pieces = decompose_with_ilp(edges, n)
            except Exception:
                pieces = decompose_greedy_fallback(edges, n)
                
        results.append(pieces)
        
        if time.time() - start_time > 120:
            # Fallback for remaining
            for spec in specs[specs.index((n,p,seed))+1:]:
                e = instance(*spec)
                results.append(decompose_greedy_fallback(e, spec[0]))
            break
            
    while len(results) < 3:
        results.append([])
        
    return results[:3]

if __name__ == "__main__":
    construct()
