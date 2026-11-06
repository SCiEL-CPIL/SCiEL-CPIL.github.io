"""Solve minimum hitting set of maximal cliques via MILP."""
# EVOLVE-BLOCK-START
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        P_union_X = P | X
        if not P_union_X:
            return
        # Pick the vertex with maximum degree in the induced subgraph on P
        best_u = max(P_union_X, key=lambda u: len(adj[u] & P))
        
        candidates = P - adj[best_u]
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _solve_instance(n, p, seed, time_limit=10):
    """Solve the hitting set problem for one graph instance."""
    adj = _generate_graph(n, p, seed)
    cliques = _bron_kerbosch(adj, n)
    
    if not cliques:
        return []
    
    # Build incidence matrix: rows = cliques, cols = vertices
    m = len(cliques)
    num_cliques = m
    num_vars = n
    
    A = np.zeros((num_cliques, num_vars), dtype=float)
    for i, clique in enumerate(cliques):
        for v in clique:
            A[i, v] = 1.0
    
    # Objective: minimize sum of x_v
    c = np.ones(num_vars, dtype=float)
    
    # Constraints: for each clique, sum of x_v over v in clique >= 1
    # This is: A @ x >= 1
    constraints = LinearConstraint(A, 1.0, np.inf)
    integrality = np.ones(num_vars, dtype=int)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit})
        elapsed = time.time() - start_time
        
        if res.status == 0:  # optimal
            x = res.x
            selected = [v for v in range(num_vars) if x[v] > 0.5]
            return sorted(selected)
        elif res.status in (1, 2, 3):  # feasible but not proven optimal
            x = res.x
            selected = [v for v in range(num_vars) if x[v] > 0.5]
            if selected:
                return sorted(selected)
    except Exception:
        pass
    
    # Fallback: greedy heuristic
    return _greedy_hitting_set(cliques, n)

def _greedy_hitting_set(cliques, n):
    """Greedy set cover approximation for hitting set."""
    # Count how many cliques each vertex hits
    vertex_counts = [0] * n
    for clique in cliques:
        for v in clique:
            vertex_counts[v] += 1
    
    remaining_cliques = [set(c) for c in cliques]
    selected = []
    
    while remaining_cliques:
        # Find vertex that hits the most remaining cliques
        best_v = -1
        best_count = -1
        for v in range(n):
            count = sum(1 for rc in remaining_cliques if v in rc)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v == -1 or best_count <= 0:
            break
        
        selected.append(best_v)
        # Remove all cliques containing best_v
        remaining_cliques = [rc for rc in remaining_cliques if best_v not in rc]
    
    return sorted(selected)

def construct():
    results = []
    total_deadline = 140.0  # safety margin before 150s kill
    start = time.time()
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        elapsed = time.time() - start
        remaining = total_deadline - elapsed
        if remaining < 5:
            # Not enough time, use fast greedy for all remaining
            adj = _generate_graph(n, p, seed)
            cliques = _bron_kerbosch(adj, n)
            results.append(_greedy_hitting_set(cliques, n))
        else:
            # Allocate time per instance
            time_limit = min(remaining / 3, 30)
            results.append(_solve_instance(n, p, seed, time_limit=time_limit))
    
    return results
# EVOLVE-BLOCK-END
