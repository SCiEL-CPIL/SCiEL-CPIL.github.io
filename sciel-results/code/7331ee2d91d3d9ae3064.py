"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

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
            cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X with max |P ∩ N(u)|
        if P:
            pivot_candidates = list(P)
        elif X:
            pivot_candidates = list(X)
        else:
            pivot_candidates = []
        
        if pivot_candidates:
            # Pick pivot with max degree into P
            pivot = max(pivot_candidates, key=lambda u: len(P & adj[u]))
            # Only recurse on P \ N(pivot)
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            X.add(v)
            bk(new_R, new_P, new_X)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _solve_hitting_set_ilp(cliques, n):
    """
    Solves the hitting set problem using Integer Linear Programming.
    """
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # A: constraint matrix, shape (num_cliques, n)
    # A[i, j] = 1 if vertex j is in clique i
    A = np.zeros((num_cliques, n), dtype=int)
    for i, c in enumerate(cliques):
        for v in c:
            A[i, v] = 1
            
    # Objective: minimize sum(x)
    c_obj = np.ones(n)
    
    # Constraints: A @ x >= 1
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub) where lb <= A @ x <= ub
    # We want A @ x >= 1, so lb = 1, ub = inf
    constraint = LinearConstraint(A, lb=1, ub=np.inf)
    
    # Bounds for x: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(n)
    
    try:
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
        if res.status == 0:
            # res.x is the solution vector
            selected = [i for i in range(n) if res.x[i] > 0.5]
            return sorted(selected)
        else:
            # Fallback to greedy if ILP fails
            return _greedy_hitting_set(cliques, n)
    except Exception:
        return _greedy_hitting_set(cliques, n)

def _greedy_hitting_set(cliques, n):
    """Fallback greedy heuristic."""
    vertex_to_cliques = defaultdict(set)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].add(i)
    
    S = set()
    uncovered = set(range(len(cliques)))
    while uncovered:
        best_v = None
        best_count = 0
        for v in range(n):
            if v in S:
                continue
            cnt = len(uncovered & vertex_to_cliques[v])
            if cnt > best_count:
                best_count = cnt
                best_v = v
        if best_v is None:
            break
        S.add(best_v)
        uncovered -= vertex_to_cliques[best_v]
    return sorted(S)

def construct():
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = _generate_graph(n, p, seed)
        all_cliques = _bron_kerbosch(adj, n)
        # Filter to cliques with at least 2 vertices
        cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Use ILP solver
        hit = _solve_hitting_set_ilp(cliques, n)
        results.append(hit)
    
    return results
# EVOLVE-BLOCK-END
