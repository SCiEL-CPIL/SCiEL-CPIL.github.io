import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    deadline = time.time() + 55
    
    N = 256
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Weights
    c = -np.array([1.0 / x for x in nodes])
    
    # Build conflict matrix
    rows = []
    cols = []
    vals = []
    constraint_idx = 0
    
    for i_idx in range(n_vars):
        i = nodes[i_idx]
        for j_idx in range(i_idx + 1, n_vars):
            j = nodes[j_idx]
            g = gcd(i, j)
            if i * j <= 256 * g:
                rows.append(constraint_idx)
                cols.append(i_idx)
                vals.append(1.0)
                rows.append(constraint_idx)
                cols.append(j_idx)
                vals.append(1.0)
                constraint_idx += 1
    
    num_constraints = constraint_idx
    
    if num_constraints == 0:
        return nodes
    
    A = sp.csr_matrix((vals, (rows, cols)), shape=(num_constraints, n_vars))
    
    lb = np.zeros(num_constraints)
    ub = np.ones(num_constraints)
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)
    
    best_subset = []
    
    try:
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            subset = [nodes[i] for i in range(n_vars) if x[i] > 0.5]
            if subset:
                best_subset = subset
            else:
                best_subset = [nodes[0]]
        else:
            greedy_set = []
            for i in nodes:
                ok = True
                for y in greedy_set:
                    if i * y <= 256 * gcd(i, y):
                        ok = False
                        break
                if ok:
                    greedy_set.append(i)
            best_subset = greedy_set
            
    except Exception:
        greedy_set = []
        for i in nodes:
            ok = True
            for y in greedy_set:
                if i * y <= 256 * gcd(i, y):
                    ok = False
                    break
            if ok:
                greedy_set.append(i)
        best_subset = greedy_set
    
    # Local search refinement
    if time.time() < deadline - 5:
        best_subset = _local_search(best_subset, nodes, deadline)
    
    return best_subset

def _local_search(initial_subset, nodes, deadline):
    """Local search: try to improve the subset by removing and adding elements."""
    current = set(initial_subset)
    current_score = sum(1.0 / x for x in current)
    best = set(current)
    best_score = current_score
    
    # Build adjacency: which pairs conflict
    conflict = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                if i * j <= 256 * gcd(i, j):
                    conflict.setdefault(i, set()).add(j)
    
    def is_valid(s):
        for a in s:
            for b in s:
                if a != b and b in conflict.get(a, set()):
                    return False
        return True
    
    def score(s):
        return sum(1.0 / x for x in s)
    
    # Try single removals
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for elem in list(current):
            candidate = current - {elem}
            if not candidate:
                continue
            s = score(candidate)
            if s > best_score:
                best_score = s
                best = set(candidate)
                current = set(candidate)
                improved = True
    
    # Try single additions
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for elem in nodes:
            if elem in current:
                continue
            candidate = current | {elem}
            if is_valid(candidate):
                s = score(candidate)
                if s > best_score:
                    best_score = s
                    best = set(candidate)
                    current = set(candidate)
                    improved = True
    
    # Try swaps: remove one, add another
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for rem in list(current):
            for add in nodes:
                if add in current or add == rem:
                    continue
                candidate = (current - {rem}) | {add}
                if is_valid(candidate):
                    s = score(candidate)
                    if s > best_score:
                        best_score = s
                        best = set(candidate)
                        current = set(candidate)
                        improved = True
                        break
            if improved:
                break
    
    return list(best) if best else list(current)
