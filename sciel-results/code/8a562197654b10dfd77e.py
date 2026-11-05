"""MILP-based linear triple system construction for 16 points."""
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _calc_alpha(triples):
    """Compute independence number: largest subset containing no triple."""
    n = 16
    tset = set(tuple(sorted(t)) for t in triples)
    best = 0
    # Check all subsets by size, descending
    for r in range(n, best, -1):
        found = False
        for subset in itertools.combinations(range(n), r):
            s = set(subset)
            has_triple = False
            for t in tset:
                if t[0] in s and t[1] in s and t[2] in s:
                    has_triple = True
                    break
            if not has_triple:
                best = r
                found = True
                break
        if found:
            break
    return best

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 16
    # All possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    m = len(all_triples)  # 560
    
    # All pairs
    all_pairs = list(itertools.combinations(range(n), 2))
    p = len(all_pairs)  # 120
    
    # Build incidence: pair_j is in triple_i
    # A[i,j] = 1 if pair j is contained in triple i
    A = np.zeros((p, m), dtype=np.int8)
    for i, tri in enumerate(all_triples):
        for j, pair in enumerate(all_pairs):
            if pair[0] in tri and pair[1] in tri:
                A[i, j] = 1
    
    # MILP: maximize sum(x_i) subject to A^T x <= 1 (each pair in at most 1 triple)
    # x_i in {0,1}
    # c = -1 for maximization (milp minimizes)
    c = -np.ones(m)
    
    # Constraints: A^T @ x <= 1 for each pair
    # A^T is (m, p), x is (m,), result is (p,)
    # We need A^T @ x <= 1
    # LinearConstraint: lower <= A_con @ x <= upper
    # A_con = A.T (p rows? No: A is (p, m), A.T is (m, p))
    # We want for each pair j: sum_i A[i,j] * x_i <= 1
    # That's A.T @ x <= 1, where A.T is (m, p)
    # But LinearConstraint expects (num_constraints, num_vars)
    # num_constraints = p (120), num_vars = m (560)
    # So we need matrix of shape (p, m) which is A itself
    # A @ x <= 1
    A_con = A  # (p, m)
    lower = np.zeros(p)
    upper = np.ones(p)
    constraints = LinearConstraint(A_con, lower, upper)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    
    # Integrality: all binary
    integrality = np.ones(m)
    
    try:
        result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if result.status == 0 and result.x is not None:
            x = np.round(result.x)
            selected = [all_triples[i] for i in range(m) if x[i] > 0.5]
            triples = [list(t) for t in selected]
            
            # Verify linearity
            pair_count = {}
            for t in triples:
                for a, b in itertools.combinations(t, 2):
                    key = (min(a,b), max(a,b))
                    pair_count[key] = pair_count.get(key, 0) + 1
            if all(v <= 1 for v in pair_count.values()):
                return triples
            else:
                # Fallback: greedy
                pass
        
        # Fallback: greedy construction
        triples = []
        used_pairs = set()
        for tri in all_triples:
            pairs_in_tri = [tuple(sorted((a, b))) for a, b in itertools.combinations(tri, 2)]
            if not any(p in used_pairs for p in pairs_in_tri):
                triples.append(list(tri))
                used_pairs.update(pairs_in_tri)
        return triples
        
    except Exception:
        # Fallback: greedy construction
        triples = []
        used_pairs = set()
        for tri in all_triples:
            pairs_in_tri = [tuple(sorted((a, b))) for a, b in itertools.combinations(tri, 2)]
            if not any(p in used_pairs for p in pairs_in_tri):
                triples.append(list(tri))
                used_pairs.update(pairs_in_tri)
        return triples
