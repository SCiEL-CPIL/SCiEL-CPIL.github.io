import random
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def calc_alpha_milp(triples):
    """
    Calculate the independence number using MILP.
    Returns (alpha, max_independent_set)
    """
    n = 16
    if not triples:
        return n, list(range(n))
    
    # Objective: Maximize sum(x)
    c = -np.ones(n) # Minimize -sum(x)
    
    # Constraints: For each triple, sum(x_i for i in triple) <= 2
    # A[i, j] is 1 if vertex j is in triple i
    A = np.zeros((len(triples), n))
    for i, t in enumerate(triples):
        for v in t:
            A[i, v] = 1.0
            
    # Constraints: A @ x <= 2
    # x >= 0, x <= 1
    # Integrality: x is integer
    
    constraints = [
        LinearConstraint(A, ub=2),
        LinearConstraint(np.eye(n), lb=0, ub=1)
    ]
    
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            alpha = int(round(-res.fun))
            # Get the set
            x = res.x
            independent_set = [i for i in range(n) if x[i] > 0.5]
            return alpha, independent_set
        else:
            # Fallback to brute force if MILP fails
            return calc_alpha_brute(triples)
    except Exception:
        return calc_alpha_brute(triples)

def calc_alpha_brute(triples):
    n = 16
    triple_sets = [set(t) for t in triples]
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            s = set(subset)
            is_independent = True
            for t_set in triple_sets:
                if t_set.issubset(s):
                    is_independent = False
                    break
            if is_independent:
                return k, list(subset)
    return 1, [0]

def is_linear(triple, existing_triples):
    a, b, c = triple
    new_pairs = {tuple(sorted((a,b))), tuple(sorted((a,c))), tuple(sorted((b,c)))}
    for t in existing_triples:
        x, y, z = t
        old_pairs = {tuple(sorted((x,y))), tuple(sorted((x,z))), tuple(sorted((y,z)))}
        if new_pairs & old_pairs:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_triples = None
    best_alpha = 16
    
    def random_init():
        triples = []
        used_pairs = set()
        # Greedy random
        for _ in range(200):
            a, b, c = random.sample(range(16), 3)
            p1 = tuple(sorted((a,b)))
            p2 = tuple(sorted((a,c)))
            p3 = tuple(sorted((b,c)))
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                triples.append([a,b,c])
        return triples

    for restart in range(5000):
        if time.time() > deadline:
            break
            
        triples = random_init()
        if not triples:
            continue
            
        alpha, max_set = calc_alpha_milp(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 6:
                return best_triples

        # Optimization Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            alpha, max_set = calc_alpha_milp(triples)
            
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
                if best_alpha <= 6:
                    return best_triples
            
            # Find linear triples in max_set
            candidates = []
            for t in itertools.combinations(max_set, 3):
                if is_linear(t, triples):
                    candidates.append(list(t))
            
            if candidates:
                # Pick a random candidate
                new_triple = random.choice(candidates)
                triples.append(new_triple)
                improved = True
            else:
                # Stuck. Try a kick?
                # For now, break and restart.
                break
                
    return best_triples
