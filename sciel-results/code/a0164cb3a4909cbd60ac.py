import time
from math import gcd
import random

def construct():
    deadline = time.time() + 85  # safety margin under 90s
    
    all_moduli = list(range(2, 129))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
    
    def try_greedy(order):
        """Greedy: process moduli in given order, assign first valid residue."""
        assignment = {}
        for idx in order:
            m = all_moduli[idx]
            found = False
            for r in range(m):
                ok = True
                for j, rj in assignment.items():
                    g = gcd_mat[idx][j]
                    if (r - rj) % g == 0:
                        ok = False
                        break
                if ok:
                    assignment[idx] = r
                    found = True
                    break
            if not found:
                return None
        return assignment
    
    def local_improve(assignment):
        """Try to add more moduli to the assignment."""
        if assignment is None:
            return None
        assigned_set = set(assignment.keys())
        improved = True
        while improved:
            improved = False
            for idx in range(n):
                if idx in assigned_set:
                    continue
                m = all_moduli[idx]
                for r in range(m):
                    ok = True
                    for j, rj in assignment.items():
                        g = gcd_mat[idx][j]
                        if (r - rj) % g == 0:
                            ok = False
                            break
                    if ok:
                        assignment[idx] = r
                        assigned_set.add(idx)
                        improved = True
                        break
        return assignment
    
    best_size = 0
    best_result = None
    
    # Try several deterministic orderings
    # Order 1: by descending modulus
    order1 = sorted(range(n), key=lambda i: -all_moduli[i])
    # Order 2: by ascending modulus
    order2 = sorted(range(n), key=lambda i: all_moduli[i])
    # Order 3: by descending max gcd
    max_gcds = [max(gcd_mat[i]) for i in range(n)]
    order3 = sorted(range(n), key=lambda i: -max_gcds[i])
    # Order 4: by ascending min gcd (most constrained first)
    min_gcds = [min(gcd_mat[i][j] for j in range(n) if j != i) for i in range(n)]
    order4 = sorted(range(n), key=lambda i: min_gcds[i])
    
    for order in [order1, order2, order3, order4]:
        if time.time() > deadline:
            break
        result = try_greedy(order)
        if result:
            result = local_improve(result)
            size = len(result)
            if size > best_size:
                best_size = size
                best_result = [[result[i], all_moduli[i]] for i in result]
    
    # Try random orderings
    random.seed(42)
    for trial in range(20000):
        if time.time() > deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        result = try_greedy(order)
        if result:
            result = local_improve(result)
            size = len(result)
            if size > best_size:
                best_size = size
                best_result = [[result[i], all_moduli[i]] for i in result]
    
    if best_result:
        return best_result
    
    return [[0, 2]]
