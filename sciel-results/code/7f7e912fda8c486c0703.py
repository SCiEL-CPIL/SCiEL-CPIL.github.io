"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    DEADLINE = 88.0  # safety margin before the 90s kill
    from math import gcd, lcm
    from itertools import combinations
    import numpy as np

    n = 128

    # Precompute LCM matrix
    lcm_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_mat[i][j] = lcm(i, j)
            lcm_mat[j][i] = lcm_mat[i][j]

    # Enumerate every equal-LCM triple (a < b < c) in [1,128].
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_mat[a][b]
            for c in range(b + 1, n + 1):
                if lcm_mat[a][c] == lab and lcm_mat[b][c] == lab:
                    triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    m = len(triples)

    def is_valid(S):
        for a, b, c in combinations(S, 3):
            if lcm_mat[a][b] == lcm_mat[a][c] == lcm_mat[b][c]:
                return False
        return True

    def greedy():
        selected = []
        for x in range(1, n + 1):
            ok = True
            for a, b in combinations(selected, 2):
                if lcm_mat[a][b] == lcm_mat[a][x] == lcm_mat[b][x]:
                    ok = False
                    break
            if ok:
                selected.append(x)
        return selected

    # Greedy incumbent (always valid, nonempty).
    best = greedy()
    best_size = len(best)

    if m == 0:
        return list(range(1, n + 1))

    # Build conflict graph (2-shadow): edge (i,j) if i,j appear in some triple
    # Actually for MIS on hypergraph, we need: two vertices conflict if they can't both be in S
    # But the constraint is on triples, not pairs. So we use the hypergraph directly.
    
    # Better approach: Use local search / simulated annealing on the hypergraph
    # Start from greedy solution and try to add/remove elements
    
    # Try MILP first
    if time.time() - start < DEADLINE:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            c = -np.ones(n)  # minimize -sum x  <=>  maximize |S|
            A = np.zeros((m, n))
            for i, (a, b, cc) in enumerate(triples):
                A[i, a] = 1.0
                A[i, b] = 1.0
                A[i, cc] = 1.0
            cons = LinearConstraint(A, ub=2.0)   # no hyperedge fully chosen
            bounds = Bounds(0.0, 1.0)
            res = milp(c, constraints=cons, bounds=bounds, integrality=np.ones(n))
            if res is not None and res.x is not None:
                sol = [i + 1 for i in range(n) if res.x[i] > 0.5]
                if len(sol) > 0 and is_valid(sol) and len(sol) >= len(best):
                    best = sol
                    best_size = len(best)
        except Exception:
            pass

    # Local search improvement: try to add elements not in best
    # Then try to remove elements that allow adding more
    if time.time() - start < DEADLINE - 5:
        # Convert best to set for fast lookup
        best_set = set(best)
        
        # Try adding each element not in best
        for x in range(1, n + 1):
            if x not in best_set:
                # Check if adding x creates a forbidden triple
                ok = True
                for a, b in combinations(best, 2):
                    if lcm_mat[a][b] == lcm_mat[a][x] == lcm_mat[b][x]:
                        ok = False
                        break
                if ok:
                    best.append(x)
                    best_set.add(x)
                    best_size += 1
        
        # Now try to swap: remove one element, add another
        improved = True
        while improved and time.time() - start < DEADLINE - 3:
            improved = False
            for i in range(len(best)):
                removed = best[i]
                best_set.remove(removed)
                # Try to add elements not in best_set
                for x in range(1, n + 1):
                    if x not in best_set:
                        ok = True
                        for a, b in combinations(best, 2):
                            if lcm_mat[a][b] == lcm_mat[a][x] == lcm_mat[b][x]:
                                ok = False
                                break
                        if ok:
                            # This swap is beneficial if we can add at least one element
                            # and the new set is valid
                            new_best = best + [x]
                            new_best_set = best_set | {x}
                            # Check if we can add more
                            further_adds = 0
                            for y in range(1, n + 1):
                                if y not in new_best_set:
                                    ok2 = True
                                    for a, b in combinations(new_best, 2):
                                        if lcm_mat[a][b] == lcm_mat[a][y] == lcm_mat[b][y]:
                                            ok2 = False
                                            break
                                    if ok2:
                                        further_adds += 1
                            if further_adds > 0:
                                # Accept this swap and all further adds
                                best = new_best
                                best_set = new_best_set
                                best_size += 1
                                for y in range(1, n + 1):
                                    if y not in best_set:
                                        ok2 = True
                                        for a, b in combinations(best, 2):
                                            if lcm_mat[a][b] == lcm_mat[a][y] == lcm_mat[b][y]:
                                                ok2 = False
                                                break
                                        if ok2:
                                            best.append(y)
                                            best_set.add(y)
                                            best_size += 1
                                improved = True
                                break
                # Restore removed
                best_set.add(removed)
                if improved:
                    break

    return best
# EVOLVE-BLOCK-END
