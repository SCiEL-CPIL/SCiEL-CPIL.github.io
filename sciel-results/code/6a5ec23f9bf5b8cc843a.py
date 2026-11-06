"""
Strategy: Integer Linear Programming (ILP) formulation.

Problem:
Maximize sum(x_i) for i in 1..128
Subject to:
1. x_i in {0, 1}
2. For every triple (a, b, c) with a < b < c, if gcd(a,b) == gcd(a,c) == gcd(b,c) == g,
   then x_a + x_b + x_c <= 2.
   (This forbids selecting all three elements of a monochromatic triangle).

Why this is different:
The current program uses stochastic local search (greedy + swaps). ILP explores the
solution space using branch-and-bound/cut methods which are global in nature.
For N=128, the number of constraints is manageable (number of monochromatic triples),
and the solver should be able to find the optimal or near-optimal solution quickly.
"""

import time
import random
from math import gcd
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Identify all "bad" triples (monochromatic triangles)
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # Note: We only need to consider distinct a, b, c.
    bad_triples = []
    
    # Optimization: Group by GCD value?
    # Actually, just iterating O(N^3) is 128^3 / 6 ~ 350,000 iterations.
    # This is very fast in Python if optimized, or we can use NumPy.
    # Let's use a list comprehension or simple loop.
    
    # To speed up, we can precompute for each pair (a,b) the gcd g_ab.
    # Then for each c > b, check if gcd(a,c)==g_ab and gcd(b,c)==g_ab.
    
    # Even faster: 
    # For each possible gcd value g, consider the graph G_g where edges are (u,v) with gcd(u,v)=g.
    # Any triangle in G_g is a bad triple.
    # We can find all triangles in these sparse graphs.
    
    # However, simply iterating all triples is probably fast enough for N=128.
    # 128 * 127 * 126 / 6 = 341,376 triples.
    # Checking 3 gcd lookups per triple is ~1 million ops. Negligible.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    # Formulate ILP
    # Variables x_0 ... x_{N-1} corresponding to numbers 1 ... N
    # Maximize sum(x)
    # Constraints:
    # For each (a,b,c) in bad_triples: x_{a-1} + x_{b-1} + x_{c-1} <= 2
    
    c_obj = -1.0 * [1.0] * N  # Minimize -sum(x) equivalent to Maximize sum(x)
    
    if not bad_triples:
        # No constraints, take all
        return list(range(1, N + 1))
    
    A_ub = []
    b_ub = []
    
    for a, b, c in bad_triples:
        row = [0.0] * N
        row[a - 1] = 1.0
        row[b - 1] = 1.0
        row[c - 1] = 1.0
        A_ub.append(row)
        b_ub.append(2.0)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: x_i must be integer
    integrality = [1] * N
    
    # Solve
    try:
        res = milp(
            c=c_obj,
            constraints=[LinearConstraint(A_ub, -1e9, 2)],
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # Due to floating point, threshold at 0.5
            solution_mask = [1 if xi > 0.5 else 0 for xi in res.x]
            selected = [i + 1 for i, val in enumerate(solution_mask) if val == 1]
            
            # Verify validity just in case (defensive)
            # The ILP constraint ensures no bad triple is fully selected.
            # So it should be valid.
            
            if len(selected) > 0:
                return selected
            else:
                # Fallback if solver returns empty (shouldn't happen for this problem)
                return [1]
        else:
            # If solver fails or times out, we might have a partial solution or need fallback
            # res.status 2 = Infeasible, 3 = Unbounded, 4 = Iteration limit, etc.
            # If we hit iteration limit, res.x might contain a feasible point if 'polish' was used or if it found one.
            # But milp returns x only if it found a solution.
            # If status is not 0, it's safer to fall back to a heuristic.
            
            # However, for N=128, this ILP is small. It should solve quickly.
            # If it didn't, let's try to recover a feasible solution from res.x if available
            if hasattr(res, 'x') and res.x is not None:
                solution_mask = [1 if xi > 0.5 else 0 for xi in res.x]
                selected = [i + 1 for i, val in enumerate(solution_mask) if val == 1]
                if len(selected) > 0:
                    return selected
            
            # Fallback: Simple greedy (same as seed but maybe better ordering)
            # Or just return a small valid set
            selected = []
            for i in range(1, N+1):
                ok = True
                for j in range(len(selected)):
                    a = selected[j]
                    for k in range(j+1, len(selected)):
                        b = selected[k]
                        if gcd_table[a][b] == gcd_table[a][i] == gcd_table[b][i]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    selected.append(i)
            return selected
            
    except Exception as e:
        # Fallback to greedy if scipy fails
        selected = []
        for i in range(1, N+1):
            ok = True
            for j in range(len(selected)):
                a = selected[j]
                for k in range(j+1, len(selected)):
                    b = selected[k]
                    if gcd_table[a][b] == gcd_table[a][i] == gcd_table[b][i]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(i)
        return selected

# EVOLVE-BLOCK-END
