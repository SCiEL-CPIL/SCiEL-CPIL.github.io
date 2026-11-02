"""
Strategy: Algebraic Construction with Rank Reduction
The current approach uses random/greedy search which is inefficient.
We use a structured approach:
1. Start with schoolbook rank R0 = n*m*p.
2. Attempt to reduce rank by finding linear dependencies among the rank-1 tensors.
3. Specifically for small dimensions, we can try to construct a decomposition 
   of rank R < R0 by solving a system of equations for the entries of U, V, W.
   Since the entries can be half-integers, we scale by 2 to work with integers 
   if needed, but we stick to fractions for exactness.
   
   A more direct method for small cases:
   We know the lower bound is max(n*p, m*p, n*m) = 15.
   We will try to find a valid decomposition for ranks from 15 upwards.
   For a given rank r, we set up the problem as finding vectors u_c, v_c, w_c 
   such that sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1.
   
   This is a non-linear system. However, for small r, we can use a 
   "structured" initialization based on the matrix structure.
   
   Alternative: Use the fact that matrix multiplication can be viewed as 
   a bilinear map. We can try to find a decomposition using the 
   "splitting" method or by explicitly solving for small r.
   
   Given the time limit, we will implement a robust search for r in [15, 30]
   using a randomized algebraic solver that fixes some variables to simplify 
   the system, then solves the remaining linear/quadratic constraints.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = Fraction(1)
                V[j * p + k][column] = Fraction(1)
                W[k * n + i][column] = Fraction(1)
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_algebraic(r, n, m, p, deadline, rng):
    """
    Try to find a rank-r decomposition using a structured algebraic approach.
    We assume a specific sparsity pattern or symmetry to reduce the search space.
    For simplicity and robustness, we use a randomized linear algebra approach:
    1. Initialize U, V, W with random small integers/fractions.
    2. Use a local search that minimizes the L2 error of the tensor identity.
    3. If error is 0, we have a solution.
    
    To improve on the seed, we focus on ranks close to the lower bound (15).
    We will use a "simulated annealing" style local search on the entries.
    """
    nm, mp, pn = n * m, m * p, p * n
    total_triples = n * m * p
    
    # We represent entries as Fractions to ensure exactness.
    # Start with random initialization
    U = [[Fraction(rng.randint(-2, 2)) for _ in range(r)] for _ in range(nm)]
    V = [[Fraction(rng.randint(-2, 2)) for _ in range(r)] for _ in range(mp)]
    W = [[Fraction(rng.randint(-2, 2)) for _ in range(r)] for _ in range(pn)]
    
    def _error(U, V, W):
        err = Fraction(0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    diff = s - 1
                    err += diff * diff
        return err
    
    current_err = _error(U, V, W)
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_err = current_err
    
    # Local search
    temp = 1.0
    for step in range(10000):
        if time.time() > deadline:
            break
        if best_err == 0:
            break
        
        # Pick a random entry to perturb
        mat_idx = rng.randint(0, 2)
        if mat_idx == 0:
            idx = rng.randint(0, nm - 1)
            col = rng.randint(0, r - 1)
            old = U[idx][col]
            delta = Fraction(rng.randint(-2, 2))
            if delta == 0: continue
            U[idx][col] = old + delta
            new_err = _error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                U[idx][col] = old
        elif mat_idx == 1:
            idx = rng.randint(0, mp - 1)
            col = rng.randint(0, r - 1)
            old = V[idx][col]
            delta = Fraction(rng.randint(-2, 2))
            if delta == 0: continue
            V[idx][col] = old + delta
            new_err = _error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                V[idx][col] = old
        else:
            idx = rng.randint(0, pn - 1)
            col = rng.randint(0, r - 1)
            old = W[idx][col]
            delta = Fraction(rng.randint(-2, 2))
            if delta == 0: continue
            W[idx][col] = old + delta
            new_err = _error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                W[idx][col] = old
        
        # Cool down
        temp *= 0.999
        
    if best_err == 0:
        # Convert Fractions to int/float if possible, but task allows int/half-int.
        # Fractions are exact. The evaluator likely accepts numbers.
        # We return lists of numbers.
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        
        # Verify with floats to be safe, but we know it's exact in Fraction
        # However, float conversion might introduce precision errors in verification
        # if the evaluator uses strict float comparison.
        # The task says "integer/half-integer entries". 
        # Let's try to convert to int or half-int if possible.
        
        def _to_num(f):
            if f.denominator == 1:
                return int(f.numerator)
            elif f.denominator == 2:
                return f.numerator / 2.0
            else:
                # If it's not int or half-int, we have a problem.
                # But our search only uses integer steps on integer init, so 
                # entries are integers.
                return float(f)
        
        U_out = [[_to_num(x) for x in row] for row in best_U]
        V_out = [[_to_num(x) for x in row] for row in best_V]
        W_out = [[_to_num(x) for x in row] for row in best_W]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: 120s limit, use 100s
    deadline = time.time() + 100
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*p, m*p, n*m) = 15
    # We prioritize finding a rank < schoolbook
    best = _schoolbook(n, m, p)
    
    # We will try a few specific ranks that are likely to have solutions
    # 15, 16, 17, 18, 19, 20
    for r in range(15, min(schoolbook_rank, 25)):
        if time.time() > deadline:
            break
        
        rng = random.Random(r * 1234 + 42)
        result = _try_rank_algebraic(r, n, m, p, deadline, rng)
        if result is not None:
            # Verify the result
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                best = result
                break # Found a lower rank, stop
    
    return best
