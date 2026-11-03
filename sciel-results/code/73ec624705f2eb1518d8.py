"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for c in range(rank):
        if len(U[0][c] if isinstance(U[0][c], list) else [U[0][c]]) != 1: # Simple shape check placeholder
            pass
    # U is list of lists: U[idx][c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_csp_rank(r, n, m, p, deadline, rng):
    """
    Try to find a rank-r decomposition using Backtracking CSP.
    We restrict values to small integers/half-integers to keep the search space manageable.
    We use symmetry breaking: 
    1. Sort columns? No, columns are ordered in the sum, but the set of columns is what matters.
       However, the indices i,j,k are fixed.
    2. Fix the first non-zero element of the first column to 1 (scaling).
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Candidate values for entries
    # To ensure exactness, we use Fractions.
    # Search space: {-1, -0.5, 0, 0.5, 1} is a good start.
    # If r is small, we can expand.
    vals = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    
    U = [[Fraction(0) for _ in range(r)] for _ in range(nm)]
    V = [[Fraction(0) for _ in range(r)] for _ in range(mp)]
    W = [[Fraction(0) for _ in range(r)] for _ in range(pn)]
    
    # We will fill column by column? Or entry by entry?
    # Entry by entry is too slow.
    # Column by column is better because each column contributes a rank-1 tensor.
    # Let's try to assign columns one by one.
    # State: current column index c, and the current partial sums for each (i,j,k).
    
    # To speed up, we can pre-compute the "support" of each column.
    # A column (u, v, w) satisfies a triple (i,j,k) if u[i,j]*v[j,k]*w[k,i] == 1.
    # It "helps" if the product is non-zero.
    # It "hurts" if the product makes the sum exceed 1 or go wrong.
    
    # Since we need the sum to be exactly 1, and we start from 0,
    # we need the sum of products to be 1.
    
    # Strategy:
    # 1. Generate all possible rank-1 patterns (u, v, w) from the value set.
    # 2. This is too many: (2*nm+1)*(2*mp+1)*(2*pn+1) approx.
    #    nm=6, mp=15, pn=10.
    #    13 * 31 * 21 ~ 8000 patterns.
    # 3. For each pattern, compute the vector of contributions to the 30 triples.
    # 4. We need to select r patterns such that the sum of their contribution vectors is the all-ones vector.
    #    This is a Subset Sum / Vector Partition problem.
    #    Since r is up to 30, and we want r < 30, this is hard.
    #    
    # Alternative: Backtracking on columns.
    # At step c, we have a residual vector R of size nmp.
    # We need to pick a pattern P such that R - P has non-negative entries? 
    # No, entries can be negative in intermediate steps if we allow negative values.
    # But if we restrict to positive values (1, 0.5), then we must have R >= P element-wise.
    # This allows for a much more efficient search (Knapsack-like).
    
    # Let's restrict to positive half-integers: {0, 0.5, 1}
    # This ensures that if we start with R=1 and subtract P, we stay >= 0.
    # We need to reach R=0.
    
    # Pre-compute all valid positive patterns.
    # A pattern is a tuple (u_list, v_list, w_list).
    # u_list has length nm, v_list length mp, w_list length pn.
    # Values in {0, 0.5, 1}.
    # To reduce patterns, we can require that u, v, w are not all zero.
    # Also, we can normalize: if the max entry is 1, keep it. If max is 0.5, scale up? 
    # No, scaling changes the contribution.
    # However, if we use only 0 and 1, we are in the Boolean case.
    # Let's try Boolean first (values 0, 1).
    
    bool_vals = [Fraction(0), Fraction(1)]
    
    patterns = []
    # Generate patterns with at least one 1 in each vector? 
    # Actually, if u is all 0, the contribution is 0. Useless.
    # We can generate all combinations of 0/1 for u, v, w.
    # But 2^6 * 2^15 * 2^10 is huge.
    
    # We need a smarter way.
    # Observe that the contribution to (i,j,k) is u[i,j] * v[j,k] * w[k,i].
    # For a pattern to be "useful", it must contribute to at least one triple.
    
    # Let's use a heuristic search:
    # 1. Initialize Residual R[i,j,k] = 1.
    # 2. For c in range(r):
    #    Find a pattern P that maximizes the number of triples where R[i,j,k] >= P[i,j,k] and P[i,j,k] > 0.
    #    Subtract P from R.
    #    If R becomes all 0, success.
    #    If no pattern fits (R < P for all non-zero P), fail.
    
    # This is a greedy algorithm, but with a specific structure (positive values only).
    # It is different from the random float search.
    
    R = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[(i, j, k)] = Fraction(1)
    
    # Generate candidate patterns on the fly to save memory?
    # Or pre-generate a subset of "good" patterns.
    # A "good" pattern has many 1s.
    
    # Let's generate all patterns where u, v, w are subsets of indices.
    # Actually, let's just use the greedy approach with random restarts and positive values.
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
            
        U_curr = [[Fraction(0) for _ in range(r)] for _ in range(nm)]
        V_curr = [[Fraction(0) for _ in range(r)] for _ in range(mp)]
        W_curr = [[Fraction(0) for _ in range(r)] for _ in range(pn)]
        
        R_curr = {key: Fraction(1) for key in R}
        
        success = True
        for c in range(r):
            if time.time() > deadline:
                return None
            
            # Find a pattern to subtract
            # We want to pick u, v, w in {0, 1} such that:
            # For all (i,j,k), u[i,j]*v[j,k]*w[k,i] <= R_curr[(i,j,k)]
            # And we want to maximize the sum of the products (or number of non-zero products).
            
            best_pattern = None
            best_score = -1
            
            # Heuristic: Try patterns that cover many remaining 1s.
            # Since the space is large, we sample random patterns.
            # But we must ensure feasibility: P <= R_curr.
            
            for _ in range(100):
                # Random pattern
                u = [1 if rng.random() < 0.5 else 0 for _ in range(nm)]
                v = [1 if rng.random() < 0.5 else 0 for _ in range(mp)]
                w = [1 if rng.random() < 0.5 else 0 for _ in range(pn)]
                
                # Check feasibility and score
                feasible = True
                score = 0
                for i in range(n):
                    for j in range(m):
                        u_val = u[i*m+j]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = v[j*p+k]
                            if v_val == 0: continue
                            w_val = w[k*n+i]
                            if w_val == 0: continue
                            
                            prod = 1 # u*v*w is 1
                            if R_curr[(i,j,k)] < prod:
                                feasible = False
                                break
                            score += 1
                        if not feasible: break
                    if not feasible: break
                
                if feasible and score > best_score:
                    best_score = score
                    best_pattern = (u, v, w)
            
            if best_pattern is None or best_score == 0:
                success = False
                break
            
            u, v, w = best_pattern
            for idx in range(nm):
                U_curr[idx][c] = Fraction(u[idx])
            for idx in range(mp):
                V_curr[idx][c] = Fraction(v[idx])
            for idx in range(pn):
                W_curr[idx][c] = Fraction(w[idx])
                
            # Update Residual
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0: continue
                    for k in range(p):
                        if v[j*p+k] == 0 or w[k*n+i] == 0: continue
                        R_curr[(i,j,k)] -= 1
                        if R_curr[(i,j,k)] < 0:
                            success = False
                            break
                    if not success: break
                if not success: break
                
            if not success:
                break
            
            # Check if done
            if all(val == 0 for val in R_curr.values()):
                if c < r - 1:
                    # Fill remaining columns with 0
                    pass
                # Verify final result
                # Convert Fractions to floats/ints for output
                U_out = [[float(x) for x in row] for row in U_curr]
                V_out = [[float(x) for x in row] for row in V_curr]
                W_out = [[float(x) for x in row] for row in W_curr]
                
                # Clean up floats to ints if possible
                for row in U_out:
                    for k in range(r):
                        if row[k] == int(row[k]):
                            row[k] = int(row[k])
                for row in V_out:
                    for k in range(r):
                        if row[k] == int(row[k]):
                            row[k] = int(row[k])
                for row in W_out:
                    for k in range(r):
                        if row[k] == int(row[k]):
                            row[k] = int(row[k])
                            
                # Final verification with integers/floats
                if _verify(U_out, V_out, W_out, n, m, p):
                    return {"U": U_out, "V": V_out, "W": W_out}
                else:
                    success = False
                    break
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 110
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try lower ranks
    # Theoretical lower bound 15.
    # Try from 15 up to schoolbook_rank - 1
    for r in range(15, schoolbook_rank):
        if time.time() > deadline:
            break
        
        rng = random.Random(r * 1234 + 5678)
        result = _try_csp_rank(r, n, m, p, deadline, rng)
        if result is not None:
            best = result
            break
            
    return best
# EVOLVE-BLOCK-END
