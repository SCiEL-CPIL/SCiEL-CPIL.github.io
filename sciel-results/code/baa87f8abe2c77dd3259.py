import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    r = 25  # target rank
    
    # We need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Strategy: fix U and V, solve for W linearly
    # For each (i,k), we have:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
    #
    # So for each (i,k), define A_{(i,k),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    # Then we need: A_{(i,k),c} * W[k*n+i, c] summed over c = 1
    #
    # W has 10 rows: row index is k*n+i = 5k+i, for k in 0..4, i in 0..1
    # So W[5k+i, c] for c in 0..24
    #
    # The constraint for (i,k) involves W[5k+i, c] for all c
    # So we have 10 equations (for 10 (i,k) pairs) with 25*10 = 250 unknowns? No.
    # W[5k+i, c] - for each (i,k) we have 25 values of c, but W is shared.
    # Actually W has 10 rows, each with 25 columns. So 250 unknowns total for W.
    # But each constraint (i,k) only involves row 5k+i of W.
    # So constraint (i,k) is: dot(B_{(i,k)}, W[5k+i, :]) = 1
    # This is 10 independent linear equations, each involving 25 unknowns (one row of W).
    # Each equation has a unique row of W, so they're all independent!
    # For each (i,k), we need to find W[5k+i, :] such that dot(B_{(i,k)}, W[5k+i, :]) = 1
    # This is always solvable if B_{(i,k)} is not the zero vector.
    
    # So the algorithm:
    # 1. Randomly initialize U (10x25) and V (25x25) with small integers
    # 2. For each (i,k), compute B_{(i,k),c} = sum_j U[5i+j, c] * V[5j+k, c]
    # 3. For each (i,k), find W[5k+i, :] such that dot(B_{(i,k)}, W[5k+i, :]) = 1
    #    Simplest: if any B_{(i,k),c} != 0, set W[5k+i, c0] = 1/B_{(i,k),c0} for some c0 where nonzero
    #    But we want integer/half-integer entries, so this might not work directly.
    
    # Let me try: set W[5k+i, c] = 1 for all c, then check if dot(B, W_row) = sum(B) = 1
    # That's unlikely.
    
    # Better: for each (i,k), pick a c where B_{(i,k),c} is nonzero, set W[5k+i, c] = 1/B_{(i,k),c}
    # But 1/B might not be integer/half-integer.
    
    # Alternative: make B_{(i,k),c} = 1 for some c, then W = 1 works.
    # Can we choose U and V such that for each (i,k), there exists c with sum_j U[5i+j,c]*V[5j+k,c] = 1?
    
    # This is getting complicated. Let me try a different approach: use a known construction.
    
    # I'll try random search with small integers and check.
    
    import random
    random.seed(42)
    
    for attempt in range(1000):
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
        
        # Compute B_{(i,k),c}
        B = {}
        for i in range(n):
            for k in range(p):
                B[(i,k)] = [0]*r
                for j in range(m):
                    for c in range(r):
                        B[(i,k)][c] += U[i*m+j][c] * V[j*p+k][c]
        
        # Check if we can solve for W with integer/half-integer entries
        W = [[0]*r for _ in range(p*n)]
        valid = True
        for i in range(n):
            for k in range(p):
                row_idx = k*n + i
                b = B[(i,k)]
                # Find c where b[c] != 0
                nonzero_cs = [c for c in range(r) if b[c] != 0]
                if not nonzero_cs:
                    valid = False
                    break
                # Try to find a solution with small rational entries
                # Simplest: W[row_idx, c0] = 1/b[c0] for some c0
                # Check if 1/b[c0] is integer or half-integer
                found = False
                for c0 in nonzero_cs:
                    val = Fraction(1, b[c0])
                    # Check if val is integer or half-integer: val*2 is integer
                    if (val * 2).denominator == 1:
                        W[row_idx][c0] = val
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify the full decomposition
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                # Convert fractions to int or half-integer representation
                # The problem says integer/half-integer entries
                # We can return as fractions or as float if they're .0 or .5
                U_out = [[float(x) for x in row] for row in U]
                V_out = [[float(x) for x in row] for row in V]
                W_out = [[float(x) for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
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
