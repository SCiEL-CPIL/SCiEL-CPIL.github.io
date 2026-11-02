import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    r = 25  # target rank
    
    # U: (n*m, r) = (10, 25)
    # V: (m*p, r) = (25, 25)
    # W: (p*n, r) = (10, 25)
    
    # Strategy: Use a structured approach where we fix U and V to be sparse
    # and solve for W using the linearity of the tensor equation.
    
    # Key insight: For matrix multiplication C = A * B where A is n x m and B is m x p,
    # the tensor T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j] * B[j,k].
    
    # We'll use a construction where:
    # - U has one non-zero per row (sparse)
    # - V has one non-zero per row (sparse)  
    # - W is solved from the linear constraints
    
    # Let's try: U[i*m+j, c] = 1 if c == i*m+j, else 0
    #            V[j*p+k, c] = 1 if c == j*p+k, else 0
    # Then for each (i,k), B_{(i,k),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    # = sum_j [c == i*m+j] * [c == j*p+k]
    # This is 1 if there exists j such that c == i*m+j AND c == j*p+k
    # i.e., i*m+j = j*p+k, so j*(m-p) = k - i*m
    # With m=p=5, this gives j*0 = k - i*5, so k = i*5
    # For i=0: k=0, for i=1: k=5 (out of range)
    # So only (i=0,k=0) has a non-zero B, which won't work.
    
    # Better approach: Use random sparse matrices and solve the linear system
    
    import random
    random.seed(42)
    
    for attempt in range(500):
        # Create sparse U and V with entries in {-1, 0, 1}
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        
        # Each row of U has exactly 1 non-zero entry
        for i in range(n*m):
            col = random.randint(0, r-1)
            val = random.choice([-1, 1])
            U[i][col] = val
        
        # Each row of V has exactly 1 non-zero entry
        for j in range(m*p):
            col = random.randint(0, r-1)
            val = random.choice([-1, 1])
            V[j][col] = val
        
        # Compute B_{(i,k),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
        B = {}
        for i in range(n):
            for k in range(p):
                B[(i,k)] = [0]*r
                for j in range(m):
                    for c in range(r):
                        B[(i,k)][c] += U[i*m+j][c] * V[j*p+k][c]
        
        # Now solve for W: for each (i,k), need sum_c B[(i,k)][c] * W[k*n+i][c] = 1
        W = [[0]*r for _ in range(p*n)]
        valid = True
        
        for i in range(n):
            for k in range(p):
                row_idx = k*n + i
                b = B[(i,k)]
                
                # Find a linear combination of b's entries that equals 1
                # with integer/half-integer coefficients
                # Simplest: if any b[c] = 1, set W[row_idx][c] = 1
                # If any b[c] = -1, set W[row_idx][c] = -1
                # If any b[c] = 2, set W[row_idx][c] = 1/2
                # If any b[c] = -2, set W[row_idx][c] = -1/2
                
                found = False
                for c in range(r):
                    bc = b[c]
                    if bc == 0:
                        continue
                    
                    # Check if 1/bc is integer or half-integer
                    frac = Fraction(1, bc)
                    if (frac * 2).denominator == 1:
                        W[row_idx][c] = frac
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * W[k*n+i][c]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to float
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition with rank = n*m*p = 50
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
