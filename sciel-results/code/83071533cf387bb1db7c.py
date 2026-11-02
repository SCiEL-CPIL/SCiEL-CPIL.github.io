import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Strassen-like decomposition of rank 21
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # M1: (a+d)(x+w)
    # M2: (c+d)x
    # M3: a(y-x)
    # M4: d(w-z)
    # M5: (a+b)w
    # M6: (c+b)(x-y)
    # M7: (a+d)(z-w)
    
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block b (k=2b, 2b+1):
    # x: j=0, k=2b -> 2b
    # y: j=0, k=2b+1 -> 2b+1
    # z: j=1, k=2b -> 6+2b
    # w: j=1, k=2b+1 -> 6+2b+1
    # W indices for block b:
    # C00: k=2b, i=0 -> 4b
    # C01: k=2b+1, i=0 -> 4b+2
    # C10: k=2b, i=1 -> 4b+1
    # C11: k=2b+1, i=1 -> 4b+3
    
    for b in range(3):
        base_col = 7 * b
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1
        c = base_col
        U[0][c] = 1; U[3][c] = 1
        V[k0][c] = 1; V[6+k1][c] = 1
        W[4*b][c] = 1; W[4*b+2][c] = 0; W[4*b+1][c] = 0; W[4*b+3][c] = 1
        
        # M2
        c = base_col + 1
        U[2][c] = 1; U[3][c] = 1
        V[k0][c] = 1
        W[4*b+1][c] = 1; W[4*b+3][c] = -1
        
        # M3
        c = base_col + 2
        U[0][c] = 1
        V[k1][c] = 1; V[k0][c] = -1
        W[4*b+2][c] = 1; W[4*b+3][c] = 1
        
        # M4
        c = base_col + 3
        U[3][c] = 1
        V[6+k1][c] = 1; V[6+k0][c] = -1
        W[4*b][c] = 1; W[4*b+3][c] = 1
        
        # M5
        c = base_col + 4
        U[0][c] = 1; U[1][c] = 1
        V[6+k1][c] = 1
        W[4*b][c] = -1; W[4*b+2][c] = 1
        
        # M6
        c = base_col + 5
        U[1][c] = 1; U[2][c] = 1
        V[k0][c] = 1; V[k1][c] = -1
        W[4*b+1][c] = 1
        
        # M7
        c = base_col + 6
        U[0][c] = 1; U[3][c] = 1
        V[6+k0][c] = 1; V[6+k1][c] = -1
        W[4*b][c] = 1
        
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check identity: sum_r U[i,j,r] V[j,k,r] W[k,i,r] == A[i,j]B[j,k]
    # We check for all i, j, k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Calculate LHS
                lhs = 0
                for r in range(rank):
                    lhs += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                # RHS is 1 if this is the term A[i,j]B[j,k], but the tensor is sum over r
                # The tensor T[i,j,k] should be 1 for all i,j,k?
                # No, T[i,j,k] corresponds to the coefficient of A[i,j]B[j,k] in the sum.
                # The identity is: sum_r ... = A[i,j] B[j,k]
                # So for a specific i,j,k, the sum must equal 1 (since A[i,j]B[j,k] is the monomial).
                # Wait, the tensor decomposition represents the bilinear map.
                # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                # And T_{i,j,k} should be 1 for all i,j,k?
                # Yes, because C_{i,k} = sum_j A_{i,j} B_{j,k}.
                # The term A_{i,j} B_{j,k} has coefficient 1 in the sum.
                # So for each i,j,k, the sum over r must be 1.
                if abs(lhs - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
