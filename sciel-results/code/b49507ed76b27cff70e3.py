import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using integer arithmetic. Returns True if valid."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Target rank 57
    # 21 for Strassen blocks (j=0,1; k=0-1, 2-3, 4-5)
    # 36 for Schoolbook (j=2,3,4; k=0-5)
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen definitions for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> w: C00, C11
    # M2 = (A10+A11)B00 -> w: C10
    # M3 = A00(B10-B11) -> w: C01
    # M4 = A11(B01-B00) -> w: C00, C10
    # M5 = (A00+A01)B11 -> w: C00, C01
    # M6 = (A10-A00)(B00+B01) -> w: C11
    # M7 = (A01-A11)(B10+B11) -> w: C00, C11
    
    # Coefficients for U (A), V (B), W (C)
    # Indices in local 2x2 block: 00, 01, 10, 11
    # U indices: (i,j) -> i*2 + j
    # V indices: (j,k) -> j*2 + k
    # W indices: (k,i) -> k*2 + i
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [00, 01, 10, 11]
        # v: [00, 01, 10, 11]
        # w: [00, 01, 10, 11]
        
        # M1: (A00+A11)(B00+B11)
        # u: 1 at 00, 1 at 11
        # v: 1 at 00, 1 at 11
        # w: 1 at 00, 1 at 11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # M2: (A10+A11)B00
        # u: 0 at 00, 0 at 01, 1 at 10, 1 at 11
        # v: 1 at 00, 0 at 01, 0 at 10, 0 at 11
        # w: 0 at 00, 0 at 01, 1 at 10, 0 at 11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        
        # M3: A00(B10-B11)
        # u: 1 at 00
        # v: 0 at 00, 0 at 01, 1 at 10, -1 at 11
        # w: 0 at 00, 1 at 01, 0 at 10, 0 at 11
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        
        # M4: A11(B01-B00)
        # u: 0 at 00, 0 at 01, 0 at 10, 1 at 11
        # v: -1 at 00, 1 at 01, 0 at 10, 0 at 11
        # w: 1 at 00, 0 at 01, 1 at 10, 0 at 11
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        
        # M5: (A00+A01)B11
        # u: 1 at 00, 1 at 01
        # v: 0 at 00, 0 at 01, 0 at 10, 1 at 11
        # w: -1 at 00, 1 at 01, 0 at 10, 0 at 11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        
        # M6: (A10-A00)(B00+B01)
        # u: -1 at 00, 0 at 01, 1 at 10, 0 at 11
        # v: 1 at 00, 1 at 01, 0 at 10, 0 at 11
        # w: 0 at 00, 0 at 01, 0 at 10, 1 at 11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        
        # M7: (A01-A11)(B10+B11)
        # u: 0 at 00, 1 at 01, 0 at 10, -1 at 11
        # v: 0 at 00, 0 at 01, 1 at 10, 1 at 11
        # w: -1 at 00, 0 at 01, 0 at 10, -1 at 11
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]),
    ]
    
    # Apply Strassen for j in {0, 1} and k blocks {0,1}, {2,3}, {4,5}
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for k_start, k_end in k_blocks:
        for c_idx, (u_c, v_c, w_c) in enumerate(strassen_terms):
            # u_c corresponds to A[i, j] for i in {0,1}, j in {0,1}
            # Global U index: i*m + j
            # u_c is [00, 01, 10, 11] -> i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
            for local_idx, val in enumerate(u_c):
                if val != 0:
                    i = local_idx // 2
                    j = local_idx % 2
                    U[i*m + j][col] = val
                    
            # v_c corresponds to B[j, k] for j in {0,1}, k in {k_start, k_start+1}
            # v_c is [00, 01, 10, 11] -> j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
            for local_idx, val in enumerate(v_c):
                if val != 0:
                    j = local_idx // 2
                    k_local = local_idx % 2
                    k_global = k_start + k_local
                    V[j*p + k_global][col] = val
                    
            # w_c corresponds to C[i, k] for i in {0,1}, k in {k_start, k_start+1}
            # w_c is [00, 01, 10, 11] -> k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
            # Wait, W index is k*n + i.
            # local 00: k=0, i=0
            # local 01: k=0, i=1
            # local 10: k=1, i=0
            # local 11: k=1, i=1
            for local_idx, val in enumerate(w_c):
                if val != 0:
                    k_local = local_idx // 2
                    i = local_idx % 2
                    k_global = k_start + k_local
                    W[k_global*n + i][col] = val
                    
            col += 1
            
    # Apply Schoolbook for j in {2, 3, 4} and all k in {0..5}
    for j in range(2, m):
        for k in range(p):
            # U[i*m + j]
            for i in range(n):
                U[i*m + j][col] = 1
            # V[j*p + k]
            V[j*p + k][col] = 1
            # W[k*n + i]
            for i in range(n):
                W[k*n + i][col] = 1
            col += 1
            
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
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
# EVOLVE-BLOCK-END
