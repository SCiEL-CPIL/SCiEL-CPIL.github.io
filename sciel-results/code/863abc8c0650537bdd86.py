"""Corner-peeling recursive reduction for 3x5 * 5x6 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 5, 6]

def _peel(i, j, U_rows, V_rows, W_rows, U, V, W, col):
    """Peel output element (i,j) which equals A[i][0]*B[0][j].
    U row index: i*m + 0
    V row index: 0*p + j
    W row index: j*n + i
    Returns new column index."""
    U[U_rows[i]][col] = 1
    V[V_rows[j]][col] = 1
    W[W_rows[j * 3 + i]][col] = 1
    return col + 1

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = 3, 5, 6
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p  # 90
    
    # Corner peeling: peel output elements (0,0), (0,1), ..., (0,4)
    # Each peel uses 1 rank unit
    # After peeling row 0 outputs, remaining is 2x5 * 5x6 minus linear terms
    # Actually, let's think more carefully.
    
    # Output C[i][j] = sum_k A[i][k] * B[k][j]
    # Peel C[0][j] for j=0..4: C[0][j] = A[0][0]*B[0][j] + sum_{k=1}^{4} A[0][k]*B[k][j]
    # The term A[0][0]*B[0][j] is rank 1 per j.
    # After peeling these 5 terms, we still need sum_{k=1}^{4} A[0][k]*B[k][j] for j=0..4
    # and all C[i][j] for i=1,2.
    
    # Let's just use schoolbook for the remaining 2x5*5x6 part (rank 60)
    # Total: 5 + 60 = 65
    
    # Actually, let's try a more systematic approach.
    # Peel all outputs in row 0: 5 rank units
    # Remaining: outputs (1,0..4) and (2,0..4) = 2x5 outputs
    # Each C[i][j] for i in {1,2}, j in {0..4} = sum_{k=0}^{4} A[i][k]*B[k][j]
    # This is a 2x5 * 5x6 multiplication (with the full 5x5 B matrix)
    # Schoolbook rank for that: 2*5*6 = 60
    # Total: 5 + 60 = 65
    
    # Let's construct this:
    rank = 5 + 60  # 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Peel row 0 outputs
    for j in range(5):
        U[0 * 5 + 0][col] = 1   # A[0][0]
        V[0 * 6 + j][col] = 1   # B[0][j]
        W[j * 3 + 0][col] = 1   # C[0][j]
        col += 1
    
    # Remaining: 2x5 * 5x6 schoolbook
    for i in range(1, 3):
        for j in range(5):
            for k in range(6):
                U[i * 5 + k][col] = 1   # A[i][k]
                V[k * 6 + j][col] = 1   # B[k][j]
                W[j * 3 + i][col] = 1   # C[i][j]
                col += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == delta condition
        # The tensor T[i,k,j] = 1 if this is a valid multiplication term
        # Actually the condition is: sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] 
        # should equal 1 for all (i,k,j) and 0 otherwise? No.
        # The decomposition means: C[i][j] = sum_k A[i][k]*B[k][j] = sum_r (sum_k U[i*m+k,r]*A[i][k])*(sum_k V[k*p+j,r]*B[k][j])
        # Wait, that's not right either. The standard form is:
        # C[i][j] = sum_r U[i*m+?, r] ... 
        # Let me re-read the convention.
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The tensor decomposition: T_{i,k,j} = sum_r U_{i*m+k, r} * V_{k*p+j, r} * W_{j*n+i, r}
        # where T_{i,k,j} = 1 (the multiplication tensor is all ones for the relevant indices)
        # Actually for matrix multiplication, T_{i,k,j} = 1 for all valid (i,k,j)
        # So we need: sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] = 1 for all i,k,j
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    if s != 1:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        best_rank = rank
        best = {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        best = schoolbook()
        best_rank = n * m * p
    
    if time.time() > deadline - 10:
        return best
    
    # Try more aggressive peeling: peel row 0 AND column 0 of remaining
    # After peeling row 0 (5 terms), peel column 0 of the remaining 2x5*5x6
    # C[1][0] = A[1][0]*B[0][0] + sum_{k=1}^{4} A[1][k]*B[k][0]
    # C[2][0] = A[2][0]*B[0][0] + sum_{k=1}^{4} A[2][k]*B[k][0]
    # Peel A[1][0]*B[0][0] and A[2][0]*B[0][0]: 2 more rank units
    # Remaining: 2x4 * 5x5 (since we've used B[0][0] and A[?][0] for col 0)
    # Actually this gets complicated. Let's stick with 65.
    
    # Try: peel row 0 (5) + peel column 0 for rows 1,2 (2) = 7 peeled
    # Remaining: outputs (1,1..4) and (2,1..4) = 2x4 outputs
    # Each involves sum over k of A[i][k]*B[k][j] but we've already accounted for some terms
    # This is getting complex. Let's just return what we have.
    
    return best

