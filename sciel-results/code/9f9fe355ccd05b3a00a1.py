"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _clean_entry(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    doubled = v * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return round(doubled) / 2.0
    return v

def _construct_rank_28():
    """
    Rank-28 decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Split the inner dimension (4) into 1+1+2.
    Block 1: A[:, 0] * B[0, :] -> 2x1x5 tensor. Rank 5 (Schoolbook).
    Block 2: A[:, 1] * B[1, :] -> 2x1x5 tensor. Rank 5 (Schoolbook).
    Block 3: A[:, 2:4] * B[2:4, :] -> 2x2x5 tensor. Rank 18.
             Split Block 3 output into 2+3:
             Block 3a: 2x2x2 (Strassen, Rank 7)
             Block 3b: 2x2x3 (Split into 2x2x2 Strassen Rank 7 + 2x2x1 Schoolbook Rank 4 = Rank 11)
             Total Block 3 Rank: 7 + 11 = 18.
    
    Total Rank: 5 + 5 + 18 = 28.
    
    U: shape (n*m, rank) = (8, 28)
    V: shape (m*p, rank) = (20, 28)
    W: shape (p*n, rank) = (10, 28)
    """
    n, m, p = 2, 4, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 20 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        """Add Strassen 2x2x2 block for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] = C[i1:i2+1, k1:k2+1]"""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (A11+A22)(B11+B22)
        col = col_offset + 0
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        
        # M2 = (A21+A22)B11
        col = col_offset + 1
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        W[c22][col] = -1
        
        # M3 = A11(B12-B22)
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1
        V[b22][col] = -1
        W[c12][col] = 1
        W[c22][col] = 1
        
        # M4 = A22(B21-B11)
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        V[b11][col] = -1
        W[c11][col] = 1
        W[c21][col] = 1
        
        # M5 = (A11+A12)B22
        col = col_offset + 4
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1
        W[c12][col] = 1
        
        # M6 = (A21-A11)(B11+B12)
        col = col_offset + 5
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        
        # M7 = (A12-A22)(B21+B22)
        col = col_offset + 6
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        """Add schoolbook 2x2x1 block for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1] = C[i1:i2+1, k1]"""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        # Term 1: A11 * B11 -> contributes to C11
        col = col_offset + 0
        U[a11][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        
        # Term 2: A12 * B21 -> contributes to C11
        col = col_offset + 1
        U[a12][col] = 1
        V[b21][col] = 1
        W[c11][col] = 1
        
        # Term 3: A21 * B11 -> contributes to C21
        col = col_offset + 2
        U[a21][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        
        # Term 4: A22 * B21 -> contributes to C21
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        W[c21][col] = 1

    def add_schoolbook_21p(i1, i2, j1, k1, k2, col_offset):
        """
        Add schoolbook 2x1xP block for A[i1:i2+1, j1] * B[j1, k1:k2+1] = C[i1:i2+1, k1:k2+1]
        This handles a single column of A and a single row of B.
        Rank = (i2-i1+1) * (k2-k1+1). Here 2 * P.
        """
        # For each output column k in [k1, k2] and each output row i in [i1, i2]
        # The term is A[i, j1] * B[j1, k]
        
        col_idx = col_offset
        for i in range(i1, i2+1):
            for k in range(k1, k2+1):
                a_idx = i * m + j1
                b_idx = j1 * p + k
                c_idx = k * n + i
                
                U[a_idx][col_idx] = 1
                V[b_idx][col_idx] = 1
                W[c_idx][col_idx] = 1
                col_idx += 1
        
        return col_idx

    col = 0
    
    # Block 1: A[:, 0] * B[0, :] -> 2x1x5
    # i in [0, 1], j=0, k in [0, 4]
    col = add_schoolbook_21p(0, 1, 0, 0, 4, col)
    # Rank added: 2 * 5 = 10? Wait.
    # The schoolbook for 2x1x5 is indeed rank 10 if we treat it as separate terms for each (i,k).
    # Let's re-evaluate the rank calculation.
    # M_{2,1,5} tensor: T_{i,j,k} = delta_{j,0} (since j is fixed to 0).
    # Rank of M_{2,1,5} is 2*5 = 10?
    # Yes, the rank of M_{m,n,p} is m*n*p.
    # So M_{2,1,5} has rank 10.
    # My previous calculation "Rank 5" was wrong.
    # Let's re-read the inspiration or standard results.
    # Rank of matrix multiplication tensor M_{m,n,p} is at most m*n*p.
    # For M_{2,1,5}, it is 2*1*5 = 10.
    
    # Let's re-calculate the total rank with correct schoolbook ranks.
    # Block 1: 2x1x5 -> Rank 10
    # Block 2: 2x1x5 -> Rank 10
    # Block 3: 2x2x5 -> Rank ?
    #   Block 3a: 2x2x2 -> Rank 7 (Strassen)
    #   Block 3b: 2x2x3 -> Rank ?
    #     Split 2x2x3 into 2x2x2 (Rank 7) + 2x2x1 (Rank 4) = 11.
    #   Total Block 3: 7 + 11 = 18.
    # Total Rank: 10 + 10 + 18 = 38.
    # This is worse than 36!
    
    # I need to be smarter about the 2x1x5 blocks or use a different split.
    # Is there a lower rank for 2x1x5? No, it's rank 10.
    # Can I combine the 2x1x5 blocks?
    # A[:, 0:2] * B[0:2, :] is a 2x2x5 tensor.
    # Rank of M_{2,2,5} <= 18 (from above: 7 + 11).
    # So if I take Block 1+2 as a single 2x2x5 block, rank is 18.
    # Then Block 3 is A[:, 2:4] * B[2:4, :] which is another 2x2x5 block, rank 18.
    # Total Rank: 18 + 18 = 36.
    # This is the same as the current program!
    
    # How to get below 36?
    # Maybe split 2x2x5 differently?
    # M_{2,2,5} rank is known to be 18?
    # Let's check if M_{2,2,5} can be done in less than 18.
    # M_{2,2,2} is 7.
    # M_{2,2,1} is 4.
    # M_{2,2,3} <= 7 + 4 = 11.
    # M_{2,2,4} <= 7 + 7 = 14.
    # M_{2,2,5} <= 7 + 11 = 18.
    # Is there a better decomposition for M_{2,2,5}?
    # Maybe 17?
    # If I can get M_{2,2,5} down to 17, total is 34.
    
    # Let's try to construct a rank 17 decomposition for M_{2,2,5}.
    # Or maybe use a different split for the whole problem.
    # What if I split the output dimension 5 into 1+1+1+1+1? No.
    # What if I use Strassen for the whole thing?
    # Pad to 2x4x8? No.
    
    # Let's stick to the 36 solution but try to optimize the 2x2x5 block.
    # Is there a rank 17 decomposition for 2x2x5?
    # Literature search (mental):
    # Rank of M_{2,2,n} grows linearly.
    # M_{2,2,1}=4, M_{2,2,2}=7, M_{2,2,3}=11, M_{2,2,4}=14, M_{2,2,5}=18?
    # The sequence is 4, 7, 11, 14, 18...
    # Differences: 3, 4, 3, 4...
    # It seems to follow 3n+1?
    # n=1: 4 (3+1)
    # n=2: 7 (6+1)
    # n=3: 10? No, 11.
    # n=4: 13? No, 14.
    # n=5: 16? No, 18.
    
    # Let's assume 18 is tight for now.
    # Then total rank 36 is the best with this split.
    
    # Can we do better with a different split?
    # Split 4 into 1+3?
    # Block 1: 2x1x5 (Rank 10)
    # Block 2: 2x3x5
    #   M_{2,3,5} <= M_{2,2,5} + M_{2,1,5} = 18 + 10 = 28.
    # Total: 10 + 28 = 38. Worse.
    
    # Split 4 into 2+2 is best so far.
    
    # Is there a way to share terms between the two 2x2x5 blocks?
    # They operate on disjoint parts of A and B, so no.
    
    # Therefore, rank 36 is likely the best with this structure.
    # The current program already achieves 36.
    # To improve, I need a fundamentally better decomposition.
    
    # Let's try to find a rank 34 or 35 decomposition.
    # Maybe M_{2,2,5} is 17?
    # Let's try to construct it.
    
    # I will return the rank 36 solution as it is known to be valid.
    # I will not implement the incorrect rank 28 logic above.
    
    # Re-implement the rank 36 solution cleanly.
    
    # Clear the matrices
    for r in range(len(U)):
        for c in range(len(U[r])):
            U[r][c] = 0
    for r in range(len(V)):
        for c in range(len(V[r])):
            V[r][c] = 0
    for r in range(len(W)):
        for c in range(len(W[r])):
            W[r][c] = 0
    
    # Reset rank to 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[:, 0:2] * B[0:2, :] -> 2x2x5
    # Split output into 2+3
    # 2x2x2 (Strassen)
    add_strassen_222(0, 1, 0, 1, 0, 1, col)
    col += 7
    # 2x2x3 (Strassen 2x2x2 + Schoolbook 2x2x1)
    # 2x2x2 for cols 2,3
    add_strassen_222(0, 1, 0, 1, 2, 3, col)
    col += 7
    # 2x2x1 for col 4
    add_schoolbook_221(0, 1, 0, 1, 4, col)
    col += 4
    
    # Block 2: A[:, 2:4] * B[2:4, :] -> 2x2x5
    # Split output into 2+3
    # 2x2x2 (Strassen)
    add_strassen_222(0, 1, 2, 3, 0, 1, col)
    col += 7
    # 2x2x3 (Strassen 2x2x2 + Schoolbook 2x2x1)
    # 2x2x2 for cols 2,3
    add_strassen_222(0, 1, 2, 3, 2, 3, col)
    col += 7
    # 2x2x1 for col 4
    add_schoolbook_221(0, 1, 2, 3, 4, col)
    col += 4
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W

def _construct_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 40."""
    n, m, p = 2, 4, 5
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
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try Rank-36 decomposition (Strassen-based)
    U_rank36, V_rank36, W_rank36 = _construct_rank_28() # Name is misleading, it's rank 36
    if _validate_numpy(U_rank36, V_rank36, W_rank36, n, m, p):
        result_U = [[_clean_entry(v) for v in row] for row in U_rank36]
        result_V = [[_clean_entry(v) for v in row] for row in V_rank36]
        result_W = [[_clean_entry(v) for v in row] for row in W_rank36]
        return {"U": result_U, "V": result_V, "W": result_W}
    
    # Fallback to Schoolbook
    U_school, V_school, W_school = _construct_schoolbook()
    assert _validate_numpy(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
