"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline, 20s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A is n x m = 2x2, B is m x p = 2x10, C is n x p = 2x10
    # Output W has shape (p*n, rank) = (20, rank), indexed as W[k*n + i, r]
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (20, rank)

    # We can decompose the 2x10 multiplication into 5 independent 2x2 blocks.
    # Each 2x2 block multiplication can use Strassen's algorithm (rank 7) instead of schoolbook (rank 4).
    # But wait: for 2x2, schoolbook is rank 4 and Strassen is rank 7, so Strassen is WORSE for 2x2.
    # The strategy mentioned in the directive says: "Apply the rank-7 Strassen decomposition to the first two column pairs
    # of the 2x10 matrix to handle 4 output columns, and use the standard rank-4 schoolbook decomposition for the remaining 6 output columns"
    # That gives 7*2 + 4*6 = 14 + 24 = 38. But wait, that's worse than all schoolbook (4*10=40)? No, 38 < 40.
    # Actually wait: the directive says rank of 2x2 by 2x1 is 4 (schoolbook) which is lower than Strassen for single columns.
    # Let me re-read: "rectangular matrix multiplication (2x2 by 2x1) has a trivial rank of 4 which is lower than the scaled Strassen rank for single"
    # So for a 2x2 * 2x1 (single column), rank is 4 (schoolbook), and Strassen would be 7, so schoolbook is better.
    # For 2x2 * 2x2 (two columns), we can use Strassen with rank 7 instead of 2*4=8 schoolbook.
    # So grouping pairs of columns: 5 pairs, each using Strassen rank 7 gives 5*7 = 35.
    # The directive's strategy of 7*2 + 4*6 = 38 is suboptimal compared to 5*7 = 35.
    # Let me verify: 10 columns = 5 pairs of 2 columns each. Each pair uses Strassen rank 7. Total = 35.
    
    # Let's implement 5 Strassen blocks, each handling 2 output columns.
    # For block b (b=0..4), columns are k=2b and k=2b+1.
    # A indices: (i,j) where i=0,1 and j=0,1 -> U row = i*m+j = i*2+j
    # B indices: (j,k) where j=0,1 and k=2b,2b+1 -> V row = j*p+k = j*10+k
    # C indices: (i,k) where i=0,1 and k=2b,2b+1 -> W row = k*n+i = k*2+i

    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00+A11)(B0k1+B1k2)  -- but we need to be careful with indexing
    # Let's denote A = [[A00, A01],[A10, A11]], B = [[B0k1, B0k2],[B1k1, B1k2]]
    # C = [[C0k1, C0k2],[C1k1, C1k2]]
    
    # Standard Strassen:
    # M1 = (A00+A11)(B0k1+B1k2)
    # M2 = (A10+A11)B0k1
    # M3 = A00(B1k1+B1k2)
    # M4 = A11(B0k2-B1k1)  -- wait, let me use the standard formulation
    
    # Actually, standard Strassen for C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10+B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)(B01+B11)
    # M6 = (A11-A10)(B10+B11)
    # M7 = (A01-A11)(B00+B10)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping for our case:
    # A is 2x2: A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B for block b: B0k1 = B[0,2b], B0k2 = B[0,2b+1], B1k1 = B[1,2b], B1k2 = B[1,2b+1]
    # C for block b: C0k1 = C[0,2b], C0k2 = C[0,2b+1], C1k1 = C[1,2b], C1k2 = C[1,2b+1]
    
    # U row for A[i,j] = i*2+j
    # V row for B[j,k] = j*10+k
    # W row for C[i,k] = k*2+i
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # For each block b, we assign 7 columns (rank indices) to the 7 Strassen products.
    for b in range(5):
        k1 = 2 * b
        k2 = 2 * b + 1
        base_col = b * 7
        
        # M1 = (A00+A11)(B0k1+B1k2)
        col = base_col
        U[0][col] = 1; U[3][col] = 1  # A00, A11
        V[k1][col] = 1; V[10+k2][col] = 1  # B[0,k1], B[1,k2]
        # C00 = M1 - M4 + M5 - M7, C01 = M3 + M7, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
        # W for C[i,k]: W[k*2+i]
        W[k1*2+0][col] = 1   # C[0,k1] += M1
        W[k2*2+1][col] = 1   # C[1,k2] += M1
        col += 1
        
        # M2 = (A10+A11)B0k1
        U[2][col] = 1; U[3][col] = 1  # A10, A11
        V[k1][col] = 1  # B[0,k1]
        W[k2*2+0][col] = 1   # C[0,k2] += M2? Wait, let me recheck.
        # C10 = M2 + M4, so C[1,k1] += M2. W[k1*2+1]
        # Let me redo the W assignments carefully.
        # C0k1 (i=0,k=k1): W[k1*2+0]
        # C0k2 (i=0,k=k2): W[k2*2+0]
        # C1k1 (i=1,k=k1): W[k1*2+1]
        # C1k2 (i=1,k=k2): W[k2*2+1]
        # C00 = M1 - M4 + M5 - M7 -> C[0,k1]
        # C01 = M3 + M7 -> C[0,k2]
        # C10 = M2 + M4 -> C[1,k1]
        # C11 = M1 - M2 + M3 + M6 -> C[1,k2]
        
        # Redo M2:
        # C10 = M2 + M4, so W[k1*2+1] += M2
        # I made an error above. Let me redo all.
        
        # Let me redo this block from scratch.
        # Reset col to base_col
        col = base_col
        
        # M1 = (A00+A11)(B0k1+B1k2)
        U[0][col] = 1; U[3][col] = 1
        V[k1][col] = 1; V[10+k2][col] = 1
        # M1 contributes to C00 (+1) and C11 (+1)
        W[k1*2+0][col] = 1  # C[0,k1]
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
        
        # M2 = (A10+A11)B0k1
        U[2][col] = 1; U[3][col] = 1
        V[k1][col] = 1
        # M2 contributes to C10 (+1)
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
        
        # M3 = A00(B1k1+B1k2)
        U[0][col] = 1
        V[10+k1][col] = 1; V[10+k2][col] = 1
        # M3 contributes to C01 (+1) and C11 (+1)
        W[k2*2+0][col] = 1  # C[0,k2]
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
        
        # M4 = A11(B0k2-B1k1)
        U[3][col] = 1
        V[10+k1][col] = -1; V[k2][col] = 1
        # M4 contributes to C00 (-1) and C10 (+1)
        W[k1*2+0][col] = -1  # C[0,k1]
        W[k1*2+1][col] = 1   # C[1,k1]
        col += 1
        
        # M5 = (A00+A10)(B0k2+B1k2)
        U[0][col] = 1; U[2][col] = 1
        V[k2][col] = 1; V[10+k2][col] = 1
        # M5 contributes to C00 (+1)
        W[k1*2+0][col] = 1  # C[0,k1]
        col += 1
        
        # M6 = (A11-A10)(B1k1+B1k2)
        U[3][col] = 1; U[2][col] = -1
        V[10+k1][col] = 1; V[10+k2][col] = 1
        # M6 contributes to C11 (+1)
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
        
        # M7 = (A01-A11)(B0k1+B1k1)
        U[1][col] = 1; U[3][col] = -1
        V[k1][col] = 1; V[10+k1][col] = 1
        # M7 contributes to C00 (-1) and C01 (+1)
        W[k1*2+0][col] = -1  # C[0,k1]
        W[k2*2+0][col] = 1   # C[0,k2]
        col += 1
        
        # Check time
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
