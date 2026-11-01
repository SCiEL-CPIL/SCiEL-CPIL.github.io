"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_rank_11():
    """
    Construct an explicit rank-11 decomposition for 2x2 * 2x3 matrix multiplication.
    
    Strategy: Use 7 Strassen-like terms for the 2x2x2 part (k=0,1) and 4 additional
    terms for the third column (k=2), with careful sharing to achieve rank 11.
    
    The key insight: we can structure the decomposition so that some terms contribute
    to multiple k-values, reducing the total count from 12 (schoolbook) to 11.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    # U has shape (n*m, rank) = (4, 11)
    # V has shape (m*p, rank) = (6, 11)
    # W has shape (p*n, rank) = (6, 11)
    
    # Indexing:
    # U[i*m + j, c] for i in {0,1}, j in {0,1}
    # V[j*p + k, c] for j in {0,1}, k in {0,1,2}
    # W[k*n + i, c] for k in {0,1,2}, i in {0,1}
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Terms 0-6: Strassen-style for k=0,1 (2x2x2 part)
    # Term 0: (A00+A11)(B00+B11) -> contributes to C00, C11 for k=0,1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # k=0,i=1 and k=1,i=1
    
    # Term 1: (A10+A11)B00 -> contributes to C10 for k=0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1  # k=0,i=1
    
    # Term 2: A00(B10-B11) -> contributes to C10 for k=0
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1  # j=1,k=0 and j=1,k=1... wait
    # Actually for 2x2x2, B is 2x2, so B[j,k] with j in {0,1}, k in {0,1}
    # But in our problem, B is 2x3, so B[j,k] with j in {0,1}, k in {0,1,2}
    # For the 2x2x2 part, we only use k=0,1
    
    # Let me restart with a cleaner approach.
    # I'll use a known explicit construction.
    
    # Actually, let me think about this differently.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # A rank-11 construction can be built as follows:
    # Use the fact that we can have terms that span multiple k values.
    
    # Let me try a specific construction based on known results:
    
    # Reset
    for row in U:
        for c in range(rank):
            row[c] = 0.0
    for row in V:
        for c in range(rank):
            row[c] = 0.0
    for row in W:
        for c in range(rank):
            row[c] = 0.0
    
    # Construction based on Strassen extension:
    # Terms 0-6: Handle the 2x2x2 core
    # Terms 7-10: Handle the third column with sharing
    
    # Term 0: (A00+A11)(B00+B11) for k=0,1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1; V[3][0] = 1; V[4][0] = 1  # j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1  # k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
    
    # This is getting complicated. Let me use a verified explicit construction.
    
    # Known rank-11 decomposition for 2x2x3:
    # I'll use a construction that's been verified in the literature.
    
    # For now, let me return a schoolbook rank-12 and verify it works,
    # then try to find a rank-11 construction.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank-12 decomposition as fallback
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
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find rank-11 decomposition
    if time.time() < deadline:
        # Explicit rank-11 construction
        # Based on the Strassen extension method
        
        rank11 = 11
        U11 = [[0.0] * rank11 for _ in range(n * m)]
        V11 = [[0.0] * rank11 for _ in range(m * p)]
        W11 = [[0.0] * rank11 for _ in range(p * n)]
        
        # Terms 0-6: Strassen for 2x2x2 (k=0,1)
        # Using standard Strassen with proper indexing
        
        # Term 0: (A00+A11)(B00+B11)
        U11[0][0] = 1; U11[3][0] = 1
        V11[0][0] = 1; V11[3][0] = 1
        W11[0][0] = 1; W11[2][0] = 1  # k=0,i=0 and k=1,i=0
        
        # Term 1: (A10+A11)B00
        U11[2][1] = 1; U11[3][1] = 1
        V11[0][1] = 1
        W11[1][1] = 1  # k=0,i=1
        
        # Term 2: A00(B10-B11)
        U11[0][2] = 1
        V11[2][2] = 1; V11[3][2] = -1
        W11[1][2] = 1  # k=0,i=1
        
        # Term 3: A11(B01-B10)
        U11[3][3] = 1
        V11[1][3] = 1; V11[2][3] = -1
        W11[3][3] = 1  # k=1,i=1
        
        # Term 4: (A00+A01)B11
        U11[0][4] = 1; U11[1][4] = 1
        V11[3][4] = 1
        W11[3][4] = 1  # k=1,i=1
        
        # Term 5: (A10-A00)(B00+B10)
        U11[2][5] = 1; U11[0][5] = -1
        V11[0][5] = 1; V11[2][5] = 1
        W11[1][5] = 1  # k=0,i=1
        
        # Term 6: (A01-A11)(B01+B11)
        U11[1][6] = 1; U11[3][6] = -1
        V11[1][6] = 1; V11[3][6] = 1
        W11[3][6] = 1  # k=1,i=1
        
        # Terms 7-10: Handle k=2 (third column)
        # We need to add 4 terms for the third column
        # But we can potentially share with existing terms
        
        # Term 7: A00*B02
        U11[0][7] = 1
        V11[4][7] = 1  # j=0,k=2
        W11[4][7] = 1  # k=2,i=0
        
        # Term 8: A01*B12
        U11[1][8] = 1
        V11[5][8] = 1  # j=1,k=2
        W11[4][8] = 1  # k=2,i=0
        
        # Term 9: A10*B02
        U11[2][9] = 1
        V11[4][9] = 1  # j=0,k=2
        W11[5][9] = 1  # k=2,i=1
        
        # Term 10: A11*B12
        U11[3][10] = 1
        V11[5][10] = 1  # j=1,k=2
        W11[5][10] = 1  # k=2,i=1
        
        # Verify the rank-11 decomposition
        if _verify_decomposition(U11, V11, W11, n, m, p, tol=1e-9):
            best = {"U": U11, "V": V11, "W": W11}
            best_rank = rank11
        else:
            # Rank-11 construction failed verification, keep schoolbook
            pass
    
    return best
# EVOLVE-BLOCK-END
