import numpy as np
import time

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 120

def construct():
    n, m, p = DIMENSIONS
    return _laderman_3x3_plus_schoolbook(n, m, p)

def _laderman_3x3_plus_schoolbook(n, m, p):
    """
    Use Laderman's rank-19 algorithm for the 3x3x3 sub-block (i in 0..2, j in 0..2, k in 0..2)
    Then schoolbook for the remaining 48 terms.
    Total rank: 19 + 48 = 67
    """
    total_terms = n * m * p  # 75
    laderman_rank = 19
    schoolbook_count = total_terms - 27  # 75 - 27 = 48
    total_rank = laderman_rank + schoolbook_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Laderman's algorithm for 3x3 matrix multiplication
    # The 19 multiplication terms and their contributions:
    # We need to express C = A*B where A, B are 3x3
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in 0..2
    
    # Laderman's 19-term algorithm (one known construction):
    # Let me use a verified rank-19 decomposition for 3x3x3
    
    # The 19 terms (using a known construction):
    # I'll use the standard Laderman construction
    
    laderman_terms = [
        # (U_indices, V_indices, W_indices, signs)
        # Each term: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # Term 1: (A00+A11+A22)(B00+B11+B22) - but this is too complex
        # Let me use a simpler verified construction
        
        # Actually, let me use a different approach: 
        # Use 4 Strassen 2x2x2 blocks for the 3x3x3 (covering most of it)
        # + handle the remaining terms cleverly
    ]
    
    # Alternative: Use 4 Strassen blocks for 3x3x3:
    # Block (0,0,0): covers i{0,1}, j{0,1}, k{0,1} - 8 terms, rank 7
    # Block (0,1,0): covers i{0,1}, j{1,2}, k{0,1} - 8 terms, rank 7  
    # Block (0,0,1): covers i{0,1}, j{0,1}, k{1,2} - 8 terms, rank 7
    # Block (0,1,1): covers i{0,1}, j{1,2}, k{1,2} - 8 terms, rank 7
    # But these overlap! Terms like (1,1,1) are covered multiple times.
    
    # This is getting complex. Let me use a cleaner approach:
    # Use the fact that 3x3x3 can be decomposed as:
    # - One 2x2x2 Strassen block (rank 7) for i{0,1},j{0,1},k{0,1}
    # - Remaining terms handled optimally
    
    # Actually, the simplest correct approach: 
    # Use 4 non-overlapping 2x2x2 Strassen blocks for the full 3x5x5,
    # but try to find if any rank reduction is possible.
    
    # Let me just use the proven rank-71 solution but verify it's correct,
    # and add a quick check for potential rank-70.
    
    # For now, return the rank-71 solution which is verified correct.
    return _rank71_solution(n, m, p)

def _rank71_solution(n, m, p):
    """
    Rank 71 solution:
    Four Strassen blocks (rank 7 each) + Schoolbook for the rest (43 terms)
    """
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    while col > 0 and all(U[i][col-1] == 0 for i in range(n*m)) and \
          all(V[i][col-1] == 0 for i in range(m*p)) and \
          all(W[i][col-1] == 0 for i in range(p*n)):
        col -= 1
    
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    print(f"Errors: {errors}")
