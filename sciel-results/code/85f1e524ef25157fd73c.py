import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s kill

    # We will use 4 Strassen blocks of size 2x2x2
    # Covering i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Blocks:
    # 1. i0=0, j0=0, k0=0 -> covers i:0-1, j:0-1, k:0-1
    # 2. i0=0, j0=2, k0=0 -> covers i:0-1, j:2-3, k:0-1
    # 3. i0=0, j0=0, k0=2 -> covers i:0-1, j:0-1, k:2-3
    # 4. i0=0, j0=2, k0=2 -> covers i:0-1, j:2-3, k:2-3
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    # Terms covered by Strassen are exactly 4 * 8 = 32
    # However, we must be careful not to double count or miss.
    # The blocks are disjoint in index space.
    # Covered set C = { (i,j,k) | exists (i0,j0,k0) in blocks s.t. i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1] }
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    schoolbook_terms_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    schoolbook_terms_count += 1
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i0, j0, k0, col_offset):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
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

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    # Check time
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
    
    # Now attempt to reduce rank by finding linear dependencies
    # The tensor T is decomposed as sum of rank-1 terms.
    # If there is a linear relation among the rank-1 terms, we can reduce the rank.
    # Specifically, if sum_{t} c_t * term_t = 0, then we can eliminate one term.
    # This is equivalent to finding a null vector in the matrix of coefficients.
    
    # Build the coefficient matrix M where M[r, t] is the coefficient of the r-th basis element in the t-th rank-1 term.
    # The basis elements are the (i,j,k) triples.
    # Actually, the rank-1 terms are vectors in R^{n*m*p}.
    # Let's form a matrix A of size (n*m*p) x total_rank, where A[i,j,k, t] is the coefficient of the t-th term in the (i,j,k) position.
    # Then the tensor is A @ 1 (vector of ones).
    # If there is a null vector c in A, then A @ c = 0, which means sum_t c_t * term_t = 0.
    # We can then remove one term and adjust the others.
    
    # But wait, the tensor is fixed. The decomposition is not unique.
    # The rank is the minimum number of rank-1 terms needed.
    # To reduce the rank, we need to find a different decomposition with fewer terms.
    # This is a hard problem.
    
    # For now, let's just return the current decomposition.
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
