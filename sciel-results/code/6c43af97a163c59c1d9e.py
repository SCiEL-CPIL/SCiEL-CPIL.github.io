"""Matrix multiplication 4,4,9: tensor decomposition with rank minimization."""
DIMENSIONS = [4, 4, 9]

import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Pre-compute the multiplication tensor T: T[i,j,k] = 1 if this is a valid multiplication path
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # T[i, j, k] = 1 if i (row of A), j (col of A = row of B), k (col of B)
    # Output: C[i,k] += A[i,j] * B[j,k]
    # So the tensor has 1 at (i,j,k) for all i in [0,n), j in [0,m), k in [0,p)
    # But wait, the task says "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k*n + i) where k is the column of C, i is the row of C
    
    # The tensor M_{i,j,k} = 1 for all valid (i,j,k) in [0,n) x [0,m) x [0,p)
    # A decomposition of rank R means:
    # sum_{r=0}^{R-1} U_{(i*m+j), r} * V_{(j*p+k), r} * W_{(k*n+i), r} = 1 if valid, 0 otherwise
    # Since all (i,j,k) are valid, the sum must be 1 for all (i,j,k)
    # And for "invalid" entries (which don't exist in this tensor), the sum must be 0
    
    # Wait, for matrix multiplication, the tensor is "dense" in the sense that every (i,j,k) tuple 
    # corresponds to a non-zero entry. The "zero" entries are those where the indices don't align.
    # Actually, for matrix multiplication, the tensor is defined on the full grid, and the "valid" 
    # entries are those where the middle index matches. Let me re-read the verify function.
    
    # From the verify function:
    # For a in range(n*m), b in range(m*p), c in range(p*n):
    #   i = a // m, j = a % m
    #   j2 = b // p, k = b % p
    #   k2 = c // n, i2 = c % n
    #   Valid if j == j2 and i == i2, in which case sum should be 1
    #   Otherwise sum should be 0
    
    # So the tensor is:
    # T[a,b,c] = 1 if (a%m == b//p) and (c%n == a//m), else 0
    # where a = i*m + j, b = j*p + k, c = k*n + i
    
    # This is the standard matrix multiplication tensor.
    
    # Schoolbook rank: n*m*p = 4*4*9 = 144
    
    # Strassen for 4x4x4 gives rank 56, plus 4x4x4 for another 56, plus 4x4x1 for 16
    # Total: 128
    
    # Let me try to find a better decomposition using a search approach.
    
    # Strategy: Use the known rank-56 Strassen for 4x4x4, and try to optimize the remaining part.
    # The 4x4x9 can be seen as 4x4x4 + 4x4x4 + 4x4x1
    
    # Alternative: Try to use a rank-7 Strassen-like decomposition for the 4x4x1 part more efficiently.
    
    # Let me try a different approach: build a combined decomposition.
    
    # Actually, let me try to use a lattice/linear algebra approach to find a lower rank decomposition.
    
    # For now, let me stick with the known rank-128 decomposition but try to optimize it.
    
    # Strassen 2x2x2 rank-7 decomposition coefficients
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    def build_4x4x4_strassen():
        """Build rank-56 decomposition for 4x4*4x4 using 8 Strassen 2x2x2 blocks.
        Returns (U_cols, V_cols, W_cols) each 16x56 lists.
        """
        U_cols = [[0] * 56 for _ in range(16)]
        V_cols = [[0] * 56 for _ in range(16)]
        W_cols = [[0] * 56 for _ in range(16)]
        
        products = [
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 0, 0),
            (1, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        ]
        
        for prod_idx, (ar, ac, br, bc) in enumerate(products):
            for s in range(7):
                col = prod_idx * 7 + s
                for i in range(2):
                    for j in range(2):
                        U_cols[(ar * 2 + i) * 4 + (ac * 2 + j)][col] = strassen_U[s][i * 2 + j]
                        V_cols[(br * 2 + i) * 4 + (bc * 2 + j)][col] = strassen_V[s][i * 2 + j]
                        W_cols[(bc * 2 + j) * 4 + (ar * 2 + i)][col] = strassen_W[s][i * 2 + j]
        
        return U_cols, V_cols, W_cols
    
    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify the decomposition is correct."""
        num_cols = len(U_mat[0]) if U_mat else 0
        if num_cols == 0:
            return False
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(num_cols):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check zero entries
        zero_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    zero_set.add((i * m + j, j * p + k, k * n + i))
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in zero_set:
                        continue
                    s = 0.0
                    for r in range(num_cols):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(s) > tol:
                        return False
        return True
    
    # Build rank-128 decomposition: 56 + 56 + 16
    total_rank = 128
    U_new = [[0] * total_rank for _ in range(16)]
    V_new = [[0] * total_rank for _ in range(36)]
    W_new = [[0] * total_rank for _ in range(36)]
    
    # Block 1: cols 0-3 of B (4x4 sub-problem), rank 56
    U1, V1, W1 = build_4x4x4_strassen()
    for r in range(56):
        for a in range(16):
            U_new[a][r] = U1[a][r]
        for b in range(16):
            j = b // 4
            k = b % 4
            V_new[j * 9 + k][r] = V1[b][r]
        for c in range(16):
            W_new[c][r] = W1[c][r]
    
    # Block 2: cols 4-7 of B (4x4 sub-problem), rank 56
    U2, V2, W2 = build_4x4x4_strassen()
    for r in range(56):
        for a in range(16):
            U_new[a][56 + r] = U2[a][r]
        for b in range(16):
            j = b // 4
            k = b % 4
            V_new[j * 9 + (4 + k)][56 + r] = V2[b][r]
        for c in range(16):
            k_c = c // 4
            i_c = c % 4
            W_new[(4 + k_c) * 4 + i_c][56 + r] = W2[c][r]
    
    # Block 3: col 8 of B (4x1 sub-problem), rank 16 (schoolbook)
    for i in range(4):
        for j in range(4):
            r = 112 + i * 4 + j
            U_new[i * 4 + j][r] = 1
            V_new[j * 9 + 8][r] = 1
            W_new[8 * 4 + i][r] = 1
    
    if verify(U_new, V_new, W_new, n, m, p):
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback: schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
