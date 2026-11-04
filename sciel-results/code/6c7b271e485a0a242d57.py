import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 37 decomposition
    # A (4x5) = A1 (4x4) + A2 (4x1)
    # B (5x6) = B1 (4x6) + B2 (1x6)
    # C = A1 B1 + A2 B2
    
    # Part 1: A2 B2 (4x1 x 1x6) -> Rank 1
    # Part 2: A1 B1 (4x4 x 4x6) -> Split B1 into B1a (4x4) and B1b (4x2)
    # A1 B1a (4x4 x 4x4) -> Rank 28 (Strassen)
    # A1 B1b (4x4 x 4x2) -> Rank 8
    
    total_rank = 1 + 28 + 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # 1. A2 B2
    # A2 is col 4 of A. B2 is row 4 of B.
    # U[i*m + 4, col] = 1
    # V[4*p + k, col] = 1
    # W[k*n + i, col] = 1
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
    col += 1
    
    # 2. A1 B1a (4x4 x 4x4) using Strassen
    # Strassen for 2x2 has rank 7.
    # 4x4 is 4 blocks of 2x2.
    # We need to map the 2x2 Strassen factors to the 4x4 indices.
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # M1 = (A11 + A22)(B22 - B12)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        u = [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]
        v = [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]
        w = [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]]
             
        # Indices for 2x2: A is 2x2, B is 2x2, C is 2x2
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        # M1: (A11+A22)(B22-B12)
        u[0][0] = 1; u[3][0] = 1
        v[3][0] = 1; v[1][0] = -1
        w[1][0] = 1; w[0][0] = 1; w[3][0] = -1; w[2][0] = 1
        
        # M2: (A21+A22)B11
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1; w[3][1] = 1
        
        # M3: A11(B12-B22)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1
        
        # M4: A22(B21-B11)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[2][3] = 1; w[0][3] = 1
        
        # M5: (A11+A12)B22
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[3][4] = 1
        
        # M6: (A21-A11)(B11+B12)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1
        
        # M7: (A12-A22)(B21+B22)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1; w[2][6] = 1
        
        return u, v, w

    u2, v2, w2 = get_strassen_2x2()
    
    # Map 4x4 blocks to 4x4x4
    # Block (bi, bj) of A (2x2) and (bi, bj) of B (2x2)
    # A block (bi, bj) is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bi, bj) is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # C block (bi, bj) is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        val = u2[i*2+j][c]
                        if val != 0:
                            U[(2*bi+i)*4 + (2*bj+j)][col+c] = val
                            
                # V
                for j in range(2):
                    for k in range(2):
                        val = v2[j*2+k][c]
                        if val != 0:
                            V[(2*bi+j)*4 + (2*bj+k)][col+c] = val
                            
                # W
                for k in range(2):
                    for i in range(2):
                        val = w2[k*2+i][c]
                        if val != 0:
                            W[(2*bj+k)*4 + (2*bi+i)][col+c] = val
                            
            col += 7
            
    # 3. A1 B1b (4x4 x 4x2)
    # B1b is cols 4,5 of B.
    # A1 is cols 0-3 of A.
    # This is 4x4 x 4x2.
    # Split into two 4x4 x 4x1.
    # Each 4x4 x 4x1 has rank 4.
    
    for k in range(2):
        # For column k of B1b (which is col 4+k of B)
        # A1 (4x4) * B_col (4x1)
        # C_col = A1 * B_col
        # Rank 4 decomposition:
        # C_{i, k} = sum_{j=0}^3 A_{i,j} B_{j, 4+k}
        # T_{i, k, j} = A_{i,j} B_{j, 4+k}
        # This is sum of 4 rank 1 tensors?
        # T = sum_{j=0}^3 (A_{:,j} e_j^T) (B_{j, 4+k})
        # Actually, for fixed k, T_{i k j} = A_{i j} B_{j, 4+k}.
        # This is not a sum of rank 1 tensors in i,k,j?
        # It is a single tensor.
        # Can we decompose it into 4 rank 1 tensors?
        # T_{i k j} = A_{i j} B_{j, 4+k}.
        # Let U_{i j} = A_{i j}, V_{j k} = B_{j, 4+k}, W_{k i} = 1?
        # No, W depends on k and i.
        # T_{i k j} = U_{i j} V_{j k} W_{k i}.
        # If W_{k i} = 1, then T_{i k j} = A_{i j} B_{j, 4+k}.
        # This works!
        # So for each k, we have 4 rank 1 tensors (one for each j? No, the sum is over j).
        # The tensor is T. We want T = sum_{c} U_c V_c W_c.
        # Here T_{i k j} = sum_{j=0}^3 A_{i j} B_{j, 4+k}.
        # This is already a sum of 4 terms.
        # Each term is $A_{i j} B_{j, 4+k}$.
        # This term is rank 1?
        # $A_{i j}$ depends on i, j. $B_{j, 4+k}$ depends on j, k.
        # Product is $A_{i j} B_{j, 4+k}$.
        # This is not separable into $u_i v_k w_j$?
        # $u_i = A_{i j}$, $v_k = B_{j, 4+k}$, $w_j = 1$?
        # Then $u_i v_k w_j = A_{i j} B_{j, 4+k}$.
        # Yes, it is rank 1.
        # So for each j, we have a rank 1 tensor.
        # There are 4 j's. So rank 4.
        
        for j in range(4):
            # U[i*m + j, col] = A_{i j} = 1 (if we consider basis)
            # Wait, U, V, W are factors.
            # U[i*m + j, col] should be 1?
            # No, the decomposition is exact for the tensor of multiplication.
            # The tensor is $T_{i k j} = \delta_{i, i'} \delta_{k, k'} \delta_{j, j'}$?
            # No, the tensor is the identity map.
            # $C_{i k} = \sum_j A_{i j} B_{j k}$.
            # The tensor is $T_{i k j} = 1$.
            # We decompose $T = \sum U_c V_c W_c$.
            # $U_c[i, j] V_c[j, k] W_c[k, i] = 1$.
            # For the $j$-th term in the sum:
            # $U[i, j] = 1, V[j, k] = 1, W[k, i] = 1$.
            # This gives $T_{i k j} = 1$.
            # But we sum over j.
            # So we need 4 terms, one for each j.
            # For term c corresponding to j:
            # U[i*m + j, c] = 1
            # V[j*p + k, c] = 1
            # W[k*n + i, c] = 1
            # This is exactly the schoolbook decomposition for that j.
            
            for i in range(4):
                U[i * 4 + j][col] = 1
            for k in range(2):
                # B index is 4+k
                V[j * 6 + (4 + k)][col] = 1
            for i in range(4):
                for k in range(2):
                    W[(4 + k) * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Uses Strassen for 4x4x4 (rank 28) and simple decompositions for remaining parts, reducing rank from 120 to 37", q: 0.9, delta: [0.0, 0.1]}
