import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build rank-28 Strassen decomposition for 4x4x4
    # U: (16, 28), V: (16, 28), W: (16, 28)
    # Indices:
    # U[i*4+j, r]
    # V[j*4+k, r]
    # W[k*4+i, r]
    
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    
    # Strassen terms for 4x4x4
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Block indices in 4x4 matrix:
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    # We define the coefficients for U, V, W for each of the 28 scalar multiplications.
    # Each Mk has 4 scalar multiplications (a,b,c in 0,1).
    # Mk[a,b] = sum_c Xk[a,c] Yk[c,b]
    
    def get_block(i, j, block_i, block_j):
        # block_i, block_j in 0,1
        # returns (i, j) in 0..3
        return (block_i * 2 + i, block_j * 2 + j)

    terms = []
    
    # M1
    # X1 = A11 + A22
    # Y1 = B11 + B22
    # C11 += M1, C22 += M1
    for a in range(2):
        for b in range(2):
            for c in range(2):
                # X1[a,c] involves A11[a,c] and A22[a,c]
                # Y1[c,b] involves B11[c,b] and B22[c,b]
                
                # Term for A11[a,c] * B11[c,b]
                # U: A11[a,c] -> i=a, j=c. Block 0,0.
                # V: B11[c,b] -> j=c, k=b. Block 0,0.
                # W: C11[a,b] and C22[a,b].
                # C11[a,b] -> i=a, k=b. Block 0,0.
                # C22[a,b] -> i=a, k=b. Block 1,1.
                
                # This is getting messy to hardcode 28 terms manually.
                # Let's just generate them programmatically.
                
                pass
    
    # I will use a helper to add terms.
    # add_term(U_indices, V_indices, W_indices, coeff)
    
    # Let's define the Strassen structure explicitly.
    
    # M1: X = A11+A22, Y = B11+B22
    # X blocks: (0,0) coeff 1, (1,1) coeff 1
    # Y blocks: (0,0) coeff 1, (1,1) coeff 1
    # C contributions: C11 (0,0) coeff 1, C22 (1,1) coeff 1
    
    # M2: X = A21+A22, Y = B11
    # X blocks: (1,0) coeff 1, (1,1) coeff 1
    # Y blocks: (0,0) coeff 1
    # C contributions: C21 (1,0) coeff 1
    
    # M3: X = A11, Y = B12-B22
    # X blocks: (0,0) coeff 1
    # Y blocks: (0,1) coeff 1, (1,1) coeff -1
    # C contributions: C12 (0,1) coeff 1
    
    # M4: X = A22, Y = B21-B11
    # X blocks: (1,1) coeff 1
    # Y blocks: (1,0) coeff 1, (0,0) coeff -1
    # C contributions: C11 (0,0) coeff 1, C21 (1,0) coeff 1
    
    # M5: X = A11+A12, Y = B22
    # X blocks: (0,0) coeff 1, (0,1) coeff 1
    # Y blocks: (1,1) coeff 1
    # C contributions: C11 (0,0) coeff -1, C12 (0,1) coeff 1
    
    # M6: X = A21-A11, Y = B11+B12
    # X blocks: (1,0) coeff 1, (0,0) coeff -1
    # Y blocks: (0,0) coeff 1, (0,1) coeff 1
    # C contributions: C22 (1,1) coeff 1
    
    # M7: X = A12-A22, Y = B21+B22
    # X blocks: (0,1) coeff 1, (1,1) coeff -1
    # Y blocks: (1,0) coeff 1, (1,1) coeff 1
    # C contributions: C11 (0,0) coeff 1
    
    strassen_terms = [
        # (X_blocks, Y_blocks, C_blocks)
        # X_blocks: list of (block_i, block_j, coeff)
        # Y_blocks: list of (block_i, block_j, coeff)
        # C_blocks: list of (block_i, block_j, coeff)
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )
    ]
    
    r = 0
    for X_blocks, Y_blocks, C_blocks in strassen_terms:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    # X[a,c] = sum X_coeff * A[block_i*2+a, block_j*2+c]
                    # Y[c,b] = sum Y_coeff * B[block_i*2+c, block_j*2+b]
                    # C[a,b] = sum C_coeff * C[block_i*2+a, block_j*2+b]
                    
                    # For each combination of X_block and Y_block, we have a term
                    # Actually, the multiplication is (sum X) (sum Y)
                    # = sum X_i Y_j
                    # So we need to expand the product.
                    
                    for (xi, xj, xc) in X_blocks:
                        for (yi, yj, yc) in Y_blocks:
                            coeff = xc * yc
                            
                            # U index: i = xi*2 + a, j = xj*2 + c
                            u_idx = (xi*2 + a) * 4 + (xj*2 + c)
                            U4[u_idx, r] += coeff
                            
                            # V index: j = yi*2 + c, k = yj*2 + b
                            v_idx = (yi*2 + c) * 4 + (yj*2 + b)
                            V4[v_idx, r] += coeff
                            
                            # W index: i = ci*2 + a, k = cj*2 + b
                            # Wait, C blocks are (ci, cj).
                            # C[ci*2+a, cj*2+b] gets coeff * C_coeff?
                            # No, the term is X[a,c] Y[c,b].
                            # This contributes to C[a,b] in the block sense?
                            # No, M_k is a 2x2 matrix.
                            # M_k[a,b] is the (a,b) entry.
                            # The C blocks are added to C.
                            # C_block[ci,cj] += C_coeff * M_k[a,b]?
                            # No, C_block[ci,cj] is a 2x2 matrix.
                            # C_block[ci,cj][a,b] += C_coeff * M_k[a,b].
                            # So for each (ci, cj, cc) in C_blocks:
                            for (ci, cj, cc) in C_blocks:
                                w_idx = (cj*2 + b) * 4 + (ci*2 + a) # W[k*4+i]
                                W4[w_idx, r] += cc * coeff
                            
                    r += 1
                    
    if time.time() > deadline - 10:
        # Fallback to rank 42 if too slow (shouldn't be)
        pass

    # Slice for 2x4x6
    # U: (8, 28) -> i in 0,1; j in 0,1,2,3
    # V: (24, 28) -> j in 0,1,2,3; k in 0,1,2,3,4,5
    # W: (12, 28) -> i in 0,1; k in 0,1,2,3,4,5
    
    U2 = np.zeros((8, 28))
    V2 = np.zeros((24, 28))
    W2 = np.zeros((12, 28))
    
    for i in range(2):
        for j in range(4):
            U2[i*4+j, :] = U4[i*4+j, :]
            
    for j in range(4):
        for k in range(6):
            V2[j*6+k, :] = V4[j*4+k, :]
            
    for i in range(2):
        for k in range(6):
            W2[k*2+i, :] = W4[k*4+i, :]
            
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
