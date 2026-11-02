"""Rewrite for 4x4x6 matrix multiplication tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = start + 115  # 5 second safety margin
    
    # Build the target tensor structure
    # T[I,J,K] = 1 when I=i*m+j, J=j*p+k, K=k*n+i for some i,j,k
    # I in [0, 15], J in [0, 23], K in [0, 23]
    
    # Strategy: 4x4x4 (Strassen rank 49) + 4x4x2 (schoolbook rank 32) = rank 81
    # Then try to optimize
    
    # Part 1: 4x4x4 Strassen decomposition (rank 49)
    # This computes C[:, 0:4] = A @ B[:, 0:4]
    
    # Part 2: 4x4x2 schoolbook decomposition (rank 32)
    # This computes C[:, 4:6] = A @ B[:, 4:6]
    
    def build_strassen_4x4x4():
        """Build rank-49 decomposition for 4x4x4 matrix multiplication.
        Returns U (16x49), V (16x49), W (16x49) for the 4x4x4 case.
        Note: in the 4x4x4 case, K index = k*4+i for k,i in [0,3]
        """
        # We'll build this by combining 7 Strassen 2x2x2 decompositions
        # Each Strassen 2x2x2 has rank 7
        # Total: 7 * 7 = 49
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen 2x2x2 decomposition (rank 7)
        # For 2x2x2: A is 2x2, B is 2x2, C = A@B is 2x2
        # I = i*2+j (i,j in 0,1), J = j*2+k (j,k in 0,1), K = k*2+i (k,i in 0,1)
        
        def strassen_2x2x2():
            """Return U,V,W for 2x2x2 with rank 7.
            U shape (4,7), V shape (4,7), W shape (4,7)
            """
            u = np.zeros((4, 7))
            v = np.zeros((4, 7))
            w = np.zeros((4, 7))
            
            # Strassen's 7 multiplications:
            # M1 = (a11+a22)(b11+b22)
            # M2 = (a21+a22)b11
            # M3 = a11(b12-b22)
            # M4 = a22(b21-b11)
            # M5 = (a11+a12)b22
            # M6 = (a21-a11)(b11+b12)
            # M7 = (a12-a22)(b21+b22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For each Strassen product M_r, we need a 2x2x2 rank-1 decomposition
            # Actually, each M_r is itself a 2x2x2 multiplication, which needs rank 7
            # Wait, no. M_r = X * Y where X and Y are specific linear combinations.
            # M_r is a single scalar product of two linear forms.
            # So each M_r contributes ONE rank-1 term to the tensor.
            
            # The 2x2x2 tensor is: T[i,j,k] = 1 if i,j,k define a valid (i,j,k) triple
            # i in {0,1}, j in {0,1}, k in {0,1}
            # I = i*2+j, J = j*2+k, K = k*2+i
            
            # The 7 Strassen terms:
            # Term r: U_r[i*2+j], V_r[j*2+k], W_r[k*2+i]
            
            # M1 = (A11+A22)(B11+B22)
            # A11 is at I=0*2+0=0, A22 is at I=1*2+1=3
            # B11 is at J=0*2+0=0, B22 is at J=1*2+1=3
            # M1 contributes to:
            #   C11 (K=0*2+0=0): +1
            #   C22 (K=1*2+1=3): +1
            # U1: [1, 0, 0, 1] (A11 + A22)
            # V1: [1, 0, 0, 1] (B11 + B22)
            # W1: [1, 0, 0, 1] (C11 + C22)
            
            # M2 = (A21+A22)B11
            # A21 at I=1*2+0=2, A22 at I=3
            # B11 at J=0
            # M2 contributes to:
            #   C21 (K=0*2+1=2): +1
            # U2: [0, 0, 1, 1]
            # V2: [1, 0, 0, 0]
            # W2: [0, 0, 1, 0]
            
            # M3 = A11(B12-B22)
            # A11 at I=0
            # B12 at J=0*2+1=1, B22 at J=3
            # M3 contributes to:
            #   C12 (K=1*2+0=1): +1
            # U3: [1, 0, 0, 0]
            # V3: [0, 1, 0, -1]
            # W3: [0, 1, 0, 0]
            
            # M4 = A22(B21-B11)
            # A22 at I=3
            # B21 at J=1*2+0=2, B11 at J=0
            # M4 contributes to:
            #   C11 (K=0): +1
            #   C21 (K=2): +1
            # U4: [0, 0, 0, 1]
            # V4: [0, 0, 1, -1]
            # W4: [1, 0, 1, 0]
            
            # M5 = (A11+A12)B22
            # A11 at I=0, A12 at I=0*2+1=1
            # B22 at J=3
            # M5 contributes to:
            #   C11 (K=0): -1
            #   C12 (K=1): +1
            # U5: [1, 1, 0, 0]
            # V5: [0, 0, 0, 1]
            # W5: [-1, 1, 0, 0]
            
            # M6 = (A21-A11)(B11+B12)
            # A21 at I=2, A11 at I=0
            # B11 at J=0, B12 at J=1
            # M6 contributes to:
            #   C22 (K=3): +1
            # U6: [-1, 0, 1, 0]
            # V6: [1, 1, 0, 0]
            # W6: [0, 0, 0, 1]
            
            # M7 = (A12-A22)(B21+B22)
            # A12 at I=1, A22 at I=3
            # B21 at J=2, B22 at J=3
            # M7 contributes to:
            #   C11 (K=0): +1
            # U7: [0, 1, 0, -1]
            # V7: [0, 0, 1, 1]
            # W7: [1, 0, 0, 0]
            
            u[:, 0] = [1, 0, 0, 1]
            v[:, 0] = [1, 0, 0, 1]
            w[:, 0] = [1, 0, 0, 1]
            
            u[:, 1] = [0, 0, 1, 1]
            v[:, 1] = [1, 0, 0, 0]
            w[:, 1] = [0, 0, 1, 0]
            
            u[:, 2] = [1, 0, 0, 0]
            v[:, 2] = [0, 1, 0, -1]
            w[:, 2] = [0, 1, 0, 0]
            
            u[:, 3] = [0, 0, 0, 1]
            v[:, 3] = [0, 0, 1, -1]
            w[:, 3] = [1, 0, 1, 0]
            
            u[:, 4] = [1, 1, 0, 0]
            v[:, 4] = [0, 0, 0, 1]
            w[:, 4] = [-1, 1, 0, 0]
            
            u[:, 5] = [-1, 0, 1, 0]
            v[:, 5] = [1, 1, 0, 0]
            w[:, 5] = [0, 0, 0, 1]
            
            u[:, 6] = [0, 1, 0, -1]
            v[:, 6] = [0, 0, 1, 1]
            w[:, 6] = [1, 0, 0, 0]
            
            return u, v, w
        
        # Now build 4x4x4 from 7 Strassen 2x2x2 blocks
        # The 4x4 matrices are split into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # In the 4x4x4 indexing:
        # A[i,j] where i,j in 0..3: I = i*4+j
        # B[j,k] where j,k in 0..3: J = j*4+k
        # C[i,k] where i,k in 0..3: K = k*4+i
        
        # Block A11: rows 0-1, cols 0-1 -> I = 0,1,4,5
        # Block A12: rows 0-1, cols 2-3 -> I = 2,3,6,7
        # Block A21: rows 2-3, cols 0-1 -> I = 8,9,12,13
        # Block A22: rows 2-3, cols 2-3 -> I = 10,11,14,15
        
        # Block B11: rows 0-1, cols 0-1 -> J = 0,1,4,5
        # Block B12: rows 0-1, cols 2-3 -> J = 2,3,6,7
        # Block B21: rows 2-3, cols 0-1 -> J = 8,9,12,13
        # Block B22: rows 2-3, cols 2-3 -> J = 10,11,14,15
        
        # Block C11: rows 0-1, cols 0-1 -> K = 0,1,4,5
        # Block C12: rows 0-1, cols 2-3 -> K = 2,3,6,7
        # Block C21: rows 2-3, cols 0-1 -> K = 8,9,12,13
        # Block C22: rows 2-3, cols 2-3 -> K = 10,11,14,15
        
        # For each of the 7 Strassen products, we need to map the 2x2x2 decomposition
        # to the appropriate 4x4x4 sub-tensor.
        
        # Strassen product r uses specific A-blocks and B-blocks to produce C-blocks.
        # For each Strassen product, the "A factor" is a linear combination of A-blocks,
        # the "B factor" is a linear combination of B-blocks, and the result is added
        # to specific C-blocks.
        
        # Let me define for each of the 7 Strassen products:
        # - Which A-blocks are involved and with what coefficients
        # - Which B-blocks are involved and with what coefficients  
        # - Which C-blocks receive the result and with what coefficients
        
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)B11 -> C21 += M2
        # M3 = A11(B12-B22) -> C12 += M3
        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        
        # For each Strassen product r, the 2x2x2 decomposition gives us 7 rank-1 terms.
        # Each rank-1 term in the 2x2x2 space needs to be "expanded" to the 4x4x4 space.
        
        # Actually, I think the correct way to think about this is:
        # The 4x4x4 tensor can be decomposed using the Strassen algorithm at the block level.
        # Each Strassen multiplication M_r = X_r * Y_r where X_r is a linear combo of A-blocks
        # and Y_r is a linear combo of B-blocks.
        # M_r is a 2x2 matrix (block), and we need to compute it as a 2x2x2 multiplication.
        # So for each r, we use a rank-7 decomposition of the 2x2x2 tensor.
        # Total rank = 7 * 7 = 49.
        
        # Let me define the block mappings:
        # A-blocks: A11 (I:0-3), A12 (I:4-7), A21 (I:8-11), A22 (I:12-15)
        # Wait, let me recompute. A[i,j] with i,j in 0..3, I = i*4+j
        # A11: i in {0,1}, j in {0,1} -> I in {0,1,4,5}
        # A12: i in {0,1}, j in {2,3} -> I in {2,3,6,7}
        # A21: i in {2,3}, j in {0,1} -> I in {8,9,12,13}
        # A22: i in {2,3}, j in {2,3} -> I in {10,11,14,15}
        
        # B-blocks: B[j,k] with j,k in 0..3, J = j*4+k
        # B11: j in {0,1}, k in {0,1} -> J in {0,1,4,5}
        # B12: j in {0,1}, k in {2,3} -> J in {2,3,6,7}
        # B21: j in {2,3}, k in {0,1} -> J in {8,9,12,13}
        # B22: j in {2,3}, k in {2,3} -> J in {10,11,14,15}
        
        # C-blocks: C[i,k] with i,k in 0..3, K = k*4+i
        # C11: i in {0,1}, k in {0,1} -> K in {0,1,4,5}
        # C12: i in {0,1}, k in {2,3} -> K in {2,3,6,7}
        # C21: i in {2,3}, k in {0,1} -> K in {8,9,12,13}
        # C22: i in {2,3}, k in {2,3} -> K in {10,11,14,15}
        
        # For each Strassen product r, define:
        # a_blocks[r]: list of (block_index, coefficient) for A
        # b_blocks[r]: list of (block_index, coefficient) for B
        # c_blocks[r]: list of (block_index, coefficient) for C
        
        # Block indices: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
        
        a_blocks = [
            [(0, 1), (3, 1)],   # M1: A11 + A22
            [(2, 1), (3, 1)],   # M2: A21 + A22
            [(0, 1)],            # M3: A11
            [(3, 1)],            # M4: A22
            [(0, 1), (1, 1)],   # M5: A11 + A12
            [(2, 1), (0, -1)],  # M6: A21 - A11
            [(1, 1), (3, -1)],  # M7: A12 - A22
        ]
        
        b_blocks = [
            [(0, 1), (3, 1)],   # M1: B11 + B22
            [(0, 1)],            # M2: B11
            [(1, 1), (3, -1)],  # M3: B12 - B22
            [(2, 1), (0, -1)],  # M4: B21 - B11
            [(3, 1)],            # M5: B22
            [(0, 1), (1, 1)],   # M6: B11 + B12
            [(2, 1), (3, 1)],   # M7: B21 + B22
        ]
        
        c_blocks = [
            [(0, 1), (3, 1)],   # M1: C11 + C22
            [(2, 1)],            # M2: C21
            [(1, 1)],            # M3: C12
            [(0, 1), (2, 1)],   # M4: C11 + C21
            [(0, -1), (1, 1)],  # M5: -C11 + C12
            [(3, 1)],            # M6: C22
            [(0, 1)],            # M7: C11
        ]
        
        # Now for each Strassen product r (0-6), and each of its 7 rank-1 terms (s=0-6),
        # we create one rank-1 term in the 4x4x4 decomposition.
        
        u2, v2, w2 = strassen_2x2x2()
        
        # For each (r, s) pair, create a rank-1 term in 4x4x4 space
        # The 2x2x2 term (r,s) has:
        # u2[i2*2+j2, s] for the A-part (where i2,j2 are local 2x2 indices)
        # v2[j2*2+k2, s] for the B-part
        # w2[k2*2+i2, s] for the C-part
        
        # We need to map local 2x2 indices to global 4x4 indices for each block.
        
        # For A-block b (0-3), local (i2, j2) maps to global:
        # Block 0 (A11): i = i2, j = j2 -> I = i2*4 + j2
        # Block 1 (A12): i = i2, j = j2+2 -> I = i2*4 + j2+2
        # Block 2 (A21): i = i2+2, j = j2 -> I = (i2+2)*4 + j2
        # Block 3 (A22): i = i2+2, j = j2+2 -> I = (i2+2)*4 + j2+2
        
        # Similarly for B-blocks and C-blocks.
        
        for r in range(7):
            for s in range(7):
                col = r * 7 + s
                u_val = u2[:, s]  # shape (4,)
                v_val = v2[:, s]  # shape (4,)
                w_val = w2[:, s]  # shape (4,)
                
                # Expand A-part
                for (ab, ac) in a_blocks[r]:
                    for i2 in range(2):
                        for j2 in range(2):
                            local_idx = i2 * 2 + j2
                            if ab == 0:
                                global_I = i2 * 4 + j2
                            elif ab == 1:
                                global_I = i2 * 4 + j2 + 2
                            elif ab == 2:
                                global_I = (i2 + 2) * 4 + j2
                            else:  # ab == 3
                                global_I = (i2 + 2) * 4 + j2 + 2
                            U[global_I, col] += ac * u_val[local_idx]
                
                # Expand B-part
                for (bb, bc) in b_blocks[r]:
                    for j2 in range(2):
                        for k2 in range(2):
                            local_idx = j2 * 2 + k2
                            if bb == 0:
                                global_J = j2 * 4 + k2
                            elif bb == 1:
                                global_J = j2 * 4 + k2 + 2
                            elif bb == 2:
                                global_J = (j2 + 2) * 4 + k2
                            else:  # bb == 3
                                global_J = (j2 + 2) * 4 + k2 + 2
                            V[global_J, col] += bc * v_val[local_idx]
                
                # Expand C-part
                for (cb, cc) in c_blocks[r]:
                    for k2 in range(2):
                        for i2 in range(2):
                            local_idx = k2 * 2 + i2
                            if cb == 0:
                                global_K = k2 * 4 + i2
                            elif cb == 1:
                                global_K = k2 * 4 + i2 + 2
                            elif cb == 2:
                                global_K = (k2 + 2) * 4 + i2
                            else:  # cb == 3
                                global_K = (k2 + 2) * 4 + i2 + 2
                            W[global_K, col] += cc * w_val[local_idx]
        
        return U, V, W
    
    def build_schoolbook_4x4x2():
        """Build rank-32 decomposition for 4x4x2 matrix multiplication.
        Computes C[:, 4:6] = A @ B[:, 4:6]
        Returns U (16x32), V (24x32), W (24x32)
        Note: V and W have full size for the 4x4x6 problem.
        """
        U = np.zeros((16, 32))
        V = np.zeros((24, 32))
        W = np.zeros((24, 32))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4, 6):  # k = 4, 5
                    I = i * 4 + j
                    J = j * 6 + k
                    K = k * 4 + i
                    U[I, col] = 1
                    V[J, col] = 1
                    W[K, col] = 1
                    col += 1
        
        return U, V, W
    
    # Build the rank-81 decomposition
    U1, V1, W1 = build_strassen_4x4x4()
    U2, V2, W2 = build_schoolbook_4x4x2()
    
    # Combine
    U = np.zeros((16, 81))
    V = np.zeros((24, 81))
    W = np.zeros((24, 81))
    
    U[:, 0:49] = U1
    V[:, 0:49] = V1
    W[:, 0:49] = W1
    
    U[:, 49:81] = U2
    V[:, 49:81] = V2
    W[:, 49:81] = W2
    
    # Verify the decomposition
    def validate(U, V, W):
        rank = U.shape[1]
        n, m, p = 4, 4, 6
        for _ in range(5):
            x = np.random.randn(n*m)
            y = np.random.randn(m*p)
            z = np.random.randn(p*n)
            
            lhs = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        lhs += x[I] * y[J] * z[K]
            
            rhs = 0.0
            for r in range(rank):
                xr = np.dot(x, U[:, r])
                yr = np.dot(y, V[:, r])
                zr = np.dot(z, W[:, r])
                rhs += xr * yr * zr
            
            if abs(lhs - rhs) > 1e-6:
                return False
        return True
    
    np.random.seed(42)
    
    if not validate(U, V, W):
        # Fallback to schoolbook rank 96
        U = np.zeros((16, 96))
        V = np.zeros((24, 96))
        W = np.zeros((24, 96))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    I = i * 4 + j
                    J = j * 6 + k
                    K = k * 4 + i
                    U[I, col] = 1
                    V[J, col] = 1
                    W[K, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing redundant columns
    # Check if any column is a linear combination of others
    best_rank = 81
    best_U, best_V, best_W = U, V, W
    
    # Try greedy column elimination
    if time.time() < deadline:
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Try removing columns one by one and re-checking
        # This is expensive, so try a few random subsets
        
        # Alternative: try to find a lower rank by random projection
        # or by checking if the tensor has a lower rank structure
        
        # For now, return the rank-81 decomposition
        return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
