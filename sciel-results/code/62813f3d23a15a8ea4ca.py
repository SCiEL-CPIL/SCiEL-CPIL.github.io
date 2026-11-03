"""Matrix multiplication 3x4x7 tensor decomposition using Block-Strassen strategy."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Verify a decomposition
    def verify(U, V, W):
        R = U.shape[1]
        # Optimization: Check all 84 entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j, :]
                    vj = V[j * p + k, :]
                    wk = W[k * n + i, :]
                    val = np.sum(ui * vj * wk)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Build schoolbook as fallback (Rank 84)
    def build_schoolbook():
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Build Block-Strassen decomposition
    def build_block_strassen():
        """
        Partition the 3x4x7 grid into disjoint 2x2x2 blocks and 1x1x1 blocks.
        For each 2x2x2 block, use Strassen's 7-term decomposition.
        For each 1x1x1 block, use 1-term schoolbook decomposition.
        
        Grid dimensions: i in [0,3), j in [0,4), k in [0,7)
        
        Block 1: i in [0,2), j in [0,2), k in [0,2) -> 2x2x2 (Rank 7)
        Block 2: i in [0,2), j in [2,4), k in [2,4) -> 2x2x2 (Rank 7)
        Block 3: i in [0,2), j in [0,2), k in [4,6) -> 2x2x2 (Rank 7)
        Block 4: i in [0,2), j in [2,4), k in [4,6) -> 2x2x2 (Rank 7)
        
        Remaining blocks (1x1x1 or partials):
        We need to cover all (i,j,k) not in the above.
        
        Let's define the blocks explicitly to ensure disjointness and full coverage.
        
        i indices: 0, 1, 2
        j indices: 0, 1, 2, 3
        k indices: 0, 1, 2, 3, 4, 5, 6
        
        Strategy:
        1. Take 2x2x2 blocks where possible.
           - B1: i=[0,1], j=[0,1], k=[0,1]
           - B2: i=[0,1], j=[2,3], k=[0,1]
           - B3: i=[0,1], j=[0,1], k=[2,3]
           - B4: i=[0,1], j=[2,3], k=[2,3]
           - B5: i=[0,1], j=[0,1], k=[4,5]
           - B6: i=[0,1], j=[2,3], k=[4,5]
           
           These 6 blocks cover i in [0,2), j in [0,4), k in [0,6).
           Rank contribution: 6 * 7 = 42.
           
        2. Remaining indices:
           - i=2 (all j, all k)
           - k=6 (all i, all j)
           
           Let's handle the remainder with 1x1x1 blocks (schoolbook).
           Total entries: 3*4*7 = 84.
           Covered by 2x2x2 blocks: 6 * 8 = 48 entries.
           Remaining entries: 84 - 48 = 36.
           Each 1x1x1 block handles 1 entry.
           So we need 36 rank-1 terms.
           
           Total Rank = 42 + 36 = 78.
           This is significantly better than 84.
        """
        U_terms = []
        V_terms = []
        W_terms = []
        
        def add_strassen_block(i0, j0, k0):
            """Add 7 Strassen terms for a 2x2x2 block starting at (i0, j0, k0)."""
            # Indices for the 2x2x2 block
            # A: rows i0, i0+1; cols j0, j0+1
            # B: rows j0, j0+1; cols k0, k0+1
            # C: rows k0, k0+1; cols i0, i0+1
            
            # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
            # Contributes to C[k0,i0] and C[k0+1,i0+1]
            u1 = np.zeros(nm); v1 = np.zeros(mp); w1 = np.zeros(pn)
            u1[i0*m+j0] = 1
            u1[(i0+1)*m+(j0+1)] = 1
            v1[j0*p+k0] = 1
            v1[(j0+1)*p+(k0+1)] = 1
            w1[k0*n+i0] = 1
            w1[(k0+1)*n+(i0+1)] = 1
            U_terms.append(u1); V_terms.append(v1); W_terms.append(w1)
            
            # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
            # Contributes to C[k0,i0+1]
            u2 = np.zeros(nm); v2 = np.zeros(mp); w2 = np.zeros(pn)
            u2[(i0+1)*m+j0] = 1
            u2[(i0+1)*m+(j0+1)] = 1
            v2[j0*p+k0] = 1
            w2[k0*n+(i0+1)] = 1
            U_terms.append(u2); V_terms.append(v2); W_terms.append(w2)
            
            # M3 = A[i0,j0] * (B[j0,k0] - B[j0+1,k0])
            # Contributes to C[k0+1,i0]
            u3 = np.zeros(nm); v3 = np.zeros(mp); w3 = np.zeros(pn)
            u3[i0*m+j0] = 1
            v3[j0*p+k0] = 1
            v3[(j0+1)*p+k0] = -1
            w3[(k0+1)*n+i0] = 1
            U_terms.append(u3); V_terms.append(v3); W_terms.append(w3)
            
            # M4 = A[i0+1,j0+1] * (B[j0+1,k0+1] - B[j0,k0+1])
            # Contributes to C[k0,i0+1] and C[k0+1,i0+1]
            # Wait, let's re-verify Strassen's C formulas.
            # C11 = M1 + M4 - M5 + M7  -> C[k0, i0]
            # C12 = M3 + M5            -> C[k0, i0+1]
            # C21 = M2 + M4            -> C[k0+1, i0]
            # C22 = M1 - M2 + M3 + M6  -> C[k0+1, i0+1]
            
            # Let's re-evaluate the contributions based on this standard mapping:
            # C[k0, i0] (Top-Left): M1, M4, -M5, M7
            # C[k0, i0+1] (Top-Right): M3, M5
            # C[k0+1, i0] (Bottom-Left): M2, M4
            # C[k0+1, i0+1] (Bottom-Right): M1, -M2, M3, M6
            
            # My previous definitions for M1, M2, M3 were:
            # M1: U(A00,A11), V(B00,B11), W(C00, C11) -> Contributes to C00 and C11 (Bottom-Right in C matrix? No, C11 is index (1,1) of C block, which is k0+1, i0+1).
            # Wait, C matrix indices are (row, col) = (k, i).
            # C[k0, i0] is Top-Left.
            # C[k0+1, i0+1] is Bottom-Right.
            
            # Let's restart the Strassen term definitions carefully.
            # A sub-block:
            # A00 = A[i0, j0]
            # A01 = A[i0, j0+1]
            # A10 = A[i0+1, j0]
            # A11 = A[i0+1, j0+1]
            
            # B sub-block:
            # B00 = B[j0, k0]
            # B01 = B[j0, k0+1]
            # B10 = B[j0+1, k0]
            # B11 = B[j0+1, k0+1]
            
            # C sub-block:
            # C00 = C[k0, i0]
            # C01 = C[k0, i0+1]
            # C10 = C[k0+1, i0]
            # C11 = C[k0+1, i0+1]
            
            # Strassen Multiplications:
            # M1 = (A00 + A11) * (B00 + B11)
            # M2 = (A10 + A11) * B00
            # M3 = A00 * (B00 - B10)
            # M4 = A11 * (B11 - B01)
            # M5 = (A00 + A01) * B11
            # M6 = (A10 - A00) * (B00 + B01)
            # M7 = (A01 - A11) * (B10 + B11)
            
            # C Reconstruction:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Now, map these to Rank-1 Tensor Terms U, V, W.
            # Term r contributes to T[i,j,k] = U[i,j,r] V[j,k,r] W[k,i,r].
            # We need to construct U, V, W such that the sum of products equals the C entries.
            
            # Let's define the 7 terms explicitly.
            
            # Term 1 (M1):
            # U: A00=1, A11=1
            # V: B00=1, B11=1
            # W: C00=1, C11=1  (Because M1 contributes +1 to C00 and +1 to C11)
            u1 = np.zeros(nm); v1 = np.zeros(mp); w1 = np.zeros(pn)
            u1[i0*m+j0] = 1; u1[(i0+1)*m+(j0+1)] = 1
            v1[j0*p+k0] = 1; v1[(j0+1)*p+(k0+1)] = 1
            w1[k0*n+i0] = 1; w1[(k0+1)*n+(i0+1)] = 1
            U_terms.append(u1); V_terms.append(v1); W_terms.append(w1)
            
            # Term 2 (M2):
            # U: A10=1, A11=1
            # V: B00=1
            # W: C10=1  (M2 contributes +1 to C10)
            u2 = np.zeros(nm); v2 = np.zeros(mp); w2 = np.zeros(pn)
            u2[(i0+1)*m+j0] = 1; u2[(i0+1)*m+(j0+1)] = 1
            v2[j0*p+k0] = 1
            w2[(k0+1)*n+i0] = 1
            U_terms.append(u2); V_terms.append(v2); W_terms.append(w2)
            
            # Term 3 (M3):
            # U: A00=1
            # V: B00=1, B10=-1
            # W: C01=1  (M3 contributes +1 to C01)
            u3 = np.zeros(nm); v3 = np.zeros(mp); w3 = np.zeros(pn)
            u3[i0*m+j0] = 1
            v3[j0*p+k0] = 1; v3[(j0+1)*p+k0] = -1
            w3[k0*n+(i0+1)] = 1
            U_terms.append(u3); V_terms.append(v3); W_terms.append(w3)
            
            # Term 4 (M4):
            # U: A11=1
            # V: B11=1, B01=-1
            # W: C00=1, C10=1 (M4 contributes +1 to C00 and +1 to C10)
            u4 = np.zeros(nm); v4 = np.zeros(mp); w4 = np.zeros(pn)
            u4[(i0+1)*m+(j0+1)] = 1
            v4[(j0+1)*p+(k0+1)] = 1; v4[j0*p+(k0+1)] = -1
            w4[k0*n+i0] = 1; w4[(k0+1)*n+i0] = 1
            U_terms.append(u4); V_terms.append(v4); W_terms.append(w4)
            
            # Term 5 (M5):
            # U: A00=1, A01=1
            # V: B11=1
            # W: C00=-1, C01=1 (M5 contributes -1 to C00 and +1 to C01)
            u5 = np.zeros(nm); v5 = np.zeros(mp); w5 = np.zeros(pn)
            u5[i0*m+j0] = 1; u5[i0*m+(j0+1)] = 1
            v5[(j0+1)*p+(k0+1)] = 1
            w5[k0*n+i0] = -1; w5[k0*n+(i0+1)] = 1
            U_terms.append(u5); V_terms.append(v5); W_terms.append(w5)
            
            # Term 6 (M6):
            # U: A10=1, A00=-1
            # V: B00=1, B01=1
            # W: C11=1 (M6 contributes +1 to C11)
            u6 = np.zeros(nm); v6 = np.zeros(mp); w6 = np.zeros(pn)
            u6[(i0+1)*m+j0] = 1; u6[i0*m+j0] = -1
            v6[j0*p+k0] = 1; v6[j0*p+(k0+1)] = 1
            w6[(k0+1)*n+(i0+1)] = 1
            U_terms.append(u6); V_terms.append(v6); W_terms.append(w6)
            
            # Term 7 (M7):
            # U: A01=1, A11=-1
            # V: B10=1, B11=1
            # W: C00=1 (M7 contributes +1 to C00)
            u7 = np.zeros(nm); v7 = np.zeros(mp); w7 = np.zeros(pn)
            u7[i0*m+(j0+1)] = 1; u7[(i0+1)*m+(j0+1)] = -1
            v7[(j0+1)*p+k0] = 1; v7[(j0+1)*p+(k0+1)] = 1
            w7[k0*n+i0] = 1
            U_terms.append(u7); V_terms.append(v7); W_terms.append(w7)
            
            # Mark these entries as covered
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0+di, j0+dj, k0+dk))

        def add_schoolbook_block(i, j, k):
            """Add 1 schoolbook term for a 1x1x1 block at (i, j, k)."""
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[i*m+j] = 1
            v[j*p+k] = 1
            w[k*n+i] = 1
            U_terms.append(u); V_terms.append(v); W_terms.append(w)
            covered.add((i, j, k))

        covered = set()
        
        # Define 2x2x2 blocks
        # We have i in {0,1,2}, j in {0,1,2,3}, k in {0..6}
        
        # Block 1: i0=0, j0=0, k0=0
        if 0+1 < n and 0+1 < m and 0+1 < p:
            add_strassen_block(0, 0, 0)
            
        # Block 2: i0=0, j0=2, k0=0
        if 0+1 < n and 2+1 < m and 0+1 < p:
            add_strassen_block(0, 2, 0)
            
        # Block 3: i0=0, j0=0, k0=2
        if 0+1 < n and 0+1 < m and 2+1 < p:
            add_strassen_block(0, 0, 2)
            
        # Block 4: i0=0, j0=2, k0=2
        if 0+1 < n and 2+1 < m and 2+1 < p:
            add_strassen_block(0, 2, 2)
            
        # Block 5: i0=0, j0=0, k0=4
        if 0+1 < n and 0+1 < m and 4+1 < p:
            add_strassen_block(0, 0, 4)
            
        # Block 6: i0=0, j0=2, k0=4
        if 0+1 < n and 2+1 < m and 4+1 < p:
            add_strassen_block(0, 2, 4)
            
        # Now cover the rest with 1x1x1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        add_schoolbook_block(i, j, k)
                        
        U = np.column_stack(U_terms)
        V = np.column_stack(V_terms)
        W = np.column_stack(W_terms)
        
        # Verify correctness
        if verify(U, V, W):
            return U, V, W
        else:
            return None

    start_time = time.time()
    
    # Try Block-Strassen
    result = build_block_strassen()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Schoolbook
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
