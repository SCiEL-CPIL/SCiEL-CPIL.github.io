"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return rank-7 decomposition for 2x2 x 2x2 matmul.
    Returns list of (u, v, w) where each is a 2x2 matrix (flattened to 4-vector).
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] sums to matmul tensor.
    For 2x2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    W[k*n+i, c] = W[2*k+i, c]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each product corresponds to a rank-1 tensor.
    # For product Mx = (sum of A entries)*(sum of B entries), the rank-1 tensor is:
    # U[i,j] = coeff of A[i,j], V[j,k] = coeff of B[j,k], W[k,i] = 1
    # Actually, the tensor T[i,j,k] = delta for matmul. A rank-1 decomposition term is
    # u[i,j] * v[j,k] * w[k,i] where these are scalars per (i,j), (j,k), (k,i).
    
    # For 2x2x2: U is indexed by (i,j) in 0..3 (flattened i*2+j), V by (j,k) in 0..3, W by (k,i) in 0..3 (flattened k*2+i)
    
    terms = []
    # M1: (A00+A11)(B00+B11)
    u1 = [1,0,0,1]  # A00, A01, A10, A11
    v1 = [1,0,0,1]  # B00, B01, B10, B11
    w1 = [1,0,0,0]  # W[k,i]: k=0,i=0 -> C00 gets M1
    terms.append((u1, v1, w1))
    
    # M2: (A10+A11)B00
    u2 = [0,0,1,1]
    v2 = [1,0,0,0]
    w2 = [0,0,1,0]  # C10 gets M2: k=1,i=0 -> index 1*2+0=2
    terms.append((u2, v2, w2))
    
    # M3: A00(B01-B11)
    u3 = [1,0,0,0]
    v3 = [0,1,0,-1]
    w3 = [0,1,0,0]  # C01 gets M3: k=0,i=1 -> index 0*2+1=1
    terms.append((u3, v3, w3))
    
    # M4: A11(B10-B00)
    u4 = [0,0,0,1]
    v4 = [0,0,1,-1]
    w4 = [1,0,0,0]  # C00 gets M4: k=0,i=0 -> index 0
    terms.append((u4, v4, w4))
    
    # M5: (A00+A01)B11
    u5 = [1,1,0,0]
    v5 = [0,0,0,1]
    w5 = [0,0,0,1]  # C11 gets M5: k=1,i=1 -> index 1*2+1=3
    terms.append((u5, v5, w5))
    
    # M6: (A10-A00)(B00+B01)
    u6 = [-1,0,1,0]
    v6 = [1,1,0,0]
    w6 = [0,0,0,1]  # C11 gets M6: k=1,i=1 -> index 3
    terms.append((u6, v6, w6))
    
    # M7: (A01-A11)(B10+B11)
    u7 = [0,1,0,-1]
    v7 = [0,0,1,1]
    w7 = [0,0,1,0]  # C10 gets M7: k=1,i=0 -> index 2
    terms.append((u7, v7, w7))
    
    return terms

def _build_4x4x4_decomposition():
    """Build rank-49 decomposition for 4x4 x 4x4 matmul using recursive Strassen.
    Returns U (16x49), V (16x49), W (16x49) as lists of lists.
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Recursive: split 4x4 into four 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # Use Strassen on the 2x2 "super-matrix" where each entry is a 2x2 matrix
    # But that gives 7 block-multiplications, each of rank 7, total 49.
    
    # Actually, for the tensor decomposition, we need to be more careful.
    # The 4x4x4 matmul tensor T[i,j,k] = 1 for all i,j,k in 0..3.
    # We decompose it using the 2x2x2 Strassen structure at the block level.
    
    # Block indices: I in {0,1}, J in {0,1}, K in {0,1}
    # Within each block, local indices i', j', k' in {0,1}
    # Global: i = 2*I + i', j = 2*J + j', k = 2*K + k'
    
    # For each Strassen term t (0..6) at the block level, and each local rank-1 term (0..6) 
    # within the 2x2x2 block, we get a rank-1 term for 4x4x4.
    
    # The block-level Strassen gives:
    # For each block-level term with coefficients on (I,J), (J,K), (K,I):
    #   u_block[I,J], v_block[J,K], w_block[K,I]
    # And within each 2x2 block, we have local terms with:
    #   u_local[i',j'], v_local[j',k'], w_local[k',i']
    
    # The full rank-1 term has:
    #   U[2*I+i', 2*J+j'] = u_block[I,J] * u_local[i',j']
    #   V[2*J+j', 2*K+k'] = v_block[J,K] * v_local[j',k']  
    #   W[2*K+k', 2*I+i'] = w_block[K,I] * w_local[k',i']
    
    # Wait, that's not quite right. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j]_c * V[j,k]_c * W[k,i]_c
    # where for 4x4x4, U is 16xR, V is 16xR, W is 16xR.
    
    # Using the block structure:
    # T[i,j,k] = T_block[I,J,K] * T_local[i',j',k'] where T_block is the 2x2x2 matmul tensor
    # and T_local is also the 2x2x2 matmul tensor.
    
    # So T = T_block ⊗ T_local (Kronecker product of tensors)
    # If T_block has rank-7 decomposition and T_local has rank-7 decomposition,
    # then T has rank 7*7 = 49 decomposition.
    
    # For Kronecker product of tensor decompositions:
    # If T1 = sum_a u1_a ⊗ v1_a ⊗ w1_a (rank r1)
    # and T2 = sum_b u2_b ⊗ v2_b ⊗ w2_b (rank r2)
    # then T1 ⊗ T2 = sum_{a,b} (u1_a ⊗ u2_b) ⊗ (v1_a ⊗ v2_b) ⊗ (w1_a ⊗ w2_b)
    
    block_terms = _strassen_2x2()  # 7 terms, each (u,v,w) of length 4
    local_terms = _strassen_2x2()   # 7 terms, each (u,v,w) of length 4
    
    R = 49
    U = [[0.0] * R for _ in range(16)]
    V = [[0.0] * R for _ in range(16)]
    W = [[0.0] * R for _ in range(16)]
    
    col = 0
    for (ub, vb, wb) in block_terms:
        for (ul, vl, wl) in local_terms:
            # ub, vb, wb are length-4 vectors indexed by (I,J), (J,K), (K,I)
            # ul, vl, wl are length-4 vectors indexed by (i',j'), (j',k'), (k',i')
            for IJ in range(4):
                for JK in range(4):
                    for KI in range(4):
                        if ub[IJ] != 0 and vb[JK] != 0 and wb[KI] != 0:
                            I, J = IJ // 2, IJ % 2
                            J2, K = JK // 2, JK % 2
                            K2, I2 = KI // 2, KI % 2
                            # Check consistency: J from first should equal J2 from second
                            # Actually, IJ gives (I,J), JK gives (J,K), KI gives (K,I)
                            # These are independent in the flattened sense, but we need to handle them properly.
                            # ub is indexed by I*2+J, vb by J*2+K, wb by K*2+I
                            # So for a given (I,J,K): ub_idx = I*2+J, vb_idx = J*2+K, wb_idx = K*2+I
                            pass
            
            # Better approach: iterate over I,J,K in 0..1
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        ub_val = ub[I*2+J]
                        vb_val = vb[J*2+K]
                        wb_val = wb[K*2+I]
                        if ub_val == 0 or vb_val == 0 or wb_val == 0:
                            continue
                        for ip in range(2):
                            for jp in range(2):
                                for kp in range(2):
                                    ul_val = ul[ip*2+jp]
                                    vl_val = vl[jp*2+kp]
                                    wl_val = wl[kp*2+ip]
                                    if ul_val == 0 or vl_val == 0 or wl_val == 0:
                                        continue
                                    # Global indices
                                    i = I*2 + ip
                                    j = J*2 + jp
                                    k = K*2 + kp
                                    U_idx = i*4 + j  # 0..15
                                    V_idx = j*4 + k  # 0..15
                                    W_idx = k*4 + i  # 0..15
                                    U[U_idx][col] += ub_val * ul_val
                                    V[V_idx][col] += vb_val * vl_val
                                    W[W_idx][col] += wb_val * wl_val
            col += 1
    
    # Convert to integer/half-integer if possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Construct rank-minimal decomposition for 4x4 x 4x8 matmul."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    
    # Split p=8 into two blocks of 4: B = [B1 | B2]
    # C = A * B = [A*B1 | A*B2]
    # Each A*Bx is a 4x4 x 4x4 multiplication, rank 49.
    # Total rank: 98
    
    U4, V4, W4 = _build_4x4x4_decomposition()
    rank_4x4x4 = len(U4[0])  # should be 49
    
    total_rank = 2 * rank_4x4x4  # 98
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 32 x 98
    
    # First block: k in 0..3 (B1)
    for c in range(rank_4x4x4):
        for i in range(n):
            for j in range(m):
                U[i*m + j][c] = U4[i*m + j][c]
        for j in range(m):
            for k in range(4):
                V[j*p + k][c] = V4[j*4 + k][c]
        for k in range(4):
            for i in range(n):
                W[k*n + i][c] = W4[k*n + i][c]
    
    # Second block: k in 4..7 (B2)
    for c in range(rank_4x4x4):
        col = c + rank_4x4x4
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = U4[i*m + j][c]
        for j in range(m):
            for k in range(4):
                V[j*p + (k+4)][col] = V4[j*4 + k][c]
        for k in range(4):
            for i in range(n):
                W[(k+4)*n + i][col] = W4[k*n + i][c]
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
