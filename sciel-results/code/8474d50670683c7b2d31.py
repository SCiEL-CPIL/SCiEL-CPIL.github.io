"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Decompose as (2,4,2) + (2,4,3)
    # (2,4,2) = (2,2,2) ⊗ (1,2,1), rank 7*2 = 14
    # (2,4,3) = (2,2,3) ⊗ (1,2,1), rank 12*2 = 24
    # Total rank = 38
    
    def strassen_222():
        """Return U, V, W for 2x2x2 Strassen, each 4x7."""
        U = [[0,0,0,0,0,0,0] for _ in range(4)]
        V = [[0,0,0,0,0,0,0] for _ in range(4)]
        W = [[0,0,0,0,0,0,0] for _ in range(4)]
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # C10 += M2
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1  # C01 += M3
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C11 += M6
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1  # C00 += M7, C11 -= M7
        return U, V, W
    
    def schoolbook_mnp(m, n, p):
        """Return U, V, W for m x n x p schoolbook, shapes (m*n, m*n*p), (n*p, m*n*p), (p*m, m*n*p)."""
        rank = m * n * p
        U = [[0]*rank for _ in range(m*n)]
        V = [[0]*rank for _ in range(n*p)]
        W = [[0]*rank for _ in range(p*m)]
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i*n + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*m + i][col] = 1
                    col += 1
        return U, V, W
    
    def tensor_product(U1, V1, W1, U2, V2, W2):
        """Tensor product of two decompositions."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank = r1 * r2
        m1n1 = len(U1)
        n1p1 = len(V1)
        p1m1 = len(W1)
        m2n2 = len(U2)
        n2p2 = len(V2)
        p2m2 = len(W2)
        
        m = m1n1 // len(U1[0])  # not quite, need to infer
        # Actually, let's just compute sizes from the problem
        # U1 is (m1*n1) x r1, U2 is (m2*n2) x r2
        # Result U is (m1*m2 * n1*n2) x rank
        # But we need m, n, p separately
        
        # Simpler: just do element-wise
        U = [[0]*rank for _ in range(len(U1)*len(U2))]
        V = [[0]*rank for _ in range(len(V1)*len(V2))]
        W = [[0]*rank for _ in range(len(W1)*len(W2))]
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1*r2 + c2
                for a1 in range(len(U1)):
                    for a2 in range(len(U2)):
                        a = a1*len(U2) + a2
                        U[a][c] = U1[a1][c1] * U2[a2][c2]
                for b1 in range(len(V1)):
                    for b2 in range(len(V2)):
                        b = b1*len(V2) + b2
                        V[b][c] = V1[b1][c1] * V2[b2][c2]
                for d1 in range(len(W1)):
                    for d2 in range(len(W2)):
                        d = d1*len(W2) + d2
                        W[d][c] = W1[d1][c1] * W2[d2][c2]
        return U, V, W
    
    def add_decompositions(U1, V1, W1, U2, V2, W2, offset_U, offset_V, offset_W):
        """Add two decompositions with given offsets. Returns combined U, V, W."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        total_U = len(U1) + len(U2)
        total_V = len(V1) + len(V2)
        total_W = len(W1) + len(W2)
        
        U = [[0]*total_rank for _ in range(total_U)]
        V = [[0]*total_rank for _ in range(total_V)]
        W = [[0]*total_rank for _ in range(total_W)]
        
        for i in range(len(U1)):
            for c in range(r1):
                U[i+offset_U][c] = U1[i][c]
        for i in range(len(U2)):
            for c in range(r2):
                U[i+offset_U+len(U1)][r1+c] = U2[i][c]
        
        for i in range(len(V1)):
            for c in range(r1):
                V[i+offset_V][c] = V1[i][c]
        for i in range(len(V2)):
            for c in range(r2):
                V[i+offset_V+len(V1)][r1+c] = V2[i][c]
        
        for i in range(len(W1)):
            for c in range(r1):
                W[i+offset_W][c] = W1[i][c]
        for i in range(len(W2)):
            for c in range(r2):
                W[i+offset_W+len(W1)][r1+c] = W2[i][c]
        
        return U, V, W
    
    # Part 1: (2,4,2) = (2,2,2) ⊗ (1,2,1)
    U_222, V_222, W_222 = strassen_222()
    U_121, V_121, W_121 = schoolbook_mnp(1, 2, 1)
    U1, V1, W1 = tensor_product(U_222, V_222, W_222, U_121, V_121, W_121)
    # U1 shape: (2*1 * 2*2) x 14 = (2*4) x 14 = 8x14
    # V1 shape: (2*2 * 2*1) x 14 = (4*2) x 14 = 8x14  
    # W1 shape: (2*2 * 2*1) x 14 = (4*2) x 14 = 8x14
    
    # Part 2: (2,4,3) = (2,2,3) ⊗ (1,2,1)
    U_223, V_223, W_223 = schoolbook_mnp(2, 2, 3)
    U_121_2, V_121_2, W_121_2 = schoolbook_mnp(1, 2, 1)
    U2, V2, W2 = tensor_product(U_223, V_223, W_223, U_121_2, V_121_2, W_121_2)
    # U2 shape: (2*1 * 2*2) x 24 = 8x24
    # V2 shape: (2*2 * 2*1) x 24 = 8x24
    # W2 shape: (3*2 * 2*1) x 24 = 12x24
    
    # Now we need to combine these for the full (2,4,5) problem
    # The full problem has:
    # U: (2*4) x rank = 8 x 38
    # V: (4*5) x rank = 20 x 38  
    # W: (5*2) x rank = 10 x 38
    
    # Part 1 covers k=0,1 (first 2 columns of B, first 2 rows of C in k)
    # Part 2 covers k=2,3,4 (last 3 columns of B, last 3 rows of C in k)
    
    # U for part 1: rows 0-7 (i*4+j for i in {0,1}, j in {0,1,2,3})
    # U for part 2: same rows
    # V for part 1: j*p+k for j in {0,1,2,3}, k in {0,1} -> rows 0-7
    # V for part 2: j*p+k for j in {0,1,2,3}, k in {2,3,4} -> rows 10-19 (since p=5, k=2 -> j*5+2)
    #   Wait, V is indexed by j*p + k where p=5 for the full problem
    #   For part 1 (k=0,1): rows j*5+0 and j*5+1 for j=0..3 -> rows 0,1,5,6,10,11,15,16
    #   For part 2 (k=2,3,4): rows j*5+2, j*5+3, j*5+4 for j=0..3 -> rows 2,3,4,7,8,9,12,13,14,17,18,19
    # W for part 1: k*n+i for k in {0,1}, i in {0,1} -> rows k*2+i: 0,1,2,3
    # W for part 2: k*n+i for k in {2,3,4}, i in {0,1} -> rows 4,5,6,7,8,9
    
    # Let me re-index properly for the full (2,4,5) problem
    # U index: i*m + j = i*4 + j, i in {0,1}, j in {0,1,2,3}
    # V index: j*p + k = j*5 + k, j in {0,1,2,3}, k in {0,1,2,3,4}
    # W index: k*n + i = k*2 + i, k in {0,1,2,3,4}, i in {0,1}
    
    total_rank = 14 + 24  # 38
    total_U = 8
    total_V = 20
    total_W = 10
    
    U_full = [[0]*total_rank for _ in range(total_U)]
    V_full = [[0]*total_rank for _ in range(total_V)]
    W_full = [[0]*total_rank for _ in range(total_W)]
    
    # Place part 1 (k=0,1)
    # U1 is 8x14, maps to U_full rows 0-7
    for i in range(8):
        for c in range(14):
            U_full[i][c] = U1[i][c]
    
    # V1 is 8x14, but V1 index is j*2+k for j in {0,1,2,3}, k in {0,1}
    # Need to map to V_full index j*5+k
    for i in range(8):
        # i = j*2 + k in V1, so j = i//2, k = i%2
        j = i // 2
        k = i % 2
        v_full_idx = j*5 + k
        for c in range(14):
            V_full[v_full_idx][c] = V1[i][c]
    
    # W1 is 8x14, W1 index is k*2+i for k in {0,1}, i in {0,1}
    # Wait, W1 from tensor product: W1 shape is (p1*m1 * p2*m2) x rank
    # p1=2, m1=2, p2=1, m2=1 -> (2*2 * 1*1) = 4x14
    # W1 index: k1*m1*i1... actually for (2,2,2)⊗(1,2,1):
    # W1 is (p1*p2 * m1*m2) or (p1*m1 * p2*m2)? 
    # In my tensor_product, W has size len(W1)*len(W2)
    # W_222 is 4x7 (p1*m1 = 2*2 = 4), W_121 is 2x2 (p2*m2 = 1*1 = 1, wait 1*1=1)
    # Hmm, W_121 for (1,2,1): p*m = 1*1 = 1, so W_121 is 1x2
    # So W1 is 4*1 = 4 rows x 14 cols
    # W1 index: k1*m1 + ... actually it's (k1, i1) paired with (k2, i2)
    # For (2,2,2): W index = k*2+i, k in {0,1}, i in {0,1}
    # For (1,2,1): W index = k*1+i, k in {0}, i in {0} -> just 0
    # Combined: index = w1_idx * 1 + w2_idx = w1_idx (since len(W_121)=1)
    # So W1[i1][c] corresponds to W_full[k1*2+i1][c] where i1 = k1*2+i_full_i... 
    
    # This is getting complex. Let me just verify by checking the dimensions work out.
    # Actually, let me restart with a cleaner approach.
    
    # For the full (2,4,5) problem:
    # We compute C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Split j into j1 in {0,1} and j2 in {2,3}:
    # C[i,k] = sum_{j1=0}^{1} A[i,j1]*B[j1,k] + sum_{j2=2}^{3} A[i,j2]*B[j2,k]
    #          = (A1 * B1)[i,k] + (A2 * B2)[i,k]
    # where A1 = A[:,0:2] (2x2), B1 = B[0:2,:] (2x5)
    #       A2 = A[:,2:4] (2x2), B2 = B[2:4,:] (2x5)
    
    # So we need two (2,2,5) multiplications!
    # R(2,2,5) - what is this?
    # (2,2,5) can be written as (2,2,5) directly, or (2,2,2)⊗(1,1,5/2) no.
    # (2,2,5) = (2,2,5), schoolbook rank = 20.
    # Can we do better? (2,2,5) = (2,2,5), 5 is prime.
    # I think R(2,2,5) = 20.
    # So total rank = 2*20 = 40. Back to schoolbook.
    
    # Alternative split: split k instead
    # C[:,0:2] = A * B[:,0:2] is (2,4,2)
    # C[:,2:5] = A * B[:,2:5] is (2,4,3)
    # This is what I had before: R(2,4,2) + R(2,4,3) = 14 + 24 = 38.
    
    # Let me implement this carefully.
    
    # For (2,4,2): A is 2x4, B1 is 4x2, C1 is 2x2
    # (2,4,2) = (2,2,2)⊗(1,2,1) as computed
    # U1: (2*1 * 2*2) x 14 = 8x14, index i1*m1*j1 * (i2*m2*j2) = (i1,j1,i2,j2)
    #   i1 in {0,1} (from 2x2x2), i2 in {0} (from 1x2x1)
    #   j1 in {0,1}, j2 in {0,1}
    #   Full j = j1*2 + j2, so j in {0,1,2,3}
    #   Full i = i1*1 + i2 = i1
    #   U1 index: i1*4 + j1*2 + j2 = i1*4 + j
    
    # For (2,4,3): A is 2x4, B2 is 4x3, C2 is 2x3
    # (2,4,3) = (2,2,3)⊗(1,2,1)
    # U2: (2*1 * 2*2) x 24 = 8x24, same indexing
    
    # Let me just build it directly.
    
    # Part 1: (2,4,2) using Strassen for the 2x2x2 block
    # U1[i*4+j, c] for i in {0,1}, j in {0,1,2,3}
    # This comes from (2,2,2)⊗(1,2,1)
    # The (2,2,2) part: U_222[i1*2+j1, c1] for i1 in {0,1}, j1 in {0,1}
    # The (1,2,1) part: U_121[i2*2+j2, c2] for i2 in {0}, j2 in {0,1}
    # Combined: i = i1, j = j1*2+j2, c = c1*2+c2
    
    # So U1[i*4 + j1*2+j2, c1*2+c2] = U_222[i*2+j1, c1] * U_121[0*2+j2, c2]
    
    # Similarly for V and W.
    
    # Let me just compute the full tensors directly.
    
    U_full = [[0]*38 for _ in range(8)]
    V_full = [[0]*38 for _ in range(20)]
    W_full = [[0]*38 for _ in range(10)]
    
    # Part 1: (2,4,2), rank 14
    # U_222, V_222, W_222 are 4x7
    # U_121, V_121, W_121 are 4x2 and 2x2 and 1x2
    for c1 in range(7):
        for c2 in range(2):
            c = c1*2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(1):
                        for j2 in range(2):
                            i = i1
                            j = j1*2 + j2
                            u_idx = i*4 + j
                            u_val = U_222[i1*2+j1][c1] * U_121[i2*2+j2][c2]
                            U_full[u_idx][c] += u_val
                            
                            # V: for (2,2,2), V_222[j1*2+k1, c1] for k1 in {0,1}
                            # for (1,2,1), V_121[j2*1+k2, c2] for k2 in {0}
                            # Full: j = j1*2+j2, k = k1 (since k2=0 only, and p=2 for first part)
                            # Wait, for (2,4,2), k is 0 or 1
                            # V_222 index: j1*2+k1, V_121 index: j2*1+k2 = j2
                            # Combined V index in (2,4,2): j1*2+j2 * 2 + k1 = j*2+k1
                            # Full V index: j*5+k1
                            for k1 in range(2):
                                for k2 in range(1):
                                    v_idx_local = (j1*2+j2)*2 + k1  # in (2,4,2) space
                                    v_full_idx = (j1*2+j2)*5 + k1  # in full (2,4,5) space
                                    v_val = V_222[j1*2+k1][c1] * V_121[j2][c2]
                                    V_full[v_full_idx][c] += v_val
                            
                            # W: for (2,2,2), W_222[k1*2+i1, c1]
                            # for (1,2,1), W_121[k2*1+i2, c2] = W_121[0, c2]
                            # Combined W index in (2,4,2): k1*2+i1 (since k2=0, i2=0)
                            # Full W index: k1*2+i1
                            w_idx_local = k1*2 + i1
                            w_full_idx = k1*2 + i1
                            w_val = W_222[k1*2+i1][c1] * W_121[0][c2]
                            W_full[w_full_idx][c] += w_val
    
    # Part 2: (2,4,3), rank 24
    # U_223 is 4x12, U_121_2 is 4x2
    for c1 in range(12):
        for c2 in range(2):
            c = 14 + c1*2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(1):
                        for j2 in range(2):
                            i = i1
                            j = j1*2 + j2
                            u_idx = i*4 + j
                            u_val = U_223[i1*2+j1][c1] * U_121_2[i2*2+j2][c2]
                            U_full[u_idx][c] += u_val
                            
                            # V: for (2,2,3), V_223[j1*3+k1, c1] for k1 in {0,1,2}
                            # for (1,2,1), V_121_2[j2*1+k2, c2] = V_121_2[j2, c2] for k2=0
                            # Full k = k1 + 2 (offset by 2 since this is the second part)
                            for k1 in range(3):
                                v_full_idx = (j1*2+j2)*5 + (k1+2)
                                v_val = V_223[j1*3+k1][c1] * V_121_2[j2][c2]
                                V_full[v_full_idx][c] += v_val
                            
                            # W: for (2,2,3), W_223[k1*2+i1, c1]
                            # Full W index: (k1+2)*2 + i1
                            w_full_idx = (k1+2)*2 + i1
                            w_val = W_223[k1*2+i1][c1] * W_121_2[0][c2]
                            W_full[w_full_idx][c] += w_val
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END

