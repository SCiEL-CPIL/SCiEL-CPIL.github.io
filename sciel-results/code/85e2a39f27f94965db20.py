import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _strassen_222():
    """Return (U, V, W) for 2x2x2 matrix mult, rank 7 (Strassen)."""
    # A = [[a0,a1],[a2,a3]] (2x2), B = [[b0,b1],[b2,b3]] (2x2)
    # C = [[c0,c1],[c2,c3]] (2x2)
    # U: 4 entries (A), V: 4 entries (B), W: 4 entries (C)
    # 7 products:
    # P1 = a0 * b0
    # P2 = a2 * b2
    # P3 = (a0+a1) * (b0+b2)
    # P4 = (a2+a3) * (b1+b3)
    # P5 = (a0+a3) * (b0+b3)
    # P6 = (a1+a3) * (b2-b3)
    # P7 = (a0+a2) * (b1-b2)
    # c0 = P1 + P4 - P5 + P7
    # c1 = P3 + P5 - P1 - P4  (wait, let me recheck)
    # Actually standard Strassen:
    # c0 = P1 + P4 - P5 + P7
    # c1 = P3 + P5 - P1 - P4  -- no
    # Let me use the standard form:
    # M1 = a00*(b00+b10)
    # M2 = (a00+a10)*b11
    # M3 = (a01-a11)*(b00+b10)
    # M4 = a11*(b01+b11)
    # M5 = (a00+a11)*(b00+b11)
    # M6 = (a01+a11)*(b10-b00)
    # M7 = (a00-a10)*(b01+b11)
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M5 + M1 - M3 - M7
    #
    # U indices: a00=0, a01=1, a10=2, a11=3
    # V indices: b00=0, b01=1, b10=2, b11=3
    # W indices: c00=0, c01=1, c10=2, c11=3
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = a00*(b00+b10)
    U[0,0]=1; V[0,0]=1; V[2,0]=1
    # M2 = (a00+a10)*b11
    U[0,1]=1; U[2,1]=1; V[3,1]=1
    # M3 = (a01-a11)*(b00+b10)
    U[1,2]=1; U[3,2]=-1; V[0,2]=1; V[2,2]=1
    # M4 = a11*(b01+b11)
    U[3,3]=1; V[1,3]=1; V[3,3]=1
    # M5 = (a00+a11)*(b00+b11)
    U[0,4]=1; U[3,4]=1; V[0,4]=1; V[3,4]=1
    # M6 = (a01+a11)*(b10-b00)
    U[1,5]=1; U[3,5]=1; V[2,5]=1; V[0,5]=-1
    # M7 = (a00-a10)*(b01+b11)
    U[0,6]=1; U[2,6]=-1; V[1,6]=1; V[3,6]=1
    # C00 = M5 + M4 - M2 + M6
    W[0,4]=1; W[0,3]=1; W[0,1]=-1; W[0,5]=1
    # C01 = M1 + M2
    W[1,0]=1; W[1,1]=1
    # C10 = M3 + M4
    W[2,2]=1; W[2,3]=1
    # C11 = M5 + M1 - M3 - M7
    W[3,4]=1; W[3,0]=1; W[3,2]=-1; W[3,6]=-1
    return U, V, W

def _matvec_221():
    """Return (U, V, W) for 2x2x1 (matrix times 1-vector), rank 4."""
    # A = [[a0,a1],[a2,a3]] (2x2), x = [x0, x1] (2x1)
    # C = [c0, c1] (2x1)
    # c0 = a0*x0 + a1*x1
    # c1 = a2*x0 + a3*x1
    # 4 products:
    # P1 = a0*x0
    # P2 = a1*x1
    # P3 = a2*x0
    # P4 = a3*x1
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    # P1 = a0*x0
    U[0,0]=1; V[0,0]=1
    # P2 = a1*x1
    U[1,1]=1; V[1,1]=1
    # P3 = a2*x0
    U[2,2]=1; V[0,2]=1
    # P4 = a3*x1
    U[3,3]=1; V[1,3]=1
    # c0 = P1 + P2
    W[0,0]=1; W[0,1]=1
    # c1 = P3 + P4
    W[1,2]=1; W[1,3]=1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    start_time = time.time()
    deadline = 140
    
    # Schoolbook fallback
    rank_school = n * m * p  # 56
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Build the Strassen-based decomposition
    # Split middle dim (4) into two 2-blocks: j in {0,1} and j in {2,3}
    # Split output dim (7) into: k in {0,1}, k in {2,3}, k in {4,5,6}
    # For each (i-block, j-block, k-block) combination:
    #   i-block: i in {0,1} (only one 2x2 block since n=2)
    #   j-block: 2 blocks of size 2
    #   k-block: 2 blocks of size 2 + 1 block of size 1
    
    # We'll build U, V, W by concatenating the block decompositions
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Get the Strassen and matvec components
    sU, sV, sW = _strassen_222()  # (4,7), (4,7), (4,7)
    mU, mV, mW = _matvec_221()    # (4,4), (2,4), (2,4)
    
    # Block structure:
    # For j-block 0 (j=0,1):
    #   For k-block 0 (k=0,1): 2x2x2 -> Strassen
    #   For k-block 1 (k=2,3): 2x2x2 -> Strassen
    #   For k-block 2 (k=4,5,6): 2x2x1 -> matvec
    # For j-block 1 (j=2,3):
    #   Same structure
    
    # U indices: i*m + j = i*4 + j, i in {0,1}, j in {0,1,2,3}
    #   j-block 0: j=0,1 -> U indices 0,1,4,5 (for i=0: 0,1; for i=1: 4,5)
    #   j-block 1: j=2,3 -> U indices 2,3,6,7 (for i=0: 2,3; for i=1: 6,7)
    
    # V indices: j*p + k = j*7 + k, j in {0,1,2,3}, k in {0,...,6}
    #   j-block 0, k-block 0: j=0,1; k=0,1 -> V indices 0,1,7,8
    #   j-block 0, k-block 1: j=0,1; k=2,3 -> V indices 2,3,9,10
    #   j-block 0, k-block 2: j=0,1; k=4,5,6 -> V indices 4,5,6,11,12,13
    #   j-block 1, k-block 0: j=2,3; k=0,1 -> V indices 14,15,21,22
    #   j-block 1, k-block 1: j=2,3; k=2,3 -> V indices 16,17,23,24
    #   j-block 1, k-block 2: j=2,3; k=4,5,6 -> V indices 18,19,20,25,26,27
    
    # W indices: k*n + i = k*2 + i, k in {0,...,6}, i in {0,1}
    #   k-block 0: k=0,1 -> W indices 0,1,2,3
    #   k-block 1: k=2,3 -> W indices 4,5,6,7
    #   k-block 2: k=4,5,6 -> W indices 8,9,10,11,12,13
    
    # Build the full decomposition
    # Total rank = 2 (j-blocks) * (7 + 7 + 4) = 2 * 18 = 36
    
    rank_total = 36
    U_full = np.zeros((dim_u, rank_total))
    V_full = np.zeros((dim_v, rank_total))
    W_full = np.zeros((dim_w, rank_total))
    
    col = 0
    for jb in range(2):  # j-block
        for kb in range(3):  # k-block
            if kb < 2:
                # 2x2x2 block -> Strassen
                bU, bV, bW = sU, sV, sW
                b_rank = 7
                # U indices for this j-block: i in {0,1}, j in {2*jb, 2*jb+1}
                # U global index = i*4 + (2*jb + i_local) where i_local in {0,1}
                # bU has indices 0,1,2,3 corresponding to a00,a01,a10,a11
                # a00 -> i=0, j=2*jb; a01 -> i=0, j=2*jb+1
                # a10 -> i=1, j=2*jb; a11 -> i=1, j=2*jb+1
                u_map = [0*4 + (2*jb + 0), 0*4 + (2*jb + 1), 1*4 + (2*jb + 0), 1*4 + (2*jb + 1)]
                # V indices: j in {2*jb, 2*jb+1}, k in {2*kb, 2*kb+1}
                # V global index = j*7 + k
                # bV has indices 0,1,2,3 corresponding to b00,b01,b10,b11
                # b00 -> j=2*jb, k=2*kb; b01 -> j=2*jb, k=2*kb+1
                # b10 -> j=2*jb+1, k=2*kb; b11 -> j=2*jb+1, k=2*kb+1
                v_map = [(2*jb)*7 + (2*kb + 0), (2*jb)*7 + (2*kb + 1), (2*jb+1)*7 + (2*kb + 0), (2*jb+1)*7 + (2*kb + 1)]
                # W indices: k in {2*kb, 2*kb+1}, i in {0,1}
                # W global index = k*2 + i
                # bW has indices 0,1,2,3 corresponding to c00,c01,c10,c11
                # c00 -> k=2*kb, i=0; c01 -> k=2*kb, i=1
                # c10 -> k=2*kb+1, i=0; c11 -> k=2*kb+1, i=1
                w_map = [(2*kb + 0)*2 + 0, (2*kb + 0)*2 + 1, (2*kb + 1)*2 + 0, (2*kb + 1)*2 + 1]
                
                for r in range(b_rank):
                    for lu in range(4):
                        if bU[lu, r] != 0:
                            U_full[u_map[lu], col + r] = bU[lu, r]
                    for lv in range(4):
                        if bV[lv, r] != 0:
                            V_full[v_map[lv], col + r] = bV[lv, r]
                    for lw in range(4):
                        if bW[lw, r] != 0:
                            W_full[w_map[lw], col + r] = bW[lw, r]
            else:
                # 2x2x1 block -> matvec
                bU, bV, bW = mU, mV, mW
                b_rank = 4
                # U indices: same as above
                u_map = [0*4 + (2*jb + 0), 0*4 + (2*jb + 1), 1*4 + (2*jb + 0), 1*4 + (2*jb + 1)]
                # V indices: j in {2*jb, 2*jb+1}, k in {4,5,6} (only first 2 k's for 2x2x1)
                # Wait, 2x2x1 means 2x2 matrix times 1x1 vector? No, 2x2 times 2x1.
                # k-block 2 has k=4,5,6. For 2x2x1, we use k=4,5 (first two).
                # Actually, the 2x2x1 block is 2x2 A-block times 2x1 B-block.
                # B-block has j in {2*jb, 2*jb+1} and k in {4,5} (2 rows, 2 cols? No.)
                # Wait, B is 4x7. B[j,k] for j in {2*jb, 2*jb+1} and k in {4,5,6}.
                # For 2x2x1: A is 2x2 (i in {0,1}, j in {2*jb, 2*jb+1})
                # B is 2x2 (j in {2*jb, 2*jb+1}, k in {4,5})? No, that's 2x2.
                # Actually 2x2x1 means output is 2x1. So k has 1 value.
                # But k-block 2 has 3 values (4,5,6). 
                # I think I need to reconsider.
                # 
                # Let me re-examine: the output C is 2x7. 
                # For j-block jb, the relevant B entries are B[2*jb:2*jb+2, :].
                # The output C[:, k] for k in the k-block.
                # For k-block 2 (k=4,5,6), we have 3 output columns.
                # This is 2x2 (A) times 2x3 (B block) = 2x3 (C block).
                # That's not 2x2x1.
                #
                # I think I made an error. Let me reconsider the block decomposition.
                #
                # Actually, the standard approach:
                # C = A * B, where A is 2x4, B is 4x7.
                # Split A into A_L (2x2) and A_R (2x2) along the 4 columns.
                # Split B into B_L (2x7) and B_R (2x7) along the 4 rows.
                # C = A_L * B_L + A_R * B_R.
                #
                # A_L * B_L: 2x2 times 2x7.
                # Split B_L into 3 blocks: B_L1 (2x2), B_L2 (2x2), B_L3 (2x3).
                # A_L * B_L1: 2x2x2 -> Strassen (rank 7)
                # A_L * B_L2: 2x2x2 -> Strassen (rank 7)
                # A_L * B_L3: 2x2x3 -> need to handle
                #
                # For 2x2x3: 2x2 matrix times 2x3 matrix.
                # This is 3 independent 2x2x1 products? No.
                # It's computing 3 output columns, each from a 2x2 times 2x1.
                # Each 2x2x1 has rank 4.
                # But we can share the A factors.
                #
                # Actually, 2x2x3 can be done with rank 8 (2x4) or less?
                # Let me just use schoolbook for 2x2x3: rank 6? No, 2*2*3=12.
                # Or use the fact that it's 3 matvec products sharing A.
                #
                # This is getting complicated. Let me just use a simpler approach.
                #
                # Alternative: Use Strassen for the full 2x4x7 by splitting differently.
                # Or just use the schoolbook for the 2x2x3 part.
                #
                # For now, let me handle k-block 2 as 2x2x3 with schoolbook (rank 6 per j-block).
                # Total rank = 2*(7+7+6) = 40.
                
                # 2x2x3 schoolbook: 6 products
                # A = [[a0,a1],[a2,a3]], B = [[b0,b1,b2],[b3,b4,b5]]
                # C = [[c0,c1,c2],[c3,c4,c5]]
                # c0 = a0*b0 + a1*b3
                # c1 = a0*b1 + a1*b4
                # c2 = a0*b2 + a1*b5
                # c3 = a2*b0 + a3*b3
                # c4 = a2*b1 + a3*b4
                # c5 = a2*b2 + a3*b5
                # 6 products: a0*b0, a1*b3, a0*b1, a1*b4, a0*b2, a1*b5, a2*b0, a3*b3, a2*b1, a3*b4, a2*b2, a3*b5
                # That's 12 products. Schoolbook for 2x2x3 is rank 12.
                
                # Let me just use schoolbook for the 2x2x3 part.
                # Rank = 12 per j-block for k-block 2.
                # Total rank = 2*(7+7+12) = 52.
                
                # Hmm, that's only slightly better than 56.
                # Let me try a different approach.
                
                # Actually, for 2x2x3, we can do better.
                # Note that 2x2x3 = 2x2x2 + 2x2x1.
                # 2x2x2 has rank 7, 2x2x1 has rank 4.
                # But they share the A factors.
                # Total rank = 7 + 4 = 11? Not quite, because the B factors are different.
                #
                # Let me just implement it properly with the full 2x2x3 decomposition.
                # For simplicity, I'll use schoolbook (rank 12) for now.
                
                # Actually, let me reconsider the whole approach.
                # The cell finding says rank 50 is achievable.
                # 50 = 2*7 + 2*7 + 2*4 + 2*4? No, that's 40.
                # 50 = 2*(7+7+4) + 2*4? = 36+8 = 44.
                #
                # Let me just go with the simple block decomposition:
                # 2 j-blocks, each with:
                #   k-block 0 (2x2x2): Strassen rank 7
                #   k-block 1 (2x2x2): Strassen rank 7  
                #   k-block 2 (2x2x3): schoolbook rank 12
                # Total: 2*(7+7+12) = 52.
                
                # For 2x2x3 schoolbook:
                # 12 products, each a_ij * b_kl
                # U: 4 entries, V: 6 entries, W: 6 entries
                b_rank = 12
                u_map = [0*4 + (2*jb + 0), 0*4 + (2*jb + 1), 1*4 + (2*jb + 0), 1*4 + (2*jb + 1)]
                # V: j in {2*jb, 2*jb+1}, k in {4,5,6}
                # V global: j*7 + k
                v_map = [(2*jb)*7 + 4, (2*jb)*7 + 5, (2*jb)*7 + 6, (2*jb+1)*7 + 4, (2*jb+1)*7 + 5, (2*jb+1)*7 + 6]
                # W: k in {4,5,6}, i in {0,1}
                # W global: k*2 + i
                w_map = [4*2+0, 4*2+1, 5*2+0, 5*2+1, 6*2+0, 6*2+1]
                
                # 12 products:
                # P1 = a0*b0, P2 = a0*b1, P3 = a0*b2
                # P4 = a1*b3, P5 = a1*b4, P6 = a1*b5
                # P7 = a2*b0, P8 = a2*b1, P9 = a2*b2
                # P10 = a3*b3, P11 = a3*b4, P12 = a3*b5
                # c0 = P1 + P4, c1 = P2 + P5, c2 = P3 + P6
                # c3 = P7 + P10, c4 = P8 + P11, c5 = P9 + P12
                
                for r in range(b_rank):
                    # Determine which product this is
                    # r=0: a0*b0, r=1: a0*b1, r=2: a0*b2
                    # r=3: a1*b3, r=4: a1*b4, r=5: a1*b5
                    # r=6: a2*b0, r=7: a2*b1, r=8: a2*b2
                    # r=9: a3*b3, r=10: a3*b4, r=11: a3*b5
                    if r < 3:
                        U_full[u_map[0], col + r] = 1
                        V_full[v_map[r], col + r] = 1
                    elif r < 6:
                        U_full[u_map[1], col + r] = 1
                        V_full[v_map[r - 3 + 3], col + r] = 1  # v_map[3], v_map[4], v_map[5]
                    elif r < 9:
                        U_full[u_map[2], col + r] = 1
                        V_full[v_map[r - 6], col + r] = 1
                    else:
                        U_full[u_map[3], col + r] = 1
                        V_full[v_map[r - 9 + 3], col + r] = 1
                
                # W assignments
                # c0 = P1 + P4 -> W[0, 0] = 1, W[0, 3] = 1
                # c1 = P2 + P5 -> W[1, 1] = 1, W[1, 4] = 1
                # c2 = P3 + P6 -> W[2, 2] = 1, W[2, 5] = 1
                # c3 = P7 + P10 -> W[3, 6] = 1, W[3, 9] = 1
                # c4 = P8 + P11 -> W[4, 7] = 1, W[4, 10] = 1
                # c5 = P9 + P12 -> W[5, 8] = 1, W[5, 11] = 1
                W_full[w_map[0], col + 0] = 1
                W_full[w_map[0], col + 3] = 1
                W_full[w_map[1], col + 1] = 1
                W_full[w_map[1], col + 4] = 1
                W_full[w_map[2], col + 2] = 1
                W_full[w_map[2], col + 5] = 1
                W_full[w_map[3], col + 6] = 1
                W_full[w_map[3], col + 9] = 1
                W_full[w_map[4], col + 7] = 1
                W_full[w_map[4], col + 10] = 1
                W_full[w_map[5], col + 8] = 1
                W_full[w_map[5], col + 11] = 1
                
                col += b_rank
            col += b_rank
    
    # Verify the decomposition
    # Check that sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for all u,v,w
    error = 0.0
    for u in range(dim_u):
        for v in range(dim_v):
            for w in range(dim_w):
                val = np.dot(U_full[u, :] * V_full[v, :], W_full[w, :])
                # Compute T[u,v,w]
                t_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i*m+j == u and j*p+k == v and k*n+i == w:
                                t_val = 1.0
                error += (val - t_val) ** 2
    
    if error < 1e-10:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
