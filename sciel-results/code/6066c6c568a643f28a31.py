import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 140  # Safety margin from 150s

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Helper to check if a decomposition is valid (Vectorized)
    def verify(U, V, W, rank):
        if time.time() > DEADLINE:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to 3D tensors
        # U: (n*m, rank) -> (n, m, rank)
        U_3d = U_arr.reshape(n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        V_3d = V_arr.reshape(m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # We want to compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # This is a contraction over r.
        # Let's do it step by step to avoid huge intermediate tensors if possible, 
        # but n,m,p are small (3,4,6), so (3,4,6) is tiny.
        
        # Product U[i,j,r] * V[j,k,r]
        # U_3d shape: (n, m, rank)
        # V_3d shape: (m, p, rank)
        # We need to broadcast U to (n, m, p, rank) and V to (n, m, p, rank)?
        # U[i, j, r] * V[j, k, r]
        # U_expanded = U_3d[:, :, np.newaxis, :] # (n, m, 1, rank)
        # V_expanded = V_3d[np.newaxis, :, :, :] # (1, m, p, rank)
        # Prod_UV = U_expanded * V_expanded # (n, m, p, rank)
        # Sum over r: Prod_UV.sum(axis=3) -> (n, m, p)
        
        # Then multiply by W[k, i, r]?
        # Wait, the formula is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # So we need to sum over r of the product of all three.
        
        # Let's construct the full tensor product and sum.
        # U_3d: (n, m, rank)
        # V_3d: (m, p, rank)
        # W_3d: (p, n, rank)
        
        # Reshape to allow broadcasting
        U_b = U_3d[:, :, np.newaxis, :] # (n, m, 1, rank)
        V_b = V_3d[np.newaxis, :, :, :] # (1, m, p, rank)
        W_b = W_3d[:, np.newaxis, np.newaxis, :] # (p, 1, 1, rank)
        
        # We need to align indices:
        # U: i, j
        # V: j, k
        # W: k, i
        
        # The product is U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Let's broadcast to (n, m, p, rank)
        # U_b is (n, m, 1, rank)
        # V_b is (1, m, p, rank) -> need to broadcast m? 
        # U_b[i, j, :, r] * V_b[:, j, k, r] -> This works if we align j.
        # Result shape (n, m, p, rank)
        # Then multiply by W_b[k, i, r].
        # W_b shape (p, 1, 1, rank). We need to permute it to match (n, m, p, rank) dimensions?
        # W[k, i, r] depends on k and i.
        # So we need to broadcast W to (n, m, p, rank) such that:
        # W_broadcast[i, j, k, r] = W[k, i, r]
        
        # Let's use einsum or explicit reshaping.
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        # Step 1: P[i, j, k, r] = U[i, j, r] * V[j, k, r]
        # U: (n, m, rank) -> (n, m, 1, rank)
        # V: (m, p, rank) -> (1, m, p, rank)
        P = U_3d[:, :, np.newaxis, :] * V_3d[np.newaxis, :, :, :]
        # P shape: (n, m, p, rank)
        
        # Step 2: Multiply by W[k, i, r]
        # W: (p, n, rank)
        # We need to broadcast W to (n, m, p, rank)
        # W_b shape: (n, 1, p, rank) ? No.
        # W[k, i, r] -> for fixed i, k, r, it is a scalar.
        # We want W_broadcast[i, j, k, r] = W[k, i, r]
        # W_reshaped = W_3d.T # (n, p, rank) ? No, W_3d is (p, n, rank).
        # W_3d[k, i, r].
        # We want to access this as W_broadcast[i, :, k, r].
        # So we need to swap axes of W_3d to (n, p, rank) and then expand m dim.
        W_swapped = W_3d.swapaxes(0, 1) # (n, p, rank)
        W_b = W_swapped[:, np.newaxis, :, :] # (n, 1, p, rank)
        
        # Now P is (n, m, p, rank)
        # W_b is (n, 1, p, rank)
        # Product: P * W_b -> (n, m, p, rank)
        # Sum over r:
        T_calc = (P * W_b).sum(axis=3)
        
        # T_calc should be all 1s for valid (i, j, k)
        # The tensor for matrix multiplication M(n,m,p) has T[i,j,k] = 1 for all i,j,k.
        
        if not np.allclose(T_calc, 1.0, atol=1e-10):
            return False
            
        return True

    # Schoolbook decomposition for M(n,m,p) with rank n*m*p
    def schoolbook_decomp(nn, mm, pp):
        rank = nn * mm * pp
        U = [[0.0]*rank for _ in range(nn*mm)]
        V = [[0.0]*rank for _ in range(mm*pp)]
        W = [[0.0]*rank for _ in range(pp*nn)]
        r = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i*mm+j][r] = 1.0
                    V[j*pp+k][r] = 1.0
                    W[k*nn+i][r] = 1.0
                    r += 1
        return U, V, W, rank
    
    # Strassen for M(2,2,2) with rank 7
    def strassen_222():
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
        # M2 = (a10+a11)(b00) -> c10, c11
        # M3 = (a00)(b01-b11) -> c01, c11
        # M4 = (a11)(b10-b00) -> c00, c10
        # M5 = (a00+a01)(b11) -> c00, c01
        # M6 = (a10-a00)(b00+b01) -> c11
        # M7 = (a01-a11)(b10+b11) -> c00
        
        # U[i*2+j], V[j*2+k], W[k*2+i]
        
        # M1: U[0,2],[2,3]; V[0,3]; W[0,2]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: U[2,3]; V[0]; W[2,3]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3: U[0]; V[1,3]; W[1,3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4: U[3]; V[2,0]; W[0,2]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: U[0,1]; V[3]; W[0,1]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6: U[2,0]; V[0,1]; W[3]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: U[1,3]; V[2,3]; W[0]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W, 7
    
    # M(1,m,p) schoolbook: rank = m*p
    def m1_mp(mm, pp):
        rank = mm * pp
        U = [[0.0]*rank for _ in range(1*mm)]
        V = [[0.0]*rank for _ in range(mm*pp)]
        W = [[0.0]*rank for _ in range(pp*1)]
        r = 0
        for j in range(mm):
            for k in range(pp):
                U[j][r] = 1.0
                V[j*pp+k][r] = 1.0
                W[k][r] = 1.0
                r += 1
        return U, V, W, rank

    def build_66():
        nn, mm, pp = 3, 4, 6
        total_rank = 66
        U = [[0.0]*total_rank for _ in range(nn*mm)]
        V = [[0.0]*total_rank for _ in range(mm*pp)]
        W = [[0.0]*total_rank for _ in range(pp*nn)]
        current = [0]
        
        def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
            r = current[0]
            U[u_idx][r] += u_val
            V[v_idx][r] += v_val
            W[w_idx][r] += w_val
            current[0] += 1
            
        def add_1_4_2(i_off, j_off, k_off):
            # Adds M(1,4,2) decomposition
            # i is fixed to i_off
            # j ranges j_off to j_off+3
            # k ranges k_off to k_off+1
            for j in range(4):
                for k in range(2):
                    add_rank1(i_off*4 + (j_off+j), 1.0, 
                             (j_off+j)*6 + (k_off+k), 1.0,
                             (k_off+k)*3 + i_off, 1.0)
                             
        def add_2_2_2(i_off, j_off, k_off):
            # Adds M(2,2,2) decomposition using Strassen
            U_s, V_s, W_s, _ = strassen_222()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        idx_u = (i_off+i)*4 + (j_off+j)
                        U[idx_u][current[0]+r] += U_s[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        idx_v = (j_off+j)*6 + (k_off+k)
                        V[idx_v][current[0]+r] += V_s[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        idx_w = (k_off+k)*3 + (i_off+i)
                        W[idx_w][current[0]+r] += W_s[k*2+i][r]
            current[0] += 7
            
        def add_2_4_2(i_off, j_off, k_off):
            # Adds M(2,4,2) decomposition
            # Split j into two blocks of 2
            add_2_2_2(i_off, j_off, k_off)
            add_2_2_2(i_off, j_off+2, k_off)
            
        # Split p=6 into three blocks of 2
        for block_k in range(3):
            k_off = block_k * 2
            # i=0 block: M(1,4,2)
            add_1_4_2(0, 0, k_off)
            # i=1,2 block: M(2,4,2)
            add_2_4_2(1, 0, k_off)
            
        return U, V, W, total_rank

    # Generate candidates
    candidates = []
    
    # Candidate 1: Known rank 66
    U1, V1, W1, r1 = build_66()
    candidates.append((r1, U1, V1, W1, "66_block"))
    
    # Verify it
    if verify(U1, V1, W1, r1):
        print(f"Candidate 66 verified")
    else:
        print(f"Candidate 66 FAILED verification")
        # Debugging: print max error
        U_arr = np.array(U1, dtype=float)
        V_arr = np.array(V1, dtype=float)
        W_arr = np.array(W1, dtype=float)
        U_3d = U_arr.reshape(3, 4, r1)
        V_3d = V_arr.reshape(4, 6, r1)
        W_3d = W_arr.reshape(6, 3, r1)
        
        P = U_3d[:, :, np.newaxis, :] * V_3d[np.newaxis, :, :, :]
        W_swapped = W_3d.swapaxes(0, 1)
        W_b = W_swapped[:, np.newaxis, :, :]
        T_calc = (P * W_b).sum(axis=3)
        print(f"Max deviation: {np.max(np.abs(T_calc - 1.0))}")
        print(f"T_calc:\n{T_calc}")
        
    # Try schoolbook: rank = 3*4*6 = 72
    U2, V2, W2, r2 = schoolbook_decomp(3, 4, 6)
    candidates.append((r2, U2, V2, W2, "schoolbook"))
    
    # Sort by rank
    candidates.sort(key=lambda x: x[0])
    
    best = None
    for rank, U, V, W, name in candidates:
        if time.time() > DEADLINE:
            break
        if verify(U, V, W, rank):
            best = (rank, U, V, W)
            print(f"Verified {name} with rank {rank}")
            break
    
    if best is None:
        # Fallback to schoolbook
        best = (72, U2, V2, W2)
        print("Falling back to schoolbook rank 72")
        
    return {"U": best[1], "V": best[2], "W": best[3]}
