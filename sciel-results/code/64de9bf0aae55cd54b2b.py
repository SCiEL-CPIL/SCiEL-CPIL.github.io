import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140

    # 1. Construct Rank 49 decomposition for 4x4 * 4x4 using Strassen (2x2 tensor product)
    
    # First, define the 2x2 Strassen decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: i,j in {0,1} for U; j,k in {0,1} for V; i,k in {0,1} for W
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen M1-M7 definitions
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Helper to set entries in 2x2 matrices
    # A indices: 00->0, 01->1, 10->2, 11->3 (row-major? i*m+j)
    # m=2.
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 2
    # i=1,j=1 -> 3
    
    def get_2x2_idx(i, j):
        return i * 2 + j

    # r=0: M1
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C11(1) ... wait, W is output C.
    # M1 contributes to C00 and C11.
    
    # Let's list coefficients for each of the 7 multiplications
    # Each multiplication is rank 1.
    # Term r has U_col, V_col, W_col
    
    # r=0
    U2[get_2x2_idx(0,0), 0] = 1
    U2[get_2x2_idx(1,1), 0] = 1
    V2[get_2x2_idx(0,0), 0] = 1
    V2[get_2x2_idx(1,1), 0] = 1
    W2[get_2x2_idx(0,0), 0] = 1
    W2[get_2x2_idx(1,1), 0] = 1
    
    # r=1
    U2[get_2x2_idx(1,0), 1] = 1
    U2[get_2x2_idx(1,1), 1] = 1
    V2[get_2x2_idx(0,0), 1] = 1
    W2[get_2x2_idx(1,0), 1] = 1
    W2[get_2x2_idx(1,1), 1] = -1 # C11 has -M2
    
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify M definitions vs C contributions
    # M1 = (A00+A11)(B00+B11)
    #   U: A00(1), A11(1)
    #   V: B00(1), B11(1)
    #   W: C00(1), C11(1) -- Wait, M1 is the product. It contributes to both?
    #   No, M1 is a scalar. In the tensor decomposition, each scalar is a rank 1 term.
    #   The term is $u \otimes v \otimes w$.
    #   If $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$, then the term is
    #   $(e_{00}+e_{11}) \otimes (f_{00}+f_{11}) \otimes (g_{00} + g_{11})$?
    #   No, the tensor is $\sum T_{ijk} e_i \otimes f_j \otimes g_k$.
    #   The term is $M_1 \cdot g_{?}$?
    #   Actually, the decomposition is $\sum_{r} u_r \otimes v_r \otimes w_r$.
    #   If we define $u_r, v_r$ such that $u_r \cdot A \cdot v_r^T = M_r$, then $w_r$ must be the vector that distributes $M_r$ to the output $C$.
    #   For M1, $u_r = e_{00} + e_{11}$ (row 0 col 0, row 1 col 1).
    #   $v_r = f_{00} + f_{11}$ (col 0 row 0, col 1 row 1).
    #   The output $C$ receives $M_1$ at $C_{00}$ and $C_{11}$.
    #   So $w_r = g_{00} + g_{11}$.
    
    # Let's refine r=0 (M1)
    U2[:, 0] = 0
    V2[:, 0] = 0
    W2[:, 0] = 0
    U2[get_2x2_idx(0,0), 0] = 1
    U2[get_2x2_idx(1,1), 0] = 1
    V2[get_2x2_idx(0,0), 0] = 1
    V2[get_2x2_idx(1,1), 0] = 1
    W2[get_2x2_idx(0,0), 0] = 1
    W2[get_2x2_idx(1,1), 0] = 1

    # r=1 (M2) = (A10+A11)B00
    U2[:, 1] = 0
    V2[:, 1] = 0
    W2[:, 1] = 0
    U2[get_2x2_idx(1,0), 1] = 1
    U2[get_2x2_idx(1,1), 1] = 1
    V2[get_2x2_idx(0,0), 1] = 1
    W2[get_2x2_idx(1,0), 1] = 1  # C10 += M2
    W2[get_2x2_idx(1,1), 1] = -1 # C11 -= M2

    # r=2 (M3) = A00(B10-B11)
    U2[:, 2] = 0
    V2[:, 2] = 0
    W2[:, 2] = 0
    U2[get_2x2_idx(0,0), 2] = 1
    V2[get_2x2_idx(1,0), 2] = 1
    V2[get_2x2_idx(1,1), 2] = -1
    W2[get_2x2_idx(0,1), 2] = 1 # C01 += M3
    W2[get_2x2_idx(1,1), 2] = 1 # C11 += M3

    # r=3 (M4) = A11(B01-B00)
    U2[:, 3] = 0
    V2[:, 3] = 0
    W2[:, 3] = 0
    U2[get_2x2_idx(1,1), 3] = 1
    V2[get_2x2_idx(0,1), 3] = 1
    V2[get_2x2_idx(0,0), 3] = -1
    W2[get_2x2_idx(0,0), 3] = 1 # C00 += M4
    W2[get_2x2_idx(1,0), 3] = 1 # C10 += M4

    # r=4 (M5) = (A00+A01)B11
    U2[:, 4] = 0
    V2[:, 4] = 0
    W2[:, 4] = 0
    U2[get_2x2_idx(0,0), 4] = 1
    U2[get_2x2_idx(0,1), 4] = 1
    V2[get_2x2_idx(1,1), 4] = 1
    W2[get_2x2_idx(0,0), 4] = -1 # C00 -= M5
    W2[get_2x2_idx(0,1), 4] = 1  # C01 += M5

    # r=5 (M6) = (A10-A00)(B00+B01)
    U2[:, 5] = 0
    V2[:, 5] = 0
    W2[:, 5] = 0
    U2[get_2x2_idx(1,0), 5] = 1
    U2[get_2x2_idx(0,0), 5] = -1
    V2[get_2x2_idx(0,0), 5] = 1
    V2[get_2x2_idx(0,1), 5] = 1
    W2[get_2x2_idx(1,1), 5] = 1 # C11 += M6

    # r=6 (M7) = (A01-A11)(B10+B11)
    U2[:, 6] = 0
    V2[:, 6] = 0
    W2[:, 6] = 0
    U2[get_2x2_idx(0,1), 6] = 1
    U2[get_2x2_idx(1,1), 6] = -1
    V2[get_2x2_idx(1,0), 6] = 1
    V2[get_2x2_idx(1,1), 6] = 1
    W2[get_2x2_idx(0,0), 6] = 1 # C00 += M7
    
    # Now, create the 4x4 decomposition by Kronecker product
    # The indices for 4x4 are (i1 i2, j1 j2).
    # Index = i1 * 4 + i2 * 2 + j2? No.
    # Standard row-major: i * m + j.
    # For 4x4, i in 0..3, j in 0..3.
    # We can map i = i1 * 2 + i2, j = j1 * 2 + j2.
    # Index U = i * 4 + j = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
    #           = i1 * 8 + i2 * 4 + j1 * 2 + j2
    # This is not a simple Kronecker product of indices if we treat the whole index as a vector.
    # But the tensor is the Kronecker product of the tensors.
    # $T_4 = T_2 \otimes T_2$.
    # If $T_2$ is represented by vectors $u_r, v_r, w_r$, then $T_4$ is represented by $u_r \otimes u_s, v_r \otimes v_s, w_r \otimes w_s$ for $r,s \in 0..6$.
    # The rank is $7 \times 7 = 49$.
    
    R4x4 = 49
    U4x4 = np.zeros((16, R4x4))
    V4x4 = np.zeros((16, R4x4)) # Wait, V is m*p = 4*4 = 16
    W4x4 = np.zeros((16, R4x4))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # u is u_r1 \otimes u_r2
            # Index mapping:
            # i1,i2 from U2 index.
            # U2 index is i*m+j with m=2.
            # So i = idx // 2, j = idx % 2.
            
            # Iterate over non-zero entries in U2[:, r1] and U2[:, r2]
            nz_u1 = np.nonzero(U2[:, r1])[0]
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v1 = np.nonzero(V2[:, r1])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w1 = np.nonzero(W2[:, r1])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            for idx1 in nz_u1:
                for idx2 in nz_u2:
                    i1 = idx1 // 2
                    j1 = idx1 % 2
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    global_idx_u = i * 4 + j
                    U4x4[global_idx_u, r] += U2[idx1, r1] * U2[idx2, r2]
            
            for idx1 in nz_v1:
                for idx2 in nz_v2:
                    j1 = idx1 // 2
                    k1 = idx1 % 2
                    j2 = idx2 // 2
                    k2 = idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    global_idx_v = j * 4 + k
                    V4x4[global_idx_v, r] += V2[idx1, r1] * V2[idx2, r2]
            
            for idx1 in nz_w1:
                for idx2 in nz_w2:
                    i1 = idx1 // 2
                    k1 = idx1 % 2
                    i2 = idx2 // 2
                    k2 = idx2 % 2
                    i = i1 * 2 + i2
                    k = k1 * 2 + k2
                    global_idx_w = k * 4 + i
                    W4x4[global_idx_w, r] += W2[idx1, r1] * W2[idx2, r2]
    
    # 2. Construct Schoolbook decomposition for 4x4 * 4x3
    # n=4, m=4, p=3 (local p)
    # Rank 48
    R4x3 = 48
    U4x3 = np.zeros((16, R4x3))
    V4x3 = np.zeros((12, R4x3))
    W4x3 = np.zeros((12, R4x3))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U4x3[i * 4 + j, col] = 1
                V4x3[j * 3 + k, col] = 1
                W4x3[k * 4 + i, col] = 1
                col += 1
    
    # 3. Combine
    R_total = R4x4 + R4x3
    U_total = np.zeros((16, R_total))
    V_total = np.zeros((28, R_total))
    W_total = np.zeros((28, R_total))
    
    # First part: 4x4 * 4x4
    # U: 16x49
    U_total[:, 0:49] = U4x4
    # V: 16x49 -> place in first 16 rows of V_total (indices j*4+k for j,k in 0..3)
    # Global V index is j * 7 + k.
    # For the 4x4 part, k is 0..3.
    # So we need to scatter V4x4 into V_total.
    for j in range(4):
        for k in range(4):
            global_idx = j * 7 + k
            V_total[global_idx, 0:49] = V4x4[j * 4 + k, :]
            
    # W: 16x49 -> place in first 16 rows of W_total (indices k*4+i for k,i in 0..3)
    # Global W index is k * 4 + i.
    # For the 4x4 part, k is 0..3.
    # These map directly to global indices 0..15.
    W_total[0:16, 0:49] = W4x4
    
    # Second part: 4x4 * 4x3
    # U: 16x48
    U_total[:, 49:49+48] = U4x3
    # V: 12x48 -> place in last 12 "slots" of V for each j.
    # Local V index j*3+k for k in 0..2.
    # Global V index j*7 + (4 + k) ??
    # No, B is 4x7. We split into B_left (4x4) and B_right (4x3).
    # B_right has columns 4,5,6.
    # So for the right part, k_global = 4 + k_local.
    for j in range(4):
        for k_local in range(3):
            k_global = 4 + k_local
            global_idx = j * 7 + k_global
            V_total[global_idx, 49:49+48] = V4x3[j * 3 + k_local, :]
            
    # W: 12x48 -> place in last 12 rows of W_total.
    # Local W index k*4+i for k in 0..2.
    # Global W index (4 + k)*4 + i.
    for k_local in range(3):
        k_global = 4 + k_local
        for i in range(4):
            global_idx = k_global * 4 + i
            local_idx = k_local * 4 + i
            W_total[global_idx, 49:49+48] = W4x3[local_idx, :]
    
    # 4. Verify
    def verify(U, V, W):
        target = set()
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    target.add((i * 4 + j, j * 7 + k, k * 4 + i))
        # Note: Target indices for U, V, W are based on the full problem dimensions.
        # U idx: i*m+j
        # V idx: j*p+k
        # W idx: k*n+i
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all target entries
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False
        # Check all non-target entries
        for u in range(16):
            for v in range(28):
                for w in range(28):
                    if (u, v, w) in target:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True
    
    if verify(U_total.tolist(), V_total.tolist(), W_total.tolist()):
        return {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_sb = 112
        U_sb = np.zeros((16, rank_sb))
        V_sb = np.zeros((28, rank_sb))
        W_sb = np.zeros((28, rank_sb))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_sb[i * 4 + j, col] = 1
                    V_sb[j * 7 + k, col] = 1
                    W_sb[k * 4 + i, col] = 1
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
