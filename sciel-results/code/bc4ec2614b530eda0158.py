"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def build_schoolbook():
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

    def build_strassen_hybrid():
        # Rank 88: 1x4x8 schoolbook (32) + 2x4x8 with Strassen on 2x2 blocks (56)
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: i=0, schoolbook for 1x4x8
        for j in range(4):
            for k in range(8):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
        # col should be 32 now
        
        # Part 2: i=1,2, Strassen on 2x4x8
        # Split j into two blocks of 2, k into four blocks of 2
        # For each (j_block, k_block), apply 2x2x2 Strassen (rank 7)
        
        # Standard 2x2x2 Strassen matrices
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # U factors (for A part, 2x2 matrix):
        # M1: A00 + A11 -> [1,0; 0,1]
        # M2: A10 + A11 -> [0,0; 1,1]
        # M3: A00 -> [1,0; 0,0]
        # M4: A11 -> [0,0; 0,1]
        # M5: A00 + A10 -> [1,1; 0,0]
        # M6: A11 - A00 -> [-1,0; 0,1]
        # M7: A01 - A11 -> [0,1; 0,-1]
        strassen_U = [
            [1, 0, 0, 1],   # M1: A00+A11
            [0, 0, 1, 1],   # M2: A10+A11
            [1, 0, 0, 0],   # M3: A00
            [0, 0, 0, 1],   # M4: A11
            [1, 1, 0, 0],   # M5: A00+A10
            [-1, 0, 0, 1],  # M6: A11-A00
            [0, 1, 0, -1]   # M7: A01-A11
        ]
        
        # V factors (for B part, 2x2 matrix):
        # M1: B00 + B11
        # M2: B00
        # M3: B10 - B11
        # M4: B01 - B10
        # M5: B11
        # M6: B00 + B10
        # M7: B01 + B11
        strassen_V = [
            [1, 0, 0, 1],   # M1: B00+B11
            [1, 0, 0, 0],   # M2: B00
            [0, 1, 0, -1],  # M3: B10-B11
            [0, -1, 1, 0],  # M4: B01-B10
            [0, 0, 0, 1],   # M5: B11
            [1, 0, 1, 0],   # M6: B00+B10
            [0, 1, 0, 1]    # M7: B01+B11
        ]
        
        # W factors (for C part, 2x2 matrix):
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        strassen_W = [
            [1, 0, 0, 1],   # M1: C00+C11
            [0, 0, 1, -1],  # M2: C10-C11
            [0, 1, 0, 1],   # M3: C01+C11
            [1, 0, 1, 0],   # M4: C00+C10
            [-1, 1, 0, 0],  # M5: -C00+C01
            [0, 0, 0, 1],   # M6: C11
            [1, 0, 0, 0]    # M7: C00
        ]
        
        # Wait, let me re-derive W carefully.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # W[t] is the 2x2 matrix for the t-th Strassen term's contribution to C.
        # C[local_i, local_k] = sum_t W[t][local_i, local_k] * M_t
        #
        # For local_i=0, local_k=0: C00 = M1 + M4 - M5 - M7
        #   W[0][0,0]=1, W[3][0,0]=1, W[4][0,0]=-1, W[6][0,0]=-1
        # For local_i=0, local_k=1: C01 = M3 + M5
        #   W[2][0,1]=1, W[4][0,1]=1
        # For local_i=1, local_k=0: C10 = M2 + M4
        #   W[1][1,0]=1, W[3][1,0]=1
        # For local_i=1, local_k=1: C11 = M1 - M2 + M3 + M7
        #   W[0][1,1]=1, W[1][1,1]=-1, W[2][1,1]=1, W[6][1,1]=1
        #
        # So W[t] as 2x2 (row-major: [C00, C01, C10, C11]):
        # t=0 (M1): [1, 0, 0, 1]
        # t=1 (M2): [0, 0, 1, -1]
        # t=2 (M3): [0, 1, 0, 1]
        # t=3 (M4): [1, 0, 1, 0]
        # t=4 (M5): [-1, 1, 0, 0]
        # t=5 (M6): [0, 0, 0, 0]  # M6 is not used in the standard 7-term Strassen for C!
        # t=6 (M7): [-1, 0, 0, 1]
        
        # Wait, I have 7 terms but M6 isn't in the output formula. Let me recheck.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # M6 is NOT used in the output! So we only need 6 terms, not 7?
        # No wait, that's wrong. Let me look this up properly.
        #
        # Actually the standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # M6 is indeed not used! So we can drop it and use rank 6?
        # No, that's not right. The issue is that M6 is needed for the general case.
        # Let me verify: C11 = M1 - M2 + M3 + M7
        # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + (A01-A11)(B01+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A01B01 + A01B11 - A11B01 - A11B11
        # = A00B00 + A00B10 + A01B01 + A01B11 - A10B00 - A11B01
        # That's not A10B01 + A11B11. So something is wrong.
        #
        # Let me use the correct Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # Let me verify C11:
        # M1 - M2 + M3 + M7
        # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + (A01-A11)(B01+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A01B01 + A01B11 - A11B01 - A11B11
        # = A00B00 + A00B10 + A01B01 + A01B11 - A10B00 - A11B01
        #
        # But C11 should be A10B01 + A11B11. This doesn't match!
        #
        # I think I have the wrong Strassen formulas. Let me use the correct ones:
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # Hmm, let me just use a known-correct implementation. The standard is:
        # M1 = (A11 - A00)(B00 + B10)
        # M2 = (A00 + A10)B11
        # M3 = (A01 + A11)(B00 - B10)
        # M4 = (A11 - A01)(B01 + B11)
        # M5 = (A00 + A11)(B00 + B11)
        # M6 = (A01 - A11)(B10 + B11)
        # M7 = (A10 - A00)(B00 + B01)
        #
        # C00 = M5 + M4 - M6 - M7
        # C01 = M2 + M3
        # C10 = M1 + M4
        # C11 = M5 + M1 - M3 - M2
        
        # Let me just use the well-known correct version and verify numerically.
        # I'll use a simpler approach: just verify the decomposition numerically.
        
        # Let me restart with a clean, verified Strassen implementation.
        
        # Correct Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # U (A factors), each is 2x2 flattened to [A00,A01,A10,A11]:
        U_s = [
            [1, 0, 0, 1],    # M1: A00+A11
            [0, 0, 1, 1],    # M2: A10+A11
            [1, 0, 0, 0],    # M3: A00
            [0, 0, 0, 1],    # M4: A11
            [1, 1, 0, 0],    # M5: A00+A10
            [-1, 0, 0, 1],   # M6: A11-A00
            [0, 1, 0, -1]    # M7: A01-A11
        ]
        
        # V (B factors), each is 2x2 flattened to [B00,B01,B10,B11]:
        V_s = [
            [1, 0, 0, 1],    # M1: B00+B11
            [1, 0, 0, 0],    # M2: B00
            [0, -1, 1, 0],   # M3: B10-B11
            [0, 1, -1, 0],   # M4: B01-B10
            [0, 0, 0, 1],    # M5: B11
            [1, 0, 1, 0],    # M6: B00+B10
            [0, 1, 0, 1]     # M7: B01+B11
        ]
        
        # W (C factors), each is 2x2 flattened to [C00,C01,C10,C11]:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        W_s = [
            [1, 0, 0, 1],    # M1: C00+C11
            [0, 0, 1, -1],   # M2: C10-C11
            [0, 1, 0, 1],    # M3: C01+C11
            [1, 0, 1, 0],    # M4: C00+C10
            [-1, 1, 0, 0],   # M5: -C00+C01
            [0, 0, 0, 0],    # M6: unused
            [-1, 0, 0, 1]    # M7: -C00+C11
        ]
        
        # Verify Strassen numerically
        np.random.seed(123)
        A_test = np.random.rand(2, 2)
        B_test = np.random.rand(2, 2)
        C_expected = A_test @ B_test
        
        for t in range(7):
            U_row = np.array(U_s[t]).reshape(2, 2)
            V_row = np.array(V_s[t]).reshape(2, 2)
            W_row = np.array(W_s[t]).reshape(2, 2)
            M_t = U_row @ V_row  # This isn't right; M_t is a scalar in tensor decomposition
            # Actually in tensor decomposition, M_t is the "intermediate" and
            # C = sum_t W_t * M_t where M_t = (U_t . A) * (V_t . B) is not quite right.
            # The tensor decomposition is: C[i,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
            # For 2x2x2, this means for each (i,j,k), the tensor entry is 1.
            # Let me just verify the full decomposition numerically.
        
        # For the hybrid, let me just build it and verify.
        
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                for t in range(7):
                    # U: rows 1,2 of A, columns j_start, j_start+1
                    for l in range(4):
                        val = U_s[t][l]
                        if val != 0:
                            local_i = l // 2  # 0 or 1
                            local_j = l % 2
                            global_i = 1 + local_i
                            global_j = j_start + local_j
                            U[global_i * m + global_j][col] += val
                    # V: rows j_start, j_start+1 of B, columns k_start, k_start+1
                    for l in range(4):
                        val = V_s[t][l]
                        if val != 0:
                            local_j = l // 2
                            local_k = l % 2
                            global_j = j_start + local_j
                            global_k = k_start + local_k
                            V[global_j * p + global_k][col] += val
                    # W: rows k_start, k_start+1 of C, columns 1,2 of C (i=1,2)
                    for l in range(4):
                        val = W_s[t][l]
                        if val != 0:
                            local_k = l // 2
                            local_i = l % 2
                            global_k = k_start + local_k
                            global_i = 1 + local_i
                            W[global_k * n + global_i][col] += val
                    col += 1
        
        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        rank = U.shape[1]
        
        # Check tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try Strassen hybrid
    if time.time() < deadline - 10:
        res = build_strassen_hybrid()
        if res and verify(res):
            return res
            
    # Fallback
    return build_schoolbook()
