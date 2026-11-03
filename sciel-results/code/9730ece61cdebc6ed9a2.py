import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 38 solution
    # Split m=4 into 2+2.
    # Block 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x5)
    # Block 2: A[:, 2:4] * B[2:4, :] (2x2 * 2x5)
    
    # For each 2x2x5 block:
    # Split p=5 into 2+3.
    # Part 1: 2x2x2 (Strassen, Rank 7)
    # Part 2: 2x2x3 (Schoolbook, Rank 12)
    # Total per block: 19.
    # Total Rank: 38.
    
    rank = 38
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(u_base, v_base, w_base, c):
        # u_base: index offset for U (i*m + j)
        # v_base: index offset for V (j*p + k)
        # w_base: index offset for W (k*n + i)
        # Note: Strassen is for A (2x2), B (2x2), C (2x2).
        # Local indices:
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # Strassen terms:
        # M1 = (A22 - A21)(B11 + B12) -> C11, C22
        # M2 = (A21 + A22)B21 -> C21, C22
        # M3 = (A11 - A21)(B21 + B22) -> C12, C22
        # M4 = A11(B12 - B11) -> C11
        # M5 = A22(B22 - B21) -> C11, C21
        # M6 = (A11 + A12)B11 -> C12
        # M7 = (A12 - A11)(B11 + B21) -> C11
        
        # U indices (i,j):
        # A11: 0, A12: 1, A21: 2, A22: 3
        # V indices (j,k):
        # B11: 0, B12: 1, B21: 2, B22: 3
        # W indices (k,i) -> C[i,k]:
        # C11: 0, C12: 1, C21: 2, C22: 3
        # Note: W index is k*n + i.
        # C[0,0] -> k=0, i=0 -> 0
        # C[0,1] -> k=1, i=0 -> 1
        # C[1,0] -> k=0, i=1 -> 2 (Wait, k*n+i. n=2. k=0, i=1 -> 1? No. k=0, i=1 -> 0*2+1=1? No.
        # Let's check W index definition: W[k*n + i, r].
        # C[i,k].
        # If i=0, k=0: idx 0*2+0 = 0.
        # If i=0, k=1: idx 1*2+0 = 2.
        # If i=1, k=0: idx 0*2+1 = 1.
        # If i=1, k=1: idx 1*2+1 = 3.
        # So W index for C[i,k] is k*n + i.
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1
            ({3: 1, 2: -1}, {0: 1, 1: 1}, {0: 1, 3: 1}), # C11(0), C22(3)
            # M2
            ({2: 1, 3: 1}, {2: 1}, {2: 1, 3: 1}), # C21(2), C22(3)
            # M3
            ({0: 1, 2: -1}, {2: 1, 3: 1}, {1: 1, 3: 1}), # C12(1), C22(3)
            # M4
            ({0: 1}, {1: 1, 0: -1}, {0: 1}), # C11(0)
            # M5
            ({3: 1}, {3: 1, 2: -1}, {0: -1, 2: 1}), # C11(0), C21(2)
            # M6
            ({0: 1, 1: 1}, {0: 1}, {1: 1}), # C12(1)
            # M7
            ({1: 1, 0: -1}, {0: 1, 2: 1}, {0: 1}) # C11(0)
        ]
        
        for u_c, v_c, w_c in terms:
            for idx, val in u_c.items():
                i, j = idx // 2, idx % 2
                U[u_base + i * m + j, c] += val
            for idx, val in v_c.items():
                j, k = idx // 2, idx % 2
                V[v_base + j * p + k, c] += val
            for idx, val in w_c.items():
                # idx is local C index.
                # Local C indices: 0:C11, 1:C12, 2:C21, 3:C22
                # C11: i=0, k=0. W_idx = 0*2+0 = 0.
                # C12: i=0, k=1. W_idx = 1*2+0 = 2.
                # C21: i=1, k=0. W_idx = 0*2+1 = 1.
                # C22: i=1, k=1. W_idx = 1*2+1 = 3.
                if idx == 0: w_idx = 0
                elif idx == 1: w_idx = 2
                elif idx == 2: w_idx = 1
                elif idx == 3: w_idx = 3
                W[w_base + w_idx, c] += val
            c += 1
        return c

    def add_schoolbook_2x2x3(u_base, v_base, w_base, c):
        # 2x2x3 schoolbook
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[u_base + i * m + j, c] = 1
                    V[v_base + j * p + k, c] = 1
                    # W index for C[i, k]
                    # k is local 0,1,2. Global k = k_base + k.
                    # We need to pass k_base.
                    # Let's modify signature to include k_base.
                    pass
        return c

    # Refactor add_schoolbook to take k_base
    def add_schoolbook_2x2x3_v2(u_base, v_base, w_base, k_base, c):
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[u_base + i * m + j, c] = 1
                    V[v_base + j * p + (k_base + k), c] = 1
                    # W index: (k_base + k) * n + i
                    W[(k_base + k) * n + i, c] = 1
                    c += 1
        return c

    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in 0,1; j in 0,1.
    # B indices: j in 0,1; k in 0,4.
    # U base: 0 (i*m + j)
    # V base: 0 (j*p + k)
    # W base: 0 (k*n + i)
    
    # Split B into k=0,1 and k=2,3,4.
    # Strassen for k=0,1
    col = add_strassen(0, 0, 0, col)
    # Schoolbook for k=2,3,4
    col = add_schoolbook_2x2x3_v2(0, 0, 0, 2, col)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A indices: i in 0,1; j in 2,3.
    # B indices: j in 2,3; k in 0,4.
    # U base: i*m + j. j is 2,3. So base is 0*m + 2 = 2?
    # U index = i*4 + j.
    # For j=2, U index = i*4 + 2.
    # For j=3, U index = i*4 + 3.
    # My add_strassen uses local j 0,1.
    # So u_base should be 2?
    # U[u_base + i*m + j_local].
    # If u_base = 2, then U[2 + i*4 + j_local].
    # If i=0, j_local=0 -> U[2] (which is A[0,2]). Correct.
    # If i=0, j_local=1 -> U[3] (which is A[0,3]). Correct.
    # If i=1, j_local=0 -> U[6] (which is A[1,2]). Correct.
    
    # V base: j*p + k.
    # j is 2,3.
    # V index = j*5 + k.
    # My add_strassen uses local j 0,1.
    # So v_base should be 2*p = 10?
    # V[v_base + j_local*p + k].
    # If v_base = 10, j_local=0 -> V[10+k] (B[2,k]). Correct.
    # If j_local=1 -> V[15+k] (B[3,k]). Correct.
    
    # W base: k*n + i.
    # W index is independent of j.
    # So w_base = 0.
    
    col = add_strassen(2, 10, 0, col)
    col = add_schoolbook_2x2x3_v2(2, 10, 0, 2, col)
    
    return {"U": U, "V": V, "W": W}
