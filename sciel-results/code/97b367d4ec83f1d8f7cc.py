import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen decomposition for 2x2 matrix multiplication
    # U_local: (4, 7), V_local: (4, 7), W_local: (4, 7)
    # Local indices: 0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1 = (a+d)(e+h))
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    
    # r=1 (M2 = (c+d)e)
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1
    
    # r=2 (M3 = a(f-h))
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    
    # r=3 (M4 = d(g-c))
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    
    # r=4 (M5 = (a+b)h)
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    
    # r=5 (M6 = (c-a)(e+f))
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # r=6 (M7 = (b-d)(g+h))
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # 8 blocks: j in 0..3 (column pairs of B/C), s in 0..1 (row block of A/B)
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # A mapping: local 2x2 index -> global A flat index
            # A is 2x4, flat index = row*4 + col
            # s=0: cols 0,1; s=1: cols 2,3
            a_map = [0]*4
            for la in range(4):
                row = la // 2
                col = la % 2
                global_col = s * 2 + col
                a_map[la] = row * 4 + global_col
            
            # B mapping: local 2x2 index -> global B flat index
            # B is 4x8, flat index = row*8 + col
            # s=0: rows 0,1; s=1: rows 2,3
            # cols: 2j, 2j+1
            b_map = [0]*4
            for lb in range(4):
                row = lb // 2
                col = lb % 2
                global_row = s * 2 + row
                global_col = 2 * j + col
                b_map[lb] = global_row * 8 + global_col
            
            # C mapping: local 2x2 index -> global W flat index
            # W convention: W[k*n+i, r] where k=col, i=row
            # C is 2x8, so W index = col*2 + row
            c_map = [0]*4
            for lc in range(4):
                row = lc // 2
                col = lc % 2
                global_row = row
                global_col = 2 * j + col
                c_map[lc] = global_col * 2 + global_row
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    # Verify correctness with a fixed seed for reproducibility
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_recon = np.zeros((n, p))
    for r in range(total_rank):
        u_val = U[:, r] @ A_flat
        v_val = V[:, r] @ B_flat
        for i in range(n):
            for j in range(p):
                c_idx = j * n + i
                C_recon[i, j] += W[c_idx, r] * u_val * v_val
    
    valid = np.allclose(C_recon, C_expected, atol=1e-10)
    
    if not valid:
        # Fallback: return schoolbook decomposition (rank 64)
        U_school = np.zeros((n * m, n * m * p), dtype=int)
        V_school = np.zeros((m * p, n * m * p), dtype=int)
        W_school = np.zeros((p * n, n * m * p), dtype=int)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * m * p + k * p + j
                    U_school[i * m + k, r] = 1
                    V_school[k * p + j, r] = 1
                    W_school[j * n + i, r] = 1
        
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    if time.time() - start_time > 140:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
