import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 4, 8
    
    # Strassen 2x2x2x2 decomposition
    # U_local: (4,7), V_local: (4,7), W_local: (4,7)
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0: M1=(a+d)(e+h)
    U_local[0,0]=1; U_local[3,0]=1
    V_local[0,0]=1; V_local[3,0]=1
    W_local[0,0]=1; W_local[3,0]=1
    
    # r=1: M2=(c+d)e
    U_local[2,1]=1; U_local[3,1]=1
    V_local[0,1]=1
    W_local[2,1]=1; W_local[3,1]=-1
    
    # r=2: M3=a(f-h)
    U_local[0,2]=1
    V_local[1,2]=1; V_local[3,2]=-1
    W_local[1,2]=1; W_local[3,2]=1
    
    # r=3: M4=d(g-c)
    U_local[3,3]=1
    V_local[2,3]=1; V_local[0,3]=-1
    W_local[0,3]=1; W_local[2,3]=1
    
    # r=4: M5=(a+b)h
    U_local[0,4]=1; U_local[1,4]=1
    V_local[3,4]=1
    W_local[0,4]=-1; W_local[1,4]=1
    
    # r=5: M6=(c-a)(e+f)
    U_local[2,5]=1; U_local[0,5]=-1
    V_local[0,5]=1; V_local[1,5]=1
    W_local[3,5]=1
    
    # r=6: M7=(b-d)(g+h)
    U_local[1,6]=1; U_local[3,6]=-1
    V_local[2,6]=1; V_local[3,6]=1
    W_local[0,6]=1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # 8 blocks: j in 0..3 (col pairs of B/C), s in 0..1 (row block of A/B)
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # A: 2x4, flat=row*4+col. s selects col pair (2s, 2s+1)
            a_map = [0]*4
            for la in range(4):
                row = la // 2
                col = la % 2
                a_map[la] = row * 4 + (s * 2 + col)
            
            # B: 4x8, flat=row*8+col. s selects row pair (2s, 2s+1), j selects col pair (2j, 2j+1)
            b_map = [0]*4
            for lb in range(4):
                row = lb // 2
                col = lb % 2
                b_map[lb] = (s * 2 + row) * 8 + (2 * j + col)
            
            # W convention: W[k*n+i, r] where k=col, i=row. C is 2x8.
            c_map = [0]*4
            for lc in range(4):
                row = lc // 2
                col = lc % 2
                c_map[lc] = (2 * j + col) * 2 + row
            
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
    
    # Verify
    if time.time() - start_time < deadline - 5:
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
            # Fallback: schoolbook rank 64
            U_s = np.zeros((n*m, n*m*p), dtype=int)
            V_s = np.zeros((m*p, n*m*p), dtype=int)
            W_s = np.zeros((p*n, n*m*p), dtype=int)
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        r = i*m*p + k*p + j
                        U_s[i*m+k, r] = 1
                        V_s[k*p+j, r] = 1
                        W_s[j*n+i, r] = 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
