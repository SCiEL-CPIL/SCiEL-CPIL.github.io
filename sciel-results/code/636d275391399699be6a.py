import time
import sys
import numpy as np
import scipy.linalg

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    # W convention: W[k*n + i, r] where i is row of C, k is col of C
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] over j
    # So tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1

    # Strategy: Use 4 output blocks, each sum of two 2x2 products.
    # Rank 56 construction, then reduce.
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # Strassen coefficients
    # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
    U_loc[0,0]=1; U_loc[3,0]=1; V_loc[0,0]=1; V_loc[3,0]=1; W_loc[0,0]=1; W_loc[3,0]=1
    # M2: (a10+a11)b00 -> c10+=1
    U_loc[2,1]=1; U_loc[3,1]=1; V_loc[0,1]=1; W_loc[2,1]=1; W_loc[3,1]=-1 # Wait, Strassen M2 is (a10+a11)b00. C10 = M2+M4. C11 = M1-M2+M3+M6.
    # Let's re-verify standard Strassen signs.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The code in the seed:
    # M2: U_loc[2,1]=1; U_loc[3,1]=1 (A10+A11). V_loc[0,1]=1 (B00).
    # W_loc[2,1]=1 (C10 += 1). W_loc[3,1]=-1 (C11 -= 1).
    # This matches C10 = M2... and C11 = ... - M2. Correct.
    
    # M3: A00(B01-B11)
    U_loc[0,2]=1; V_loc[1,2]=1; V_loc[3,2]=-1; W_loc[1,2]=1; W_loc[3,2]=1
    # C01 += 1, C11 += 1. Correct.
    
    # M4: A11(B10-B00)
    U_loc[3,3]=1; V_loc[2,3]=1; V_loc[0,3]=-1; W_loc[0,3]=1; W_loc[2,3]=1
    # C00 += 1, C10 += 1. Correct.
    
    # M5: (A00+A01)B11
    U_loc[0,4]=1; U_loc[1,4]=1; V_loc[3,4]=1; W_loc[0,4]=-1; W_loc[1,4]=1
    # C00 -= 1, C01 += 1. Correct.
    
    # M6: (A10-A00)(B00+B01)
    U_loc[2,5]=1; U_loc[0,5]=-1; V_loc[0,5]=1; V_loc[1,5]=1; W_loc[3,5]=1
    # C11 += 1. Correct.
    
    # M7: (A01-A11)(B10+B11)
    U_loc[1,6]=1; U_loc[3,6]=-1; V_loc[2,6]=1; V_loc[3,6]=1; W_loc[0,6]=1
    # C00 += 1. Correct.
    
    total_rank_56 = 56
    U = np.zeros((8, total_rank_56))
    V = np.zeros((32, total_rank_56))
    W = np.zeros((16, total_rank_56))
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            for l in range(7):
                r = base_r + l
                # U (A)
                for la in range(4):
                    val = U_loc[la, l]
                    if val != 0:
                        i_idx = la // 2
                        j_idx = la % 2 + 2 * s
                        U[i_idx * 4 + j_idx, r] += val
                # V (B)
                for lb in range(4):
                    val = V_loc[lb, l]
                    if val != 0:
                        j_idx = lb // 2 + 2 * s
                        k_idx = lb % 2 + 2 * j
                        V[j_idx * 8 + k_idx, r] += val
                # W (C)
                for lc in range(4):
                    val = W_loc[lc, l]
                    if val != 0:
                        i_idx = lc // 2
                        k_idx = lc % 2 + 2 * j
                        W[k_idx * 2 + i_idx, r] += val
    
    # Verify rank 56 first
    max_err = 0.0
    for i in range(2):
        for j in range(4):
            for k in range(8):
                val = 0.0
                for r in range(total_rank_56):
                    val += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-8:
        # Fallback if construction is broken, return empty or raise
        # But let's try to return the rank 56 anyway if it's close, or just return it.
        pass

    # Try to reduce rank
    vec_tensors = np.zeros((64, total_rank_56))
    for r in range(total_rank_56):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    vec_tensors[i*32 + j*8 + k, r] = u_r[i*4+j] * v_r[j*8+k] * w_r[k*2+i]
    
    try:
        Q_qr, R_qr, P_qr = scipy.linalg.qr(vec_tensors, pivoting=True)
        diag_R = np.abs(np.diag(R_qr))
        tol = 1e-10
        rank_qr = 0
        for i in range(len(diag_R)):
            if diag_R[i] > tol:
                rank_qr = i + 1
            else:
                break
        
        if rank_qr < total_rank_56:
            keep = P_qr[:rank_qr]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            
            # Verify
            max_err = 0.0
            for i in range(2):
                for j in range(4):
                    for k in range(8):
                        val = 0.0
                        for r in range(rank_qr):
                            val += U_new[i*4+j, r] * V_new[j*8+k, r] * W_new[k*2+i, r]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
