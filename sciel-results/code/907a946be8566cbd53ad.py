import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = 110.0  # safety margin before 120s limit

    n, m, p = 2, 3, 8
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        # B columns: 2*blk and 2*blk+1
        # C columns: 2*blk and 2*blk+1
        # W index for C[i, k] = k*n + i = 2k + i
        # For block blk: k0 = 2*blk, k1 = 2*blk+1
        # W indices: 2*k0+i = 4*blk+i, 2*k1+i = 4*blk+2+i
        
        k0 = 2 * blk
        k1 = 2 * blk + 1
        
        # Strassen for 2x2 * 2x2 (A[:,0:2] * B[0:2,:])
        # A_s = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_s = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, base+0] = 1   # A[0,0]
        U[4, base+0] = 1   # A[1,1] (index 1*3+1=4)
        V[k0, base+0] = 1       # B[0,k0]
        V[p+k1, base+0] = 1     # B[1,k1]
        W[2*k0+0, base+0] = 1   # C[0,k0]
        W[2*k1+1, base+0] = 1   # C[1,k1]
        
        # m2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3, base+1] = 1   # A[1,0] (index 1*3+0=3)
        U[4, base+1] = 1   # A[1,1]
        V[k0, base+1] = 1       # B[0,k0]
        W[2*k0+1, base+1] = 1   # C[1,k0]
        W[2*k1+1, base+1] = -1  # C[1,k1]
        
        # m3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0, base+2] = 1   # A[0,0]
        V[k1, base+2] = 1       # B[0,k1]
        V[p+k1, base+2] = -1    # B[1,k1]
        W[2*k0+1, base+2] = 0   # C[0,k1] -- wait, let me recheck Strassen
        # C[0,k1] = m3 + m5, C[1,k1] = m1 - m2 + m3 + m6
        W[2*k1+0, base+2] = 1   # C[0,k1]
        W[2*k1+1, base+2] = 1   # C[1,k1]
        
        # m4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4, base+3] = 1   # A[1,1]
        V[p+k0, base+3] = 1     # B[1,k0]
        V[k0, base+3] = -1      # B[0,k0]
        W[2*k0+0, base+3] = 1   # C[0,k0]
        W[2*k0+1, base+3] = 1   # C[1,k0]
        
        # m5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, base+4] = 1   # A[0,0]
        U[1, base+4] = 1   # A[0,1] (index 0*3+1=1)
        V[p+k1, base+4] = 1     # B[1,k1]
        W[2*k0+0, base+4] = -1  # C[0,k0]
        W[2*k1+0, base+4] = 1   # C[0,k1]
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3, base+5] = 1   # A[1,0]
        U[0, base+5] = -1  # A[0,0]
        V[k0, base+5] = 1       # B[0,k0]
        V[k1, base+5] = 1       # B[0,k1]
        W[2*k1+1, base+5] = 1   # C[1,k1]
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1, base+6] = 1   # A[0,1]
        U[4, base+6] = -1  # A[1,1]
        V[p+k0, base+6] = 1     # B[1,k0]
        V[p+k1, base+6] = 1     # B[1,k1]
        W[2*k0+0, base+6] = 1   # C[0,k0]
        
        # Extra: A[i,2] * B[2,k] for i in {0,1}, k in {k0,k1}
        # C[0,k0] += A[0,2]*B[2,k0]
        U[2, base+7] = 1   # A[0,2] (index 0*3+2=2)
        V[2*p+k0, base+7] = 1  # B[2,k0]
        W[2*k0+0, base+7] = 1  # C[0,k0]
        
        # C[0,k1] += A[0,2]*B[2,k1]
        U[2, base+8] = 1   # A[0,2]
        V[2*p+k1, base+8] = 1  # B[2,k1]
        W[2*k1+0, base+8] = 1  # C[0,k1]
        
        # C[1,k0] += A[1,2]*B[2,k0]
        U[5, base+9] = 1   # A[1,2] (index 1*3+2=5)
        V[2*p+k0, base+9] = 1  # B[2,k0]
        W[2*k0+1, base+9] = 1  # C[1,k0]
        
        # C[1,k1] += A[1,2]*B[2,k1]
        U[5, base+10] = 1  # A[1,2]
        V[2*p+k1, base+10] = 1  # B[2,k1]
        W[2*k1+1, base+10] = 1  # C[1,k1]
    
    # Verify correctness using vectorized approach
    def verify(U, V, W, n, m, p):
        # Check that sum_d U[a,d]*V[b,d]*W[c,d] = 1 if (i,j,k) consistent, else 0
        # a = i*m + j, b = j*p + k, c = k*n + i
        # For each (i,j), check all k pairs
        # Consistent: k2 == k -> should be 1
        # Inconsistent: k2 != k -> should be 0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                u_row = U[a, :]  # shape (rank,)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    v_row = V[b, :]
                    w_row = W[c, :]
                    # Consistent triple
                    total = np.dot(np.dot(u_row, v_row), w_row) if False else np.sum(u_row * v_row * w_row)
                    if abs(total - 1) > 1e-9:
                        return False
                    # Inconsistent triples
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        b2 = j * p + k2
                        v_row2 = V[b2, :]
                        total2 = np.sum(u_row * v_row2 * w_row)
                        if abs(total2) > 1e-9:
                            return False
        return True

    # Check deadline before verification
    if time.time() > _deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if not verify(U, V, W, n, m, p):
        # Fallback: return schoolbook
        total_rank_school = n * m * p
        U_s = np.zeros((n*m, total_rank_school))
        V_s = np.zeros((m*p, total_rank_school))
        W_s = np.zeros((p*n, total_rank_school))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, idx] = 1
                    V_s[j*p+k, idx] = 1
                    W_s[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
