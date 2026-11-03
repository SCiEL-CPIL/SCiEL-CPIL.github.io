"""Matrix multiplication tensor decomposition for 2x3x8."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Build target tensor
    M = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1.0

    # Find a rank-7 decomposition for 2x2x2 using ALS
    M2 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                M2[i*2+j, j*2+k, k*2+i] = 1.0
    
    np.random.seed(42)
    U2 = np.random.randn(4, 7)
    V2 = np.random.randn(4, 7)
    W2 = np.random.randn(4, 7)
    
    for it in range(300):
        if time.time() > deadline - 20:
            break
        for r in range(7):
            O_r = np.outer(V2[:,r], W2[:,r])
            R = np.zeros((4,4,4))
            for r2 in range(7):
                if r2 != r:
                    for a in range(4):
                        R[a,:,:] += U2[a,r2] * np.outer(V2[:,r2], W2[:,r2])
            for a in range(4):
                residual = M2[a,:,:] - R[a,:,:]
                denom = np.sum(O_r * O_r)
                U2[a,r] = np.sum(residual * O_r) / denom if denom != 0 else 0
        
        for r in range(7):
            P_r = np.outer(U2[:,r], W2[:,r])
            S = np.zeros((4,4,4))
            for r2 in range(7):
                if r2 != r:
                    for b in range(4):
                        S[:,b,:] += V2[b,r2] * np.outer(U2[:,r2], W2[:,r2])
            for b in range(4):
                residual = M2[:,b,:] - S[:,b,:]
                denom = np.sum(P_r * P_r)
                V2[b,r] = np.sum(residual * P_r) / denom if denom != 0 else 0
        
        for r in range(7):
            P_r = np.outer(U2[:,r], V2[:,r])
            S = np.zeros((4,4,4))
            for r2 in range(7):
                if r2 != r:
                    for c in range(4):
                        S[:,:,c] += W2[c,r2] * np.outer(U2[:,r2], V2[:,r2])
            for c in range(4):
                residual = M2[:,:,c] - S[:,:,c]
                denom = np.sum(P_r * P_r)
                W2[c,r] = np.sum(residual * P_r) / denom if denom != 0 else 0
    
    # Check if we found a valid rank-7 decomposition
    M2_recon = np.zeros((4,4,4))
    for r in range(7):
        M2_recon += np.einsum('a,b,c->abc', U2[:,r], V2[:,r], W2[:,r])
    
    if np.allclose(M2_recon, M2, atol=1e-6):
        rank_total = 16 + 28
        U = np.zeros((n*m, rank_total))
        V = np.zeros((m*p, rank_total))
        W = np.zeros((p*n, rank_total))
        
        # Part 1: schoolbook 2x1x8
        for i in range(n):
            for k in range(p):
                col = i*p + k
                U[i*m+0, col] = 1
                V[0*p+k, col] = 1
                W[k*n+i, col] = 1
        
        # Part 2: Strassen 2x2x8
        for b in range(4):
            for t in range(7):
                col = 16 + b*7 + t
                for li in range(2):
                    for lj in range(2):
                        a_local = li*2 + lj
                        u_global = li*3 + lj + 1
                        U[u_global, col] = U2[a_local, t]
                
                for lj in range(2):
                    for lk in range(2):
                        b_local = lj*2 + lk
                        v_global = (lj+1)*8 + (2*b + lk)
                        V[v_global, col] = V2[b_local, t]
                
                for li in range(2):
                    for lk in range(2):
                        c_local = lk*2 + li
                        w_global = (2*b + lk)*2 + li
                        W[w_global, col] = W2[c_local, t]
        
        # Verify full decomposition
        M_full_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank_total):
            M_full_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        if np.allclose(M_full_recon, M, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
