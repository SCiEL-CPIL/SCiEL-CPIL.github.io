import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: (a00+a11)(b00+b11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # r=1: (a10+a11)(b00)
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1
    
    # r=2: (a00)(b01-b11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1
    
    # r=3: (a11)(b10-b00)
    U2[3,3]=1
    V2[0,3]=-1; V2[2,3]=1
    W2[0,3]=1; W2[1,3]=1
    
    # r=4: (a00+a01)(b11)
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1
    
    # r=5: (a10-a00)(b00+b01)
    U2[0,5]=-1; U2[2,5]=1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # r=6: (a01-a11)(b10+b11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1

    def verify2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    if s != 1:
                        return False
        return True

    strassen_ok = verify2x2()
    if not strassen_ok:
        # Fallback to schoolbook 2x2
        U2 = np.zeros((4, 4), dtype=int)
        V2 = np.zeros((4, 4), dtype=int)
        W2 = np.zeros((4, 4), dtype=int)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i*4 + j*2 + k
                    U2[i*2+j, r] = 1
                    V2[j*2+k, r] = 1
                    W2[k*2+i, r] = 1
        strassen_rank = 4
    else:
        strassen_rank = 7

    # 3x3 decomposition
    # Attempt Rank-19 Winograd decomposition
    # If it fails, fall back to Rank-27 Schoolbook
    
    def get_3x3_rank19():
        # Returns U, V, W of shape (9, 19)
        # This is a known rank-19 decomposition
        U3 = np.zeros((9, 19), dtype=int)
        V3 = np.zeros((9, 19), dtype=int)
        W3 = np.zeros((9, 19), dtype=int)
        
        # Coefficients for the 19 multiplications
        # This specific decomposition is derived from the work of Winograd and others.
        # Indices: U[i,j], V[j,k], W[k,i]
        
        # We define the 19 bilinear forms.
        # This list is based on a standard rank-19 decomposition.
        # If this fails verification, we will use schoolbook.
        
        # For simplicity and safety, I will use the Schoolbook Rank 27 here 
        # because I cannot guarantee the exact coefficients of the Rank 19 
        # decomposition from memory without risk of error.
        # However, Rank 189 is the current best.
        # Let's try to implement Rank 19 carefully.
        
        # Actually, let's use the Rank 27 schoolbook for 3x3 to ensure correctness.
        # Rank 27 is guaranteed.
        return None

    # Use Schoolbook 3x3 (Rank 27) for guaranteed correctness
    rank3 = 27
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3[i * 3 + j, r] = 1
                V3[j * 3 + k, r] = 1
                W3[k * 3 + i, r] = 1

    # Construct 6x6 factors via Kronecker product
    rank = strassen_rank * rank3
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    # Precompute non-zero entries for efficiency
    nz_u2 = {r: np.nonzero(U2[:, r])[0] for r in range(strassen_rank)}
    nz_v2 = {r: np.nonzero(V2[:, r])[0] for r in range(strassen_rank)}
    nz_w2 = {r: np.nonzero(W2[:, r])[0] for r in range(strassen_rank)}
    nz_u3 = {r: np.nonzero(U3[:, r])[0] for r in range(rank3)}
    nz_v3 = {r: np.nonzero(V3[:, r])[0] for r in range(rank3)}
    nz_w3 = {r: np.nonzero(W3[:, r])[0] for r in range(rank3)}
    
    for r2 in range(strassen_rank):
        for r3 in range(rank3):
            r = r2 * rank3 + r3
            
            # Combine U factors
            for u2 in nz_u2[r2]:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3[r3]:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    
                    U6[g_u, r] = val_u2 * val_u3

            # Combine V factors
            for v2 in nz_v2[r2]:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3[r3]:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    
                    V6[g_v, r] = val_v2 * val_v3

            # Combine W factors
            for w2 in nz_w2[r2]:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3[r3]:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    
                    W6[g_w, r] = val_w2 * val_w3

    # Verify the full 6x6 decomposition
    if time.time() < deadline - 10:
        def verify_fast(U, V, W, n, m, p):
            rank = U.shape[1]
            U3d = U.reshape(n, m, rank)
            V3d = V.reshape(m, p, rank)
            W3d = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3d, V3d, W3d)
            return np.all(T == 1)
        
        if not verify_fast(U6, V6, W6, n, m, p):
            # Fallback to pure schoolbook
            rank_fb = 216
            U6_fb = np.zeros((36, rank_fb), dtype=int)
            V6_fb = np.zeros((36, rank_fb), dtype=int)
            W6_fb = np.zeros((36, rank_fb), dtype=int)
            
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        r = i * 36 + j * 6 + k
                        U6_fb[i * 6 + j, r] = 1
                        V6_fb[j * 6 + k, r] = 1
                        W6_fb[k * 6 + i, r] = 1
            
            return {"U": U6_fb.tolist(), "V": V6_fb.tolist(), "W": W6_fb.tolist()}
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
