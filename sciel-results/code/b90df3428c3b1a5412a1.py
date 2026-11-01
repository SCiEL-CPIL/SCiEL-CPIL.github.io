import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the known rank-39 decomposition
    # 5 blocks of Strassen (rank 7) for 10 columns
    # 1 block of Schoolbook (rank 4) for the last column
    rank_fallback = 5 * 7 + 1 * 4
    U_fallback = np.zeros((n * m, rank_fallback), dtype=np.int64)
    V_fallback = np.zeros((m * p, rank_fallback), dtype=np.int64)
    W_fallback = np.zeros((p * n, rank_fallback), dtype=np.int64)
    
    r = 0
    # Strassen decomposition for 2x2 * 2x2 blocks
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Indices in V (B matrix)
        # B is 2x11. Row 0, Col k -> index k. Row 1, Col k -> index 11 + k.
        v_e = k0      # B[0, k0]
        v_f = k1      # B[0, k1]
        v_g = 11 + k0 # B[1, k0]
        v_h = 11 + k1 # B[1, k1]
        
        # Indices in W (C matrix)
        # C is 2x11. Row i, Col k -> index k*n + i.
        # n=2.
        # C[0, k0] -> 2*k0 + 0
        # C[1, k0] -> 2*k0 + 1
        # C[0, k1] -> 2*k1 + 0
        # C[1, k1] -> 2*k1 + 1
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # Indices in U (A matrix)
        # A is 2x2.
        # a=0, b=1, c=2, d=3
        
        # m1 = (a+d)(e+h)
        U_fallback[0, r] = 1; U_fallback[3, r] = 1
        V_fallback[v_e, r] = 1; V_fallback[v_h, r] = 1
        W_fallback[w_c00, r] = 1; W_fallback[w_c11, r] = 1
        r += 1
        
        # m2 = (c+d)e
        U_fallback[2, r] = 1; U_fallback[3, r] = 1
        V_fallback[v_e, r] = 1
        W_fallback[w_c10, r] = 1; W_fallback[w_c11, r] = -1
        r += 1
        
        # m3 = a(f-h)
        U_fallback[0, r] = 1
        V_fallback[v_f, r] = 1; V_fallback[v_h, r] = -1
        W_fallback[w_c01, r] = 1; W_fallback[w_c11, r] = 1
        r += 1
        
        # m4 = d(g-e)
        U_fallback[3, r] = 1
        V_fallback[v_g, r] = 1; V_fallback[v_e, r] = -1
        W_fallback[w_c00, r] = 1; W_fallback[w_c10, r] = 1
        r += 1
        
        # m5 = (a+b)h
        U_fallback[0, r] = 1; U_fallback[1, r] = 1
        V_fallback[v_h, r] = 1
        W_fallback[w_c00, r] = -1; W_fallback[w_c01, r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U_fallback[2, r] = 1; U_fallback[0, r] = -1
        V_fallback[v_e, r] = 1; V_fallback[v_f, r] = 1
        W_fallback[w_c11, r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U_fallback[1, r] = 1; U_fallback[3, r] = -1
        V_fallback[v_g, r] = 1; V_fallback[v_h, r] = 1
        W_fallback[w_c00, r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # C[0, 10] = a*e2 + b*g2
    U_fallback[0, r] = 1
    V_fallback[v_e2, r] = 1
    W_fallback[w_c02, r] = 1
    r += 1
    
    U_fallback[1, r] = 1
    V_fallback[v_g2, r] = 1
    W_fallback[w_c02, r] = 1
    r += 1
    
    # C[1, 10] = c*e2 + d*g2
    U_fallback[2, r] = 1
    V_fallback[v_e2, r] = 1
    W_fallback[w_c12, r] = 1
    r += 1
    
    U_fallback[3, r] = 1
    V_fallback[v_g2, r] = 1
    W_fallback[w_c12, r] = 1
    r += 1
    
    # Verification step (optional, but good for safety)
    # Check if U @ diag(1) @ V^T matches W for the tensor
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] ? No.
    # The tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r
    # So T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here indices are:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    
    # Let's verify one element to ensure correctness
    # T[0, 0, 0] should be 1 (A[0,0]B[0,0])
    # i=0, j=0, k=0
    # U index: 0*2+0 = 0
    # V index: 0*11+0 = 0
    # W index: 0*2+0 = 0
    # Sum_r U[0, r] V[0, r] W[0, r]
    
    # In our construction:
    # Block 0, k0=0, k1=1
    # m1: U[0]=1, V[0]=1, W[0]=1 -> 1
    # m2: U[0]=0, ...
    # m3: U[0]=1, V[1]=1, W[2]=1 -> V[0] is 0 for m3?
    # m3: V_fallback[v_f, r] = 1 -> v_f = 1. V[0] is 0.
    # m4: U[0]=0
    # m5: U[0]=1, V[11+1]=1, W[0]=-1 -> V[0] is 0.
    # m6: U[0]=-1, V[0]=1, W[3]=1 -> W[0] is 0.
    # m7: U[0]=0
    # So sum is 1. Correct.
    
    return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
