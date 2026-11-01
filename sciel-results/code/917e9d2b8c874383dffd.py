"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110  # 110s budget, leave margin for safety
    
    n, m, p = DIMENSIONS
    # Use Strassen for each 2x2 block of B, giving rank 4*7=28
    num_blocks = p // 2  # 4 blocks of 2x2
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen coefficients for 2x2 x 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    
    # For each block b (0..3), B block is B[:, 2b:2b+2]
    # A entries: (0,0)=a idx 0, (0,1)=b idx 1, (1,0)=c idx 2, (1,1)=d idx 3
    # B block entries: (0,2b)=e idx 0*p+2b, (0,2b+1)=f idx 0*p+2b+1
    #                 (1,2b)=g idx 1*p+2b, (1,2b+1)=h idx 1*p+2b+1
    # C entries: (0,2b) idx 2b*n+0, (0,2b+1) idx (2b+1)*n+0
    #           (1,2b) idx 2b*n+1, (1,2b+1) idx (2b+1)*n+1
    
    for b in range(num_blocks):
        base = b * 7
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1  # A[0,0]
        U[3][base + 0] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 0] += 1  # B[0,2b]
        V[1 * p + 2 * b + 1][base + 0] += 1  # B[1,2b+1]
        # C[0,2b] += M1, C[1,2b+1] += M1
        W[2 * b * n + 0][base + 0] += 1  # C[0,2b]
        W[(2 * b + 1) * n + 1][base + 0] += 1  # C[1,2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1  # A[1,0]
        U[3][base + 1] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 1] += 1  # B[0,2b]
        # C[1,2b] += M2, C[1,2b+1] -= M2
        W[2 * b * n + 1][base + 1] += 1  # C[1,2b]
        W[(2 * b + 1) * n + 1][base + 1] -= 1  # C[1,2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1  # A[0,0]
        V[0 * p + 2 * b + 1][base + 2] += 1  # B[0,2b+1]
        V[1 * p + 2 * b + 1][base + 2] -= 1  # B[1,2b+1]
        # C[0,2b+1] += M3, C[1,2b+1] += M3
        W[(2 * b + 1) * n + 0][base + 2] += 1  # C[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 2] += 1  # C[1,2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1  # A[1,1]
        V[1 * p + 2 * b][base + 3] += 1  # B[1,2b]
        V[0 * p + 2 * b][base + 3] -= 1  # B[0,2b]
        # C[0,2b] += M4, C[1,2b] += M4
        W[2 * b * n + 0][base + 3] += 1  # C[0,2b]
        W[2 * b * n + 1][base + 3] += 1  # C[1,2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1  # A[0,0]
        U[1][base + 4] += 1  # A[0,1]
        V[1 * p + 2 * b + 1][base + 4] += 1  # B[1,2b+1]
        # C[0,2b] -= M5, C[0,2b+1] += M5
        W[2 * b * n + 0][base + 4] -= 1  # C[0,2b]
        W[(2 * b + 1) * n + 0][base + 4] += 1  # C[0,2b+1]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1  # A[1,0]
        U[0][base + 5] -= 1  # A[0,0]
        V[0 * p + 2 * b][base + 5] += 1  # B[0,2b]
        V[0 * p + 2 * b + 1][base + 5] += 1  # B[0,2b+1]
        # C[1,2b+1] += M6
        W[(2 * b + 1) * n + 1][base + 5] += 1  # C[1,2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1  # A[0,1]
        U[3][base + 6] -= 1  # A[1,1]
        V[1 * p + 2 * b][base + 6] += 1  # B[1,2b]
        V[1 * p + 2 * b + 1][base + 6] += 1  # B[1,2b+1]
        # C[0,2b] += M7
        W[2 * b * n + 0][base + 6] += 1  # C[0,2b]
    
    # Verify correctness with numpy
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with multiple random matrices for robustness
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test  # shape (n, p) = (2, 8)
            
            A_flat = A_test.flatten()  # row-major: [A[0,0], A[0,1], A[1,0], A[1,1]]
            B_flat = B_test.flatten()  # row-major: [B[0,0], ..., B[0,7], B[1,0], ..., B[1,7]]
            
            C_computed = np.zeros(p * n)  # indexed as k*n+i
            for r in range(rank):
                u_r = U_np[:, r]
                v_r = V_np[:, r]
                w_r = W_np[:, r]
                prod_u = np.dot(u_r, A_flat)
                prod_v = np.dot(v_r, B_flat)
                C_computed += w_r * prod_u * prod_v
            
            # C_computed is indexed as k*n+i, reshape to (p, n) then transpose to (n, p)
            C_computed_reshaped = C_computed.reshape(p, n).T
            
            max_err = np.max(np.abs(C_computed_reshaped - C_expected))
            if max_err > 1e-10:
                # Fallback: schoolbook rank 32
                rank_s = n * m * p
                U_s = [[0] * rank_s for _ in range(n * m)]
                V_s = [[0] * rank_s for _ in range(m * p)]
                W_s = [[0] * rank_s for _ in range(p * n)]
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_s[i * m + j][idx] = 1
                            V_s[j * p + k][idx] = 1
                            W_s[k * n + i][idx] = 1
                            idx += 1
                return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
