import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for P = XY
    # M1 = (X00+X11)(Y00+Y11) -> P00 += 1, P11 += 1
    # M2 = (X10+X11)Y00       -> P10 += 1
    # M3 = X00(Y01-Y11)       -> P01 += 1, P11 += -1
    # M4 = X11(Y10-Y00)       -> P00 += 1, P10 += 1
    # M5 = (X00+X01)Y11       -> P00 += -1, P01 += 1
    # M6 = (X10-X00)(Y00+Y01) -> P11 += 1
    # M7 = (X01-X11)(Y10+Y11) -> P00 += 1, P11 += -1
    
    # Let's define the Strassen steps explicitly as (U_indices, V_indices, W_indices, W_coeff)
    # X is 2x2, Y is 2x2.
    # X indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Y indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (X0+X3)(Y0+Y3) -> P0+=1, P3+=1
    # M2: (X2+X3)Y0       -> P2+=1
    # M3: X0(Y1-Y3)       -> P1+=1, P3+=-1
    # M4: X3(Y2-Y0)       -> P0+=1, P2+=1
    # M5: (X0+X1)Y3       -> P0+=-1, P1+=1
    # M6: (X2-X0)(Y0+Y1)  -> P3+=1
    # M7: (X1-X3)(Y2+Y3)  -> P0+=1, P3+=-1
    
    strassen_steps = [
        # (u_idx, v_idx, [(w_idx, coeff), ...])
        ([0, 3], [0, 3], [(0, 1), (3, 1)]),
        ([2, 3], [0], [(2, 1)]),
        ([0], [1, 3], [(1, 1), (3, -1)]),
        ([3], [2, 0], [(0, 1), (2, 1)]),
        ([0, 1], [3], [(0, -1), (1, 1)]),
        ([2, 0], [0, 1], [(3, 1)]),
        ([1, 3], [2, 3], [(0, 1), (3, -1)])
    ]
    
    # For each block k in 0..3
    for k in range(4):
        # Block k of C is at columns 2k, 2k+1
        # C indices for this block:
        # C[0, 2k] is p*n index: (2k)*n + 0 = 2k
        # C[0, 2k+1] is p*n index: (2k+1)*n + 0 = 2k+1
        # C[1, 2k] is p*n index: (2k)*n + 1 = 2k+1? No.
        # W index is k*n + i.
        # i=0, k=2k -> 2k*n + 0 = 4k
        # i=0, k=2k+1 -> (2k+1)*n + 0 = 4k+2
        # i=1, k=2k -> 2k*n + 1 = 4k+1
        # i=1, k=2k+1 -> (2k+1)*n + 1 = 4k+3
        
        c_w_indices = [4*k, 4*k+2, 4*k+1, 4*k+3] # Corresponds to P0, P1, P2, P3
        
        # Part 1: A11 * B1k
        # A11 indices in U:
        # A[0,0] -> 0*4+0 = 0
        # A[0,1] -> 0*4+1 = 1
        # A[1,0] -> 1*4+0 = 4
        # A[1,1] -> 1*4+1 = 5
        a11_u_indices = [0, 1, 4, 5]
        
        # B1k indices in V:
        # B rows 0,1. Cols 2k, 2k+1.
        # B[0, 2k]   -> 0*8 + 2k   = 2k
        # B[0, 2k+1] -> 0*8 + 2k+1 = 2k+1
        # B[1, 2k]   -> 1*8 + 2k   = 8+2k
        # B[1, 2k+1] -> 1*8 + 2k+1 = 8+2k+1
        b1k_v_indices = [2*k, 2*k+1, 8+2*k, 8+2*k+1]
        
        # Part 2: A12 * B2k
        # A12 indices in U:
        # A[0,2] -> 0*4+2 = 2
        # A[0,3] -> 0*4+3 = 3
        # A[1,2] -> 1*4+2 = 6
        # A[1,3] -> 1*4+3 = 7
        a12_u_indices = [2, 3, 6, 7]
        
        # B2k indices in V:
        # B rows 2,3. Cols 2k, 2k+1.
        # B[2, 2k]   -> 2*8 + 2k   = 16+2k
        # B[2, 2k+1] -> 2*8 + 2k+1 = 16+2k+1
        # B[3, 2k]   -> 3*8 + 2k   = 24+2k
        # B[3, 2k+1] -> 3*8 + 2k+1 = 24+2k+1
        b2k_v_indices = [16+2*k, 16+2*k+1, 24+2*k, 24+2*k+1]
        
        # Fill U, V, W for these 14 terms
        for s in range(7):
            # Term for A11 * B1k
            col = k * 14 + s
            u_idx_local, v_idx_local, w_terms = strassen_steps[s]
            for u_l in u_idx_local:
                U[a11_u_indices[u_l]][col] = 1
            for v_l in v_idx_local:
                V[b1k_v_indices[v_l]][col] = 1
            for w_l, coeff in w_terms:
                W[c_w_indices[w_l]][col] = coeff
                
            # Term for A12 * B2k
            col = k * 14 + s + 7
            u_idx_local, v_idx_local, w_terms = strassen_steps[s]
            for u_l in u_idx_local:
                U[a12_u_indices[u_l]][col] = 1
            for v_l in v_idx_local:
                V[b2k_v_indices[v_l]][col] = 1
            for w_l, coeff in w_terms:
                W[c_w_indices[w_l]][col] = coeff

    return {"U": U, "V": V, "W": W}
