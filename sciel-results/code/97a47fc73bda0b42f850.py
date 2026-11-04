import numpy as np

def construct():
    n, m, p = 2, 3, 10
    # Rank calculation:
    # 5 blocks of 2x2 Strassen (rank 7 each) = 35
    # 1 block for remainder (rank 1) = 1
    # Total rank = 36
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strassen decomposition for 2x2 blocks
    # A1 is 2x2 (cols 0,1 of A), B1 is 2x10 (rows 0,1 of B)
    # Split B1 into 5 blocks of 2x2
    
    # Strassen formulas:
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # Indices in U for A1 (x11, x12, x21, x22):
    # x11: 0*3+0 = 0
    # x12: 0*3+1 = 1
    # x21: 1*3+0 = 3
    # x22: 1*3+1 = 4
    
    u_idx_x11 = 0
    u_idx_x12 = 1
    u_idx_x21 = 3
    u_idx_x22 = 4
    
    for t in range(5):
        # Indices in V for B1 block t (y11, y12, y21, y22):
        # y11: 0*10 + 2t = 2t
        # y12: 0*10 + 2t+1 = 2t+1
        # y21: 1*10 + 2t = 10+2t
        # y22: 1*10 + 2t+1 = 11+2t
        
        v_idx_y11 = 2 * t
        v_idx_y12 = 2 * t + 1
        v_idx_y21 = 10 + 2 * t
        v_idx_y22 = 11 + 2 * t
        
        # Indices in W for C block t (z11, z12, z21, z22):
        # z11: (2t)*2 + 0 = 4t
        # z12: (2t+1)*2 + 0 = 4t+1  (Wait, W index is k*n+i. C[0, 2t+1] -> k=2t+1, i=0 -> (2t+1)*2+0 = 4t+2? No.)
        
        # Let's re-verify W index.
        # W[k*n+i, r]. n=2.
        # C[i, k].
        # z11 is C[0, 2t]. i=0, k=2t. Index = 2t*2 + 0 = 4t.
        # z12 is C[0, 2t+1]. i=0, k=2t+1. Index = (2t+1)*2 + 0 = 4t+2.
        # z21 is C[1, 2t]. i=1, k=2t. Index = 2t*2 + 1 = 4t+1.
        # z22 is C[1, 2t+1]. i=1, k=2t+1. Index = (2t+1)*2 + 1 = 4t+3.
        
        w_idx_z11 = 4 * t
        w_idx_z12 = 4 * t + 2
        w_idx_z21 = 4 * t + 1
        w_idx_z22 = 4 * t + 3
        
        # Term M1
        U[u_idx_x11, col] += 1
        U[u_idx_x22, col] += 1
        V[v_idx_y11, col] += 1
        V[v_idx_y22, col] += 1
        W[w_idx_z11, col] += 1
        W[w_idx_z22, col] += 1
        col += 1
        
        # Term M2
        U[u_idx_x21, col] += 1
        U[u_idx_x22, col] += 1
        V[v_idx_y11, col] += 1
        W[w_idx_z21, col] += 1
        W[w_idx_z22, col] -= 1
        col += 1
        
        # Term M3
        U[u_idx_x11, col] += 1
        V[v_idx_y12, col] += 1
        V[v_idx_y22, col] -= 1
        W[w_idx_z12, col] += 1
        W[w_idx_z22, col] += 1
        col += 1
        
        # Term M4
        U[u_idx_x22, col] += 1
        V[v_idx_y21, col] += 1
        V[v_idx_y11, col] -= 1
        W[w_idx_z11, col] += 1
        W[w_idx_z21, col] += 1
        col += 1
        
        # Term M5
        U[u_idx_x11, col] += 1
        U[u_idx_x12, col] += 1
        V[v_idx_y22, col] += 1
        W[w_idx_z11, col] -= 1
        W[w_idx_z12, col] += 1
        col += 1
        
        # Term M6
        U[u_idx_x21, col] += 1
        U[u_idx_x11, col] -= 1
        V[v_idx_y11, col] += 1
        V[v_idx_y12, col] += 1
        W[w_idx_z22, col] += 1
        col += 1
        
        # Term M7
        U[u_idx_x12, col] += 1
        U[u_idx_x22, col] -= 1
        V[v_idx_y21, col] += 1
        V[v_idx_y22, col] += 1
        W[w_idx_z11, col] += 1
        col += 1
        
    # Term for a*b (j=2)
    # a = A[0,2], A[1,2]. Indices 2, 5.
    # b = B[2,0]...B[2,9]. Indices 20...29.
    # C[0,k], C[1,k]. W indices k*2+0, k*2+1.
    
    U[2, col] += 1
    U[5, col] += 1
    for k in range(10):
        V[20 + k, col] += 1
        W[k * 2 + 0, col] += 1
        W[k * 2 + 1, col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
