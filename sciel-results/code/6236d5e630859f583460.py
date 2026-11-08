import numpy as np
import time

DIMENSIONS = [2, 3, 9]
DEADLINE = 115.0  # 5s buffer before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 50 initial decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # 4 blocks of 2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[k0 * n + 1][col+1] += 1
        W[k1 * n + 1][col+1] -= 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        W[k1 * n + 0][col+2] += 1
        W[k1 * n + 1][col+2] += 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        W[k0 * n + 0][col+3] += 1
        W[k0 * n + 1][col+3] += 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        W[k0 * n + 0][col+4] -= 1
        W[k1 * n + 0][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        W[k1 * n + 1][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        W[k0 * n + 0][col+6] += 1
        
        col += 7
    
    # Part 1b: Last column k=8 for A_L * B_T
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # Direct: C08 = a*e + b*g, C18 = c*e + d*g
    U[u_a][col] += 1
    V[v_e][col] += 1
    W[k8 * n + 0][col] += 1
    
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    W[k8 * n + 0][col+1] += 1
    
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    W[k8 * n + 1][col+2] += 1
    
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    W[k8 * n + 1][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    def verify_fast(U, V, W, n, m, p):
        """Fast verification using NumPy"""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        
        U_3d = U_np.reshape(n, m, -1)  # (n, m, R)
        V_3d = V_np.reshape(m, p, -1)  # (m, p, R)
        W_3d = W_np.reshape(p, n, -1)  # (p, n, R)
        
        # Check all (i, j, k) triples
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify initial decomposition
    if not verify_fast(U, V, W, n, m, p):
        raise ValueError("Initial decomposition verification failed")
    
    # Column removal optimization
    # Strategy: try removing each column, check if still valid
    # Order columns by how "essential" they are (fewer non-zero entries = less essential)
    
    def column_weight(U, V, W, c):
        """Count non-zero entries in column c across all three matrices"""
        w = 0
        for row in U:
            if row[c] != 0:
                w += 1
        for row in V:
            if row[c] != 0:
                w += 1
        for row in W:
            if row[c] != 0:
                w += 1
        return w
    
    current_time = time.time()
    removed_any = True
    while removed_any and (time.time() - start_time) < DEADLINE - 5:
        removed_any = False
        
        # Sort columns by weight (ascending - try removing lighter columns first)
        col_weights = [(column_weight(U, V, W, c), c) for c in range(rank)]
        col_weights.sort()
        
        for weight, col_to_remove in col_weights:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Try removing this column
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            
            if verify_fast(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                break
    
    return {"U": U, "V": V, "W": W}
