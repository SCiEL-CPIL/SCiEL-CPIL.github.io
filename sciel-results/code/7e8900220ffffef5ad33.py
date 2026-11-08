import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def build_rank50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
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
            
            # M2 = (c+d)e
            U[u_c][col+1] += 1
            U[u_d][col+1] += 1
            V[v_e][col+1] += 1
            
            # M3 = a(f-h)
            U[u_a][col+2] += 1
            V[v_f][col+2] += 1
            V[v_h][col+2] -= 1
            
            # M4 = d(g-e)
            U[u_d][col+3] += 1
            V[v_g][col+3] += 1
            V[v_e][col+3] -= 1
            
            # M5 = (a+b)h
            U[u_a][col+4] += 1
            U[u_b][col+4] += 1
            V[v_h][col+4] += 1
            
            # M6 = (c-a)(e+f)
            U[u_c][col+5] += 1
            U[u_a][col+5] -= 1
            V[v_e][col+5] += 1
            V[v_f][col+5] += 1
            
            # M7 = (b-d)(g+h)
            U[u_b][col+6] += 1
            U[u_d][col+6] -= 1
            V[v_g][col+6] += 1
            V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1
            W[w_00][col+3] += 1
            W[w_00][col+4] -= 1
            W[w_00][col+6] += 1
            
            W[w_01][col+2] += 1
            W[w_01][col+4] += 1
            
            W[w_10][col+1] += 1
            W[w_10][col+3] += 1
            
            W[w_11][col] += 1
            W[w_11][col+1] -= 1
            W[w_11][col+2] += 1
            W[w_11][col+5] += 1
            
            col += 7
        
        # 1 block of 2x1 (last column k=8)
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1
        V[v_e][col] += 1
        U[u_b][col+1] += 1
        V[v_g][col+1] += 1
        U[u_c][col+2] += 1
        V[v_e][col+2] += 1
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        
        W[w_08][col] += 1
        W[w_08][col+1] += 1
        W[w_18][col+2] += 1
        W[w_18][col+3] += 1
        
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
        
        return U, V, W
    
    def verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, max_time):
        """Try to zero out entries to reduce effective rank."""
        rank = len(U[0])
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        # Try to zero out each column and check if it's still valid
        # A column r is "removable" if for all i,j,k: U[i,j,r]*V[j,k,r]*W[k,i,r] = 0
        # This means the column contributes nothing.
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        removed = []
        for r in range(rank):
            if time.time() > max_time:
                break
            # Check if column r can be removed
            # It can be removed if for all i,j,k: U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 0
            can_remove = True
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i*m+j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_arr[j*p+k, r]
                        w_val = W_arr[k*n+i, r]
                        if u_val * v_val * w_val != 0:
                            can_remove = False
                            break
                    if not can_remove:
                        break
                if not can_remove:
                    break
            
            if can_remove:
                # Zero out this column
                U_arr[:, r] = 0
                V_arr[:, r] = 0
                W_arr[:, r] = 0
                removed.append(r)
        
        # Convert back to lists
        U_new = U_arr.tolist()
        V_new = V_arr.tolist()
        W_new = W_arr.tolist()
        
        # Verify
        if not verify(U_new, V_new, W_new):
            return U, V, W
        
        return U_new, V_new, W_new
    
    # Build baseline
    U, V, W = build_rank50()
    
    # Verify
    if not verify(U, V, W):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U, V, W = try_reduce_rank(U, V, W, deadline - 5)
    
    return {"U": U, "V": V, "W": W}
