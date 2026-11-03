import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=i+2j (column-major flattening of C)
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1

    # Current known solution rank 28 (4 blocks of Strassen 2x2x2)
    rank_known = 28
    U_known = np.zeros((8, rank_known))
    V_known = np.zeros((16, rank_known))
    W_known = np.zeros((8, rank_known))
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen 2x2x2 decomposition
        # M1 = (A00+A11)(B00+B11)
        U_known[X_rows[0]][col_offset + 0] += 1
        U_known[X_rows[3]][col_offset + 0] += 1
        V_known[Y_rows[0]][col_offset + 0] += 1
        V_known[Y_rows[3]][col_offset + 0] += 1
        W_known[Z_rows[0]][col_offset + 0] += 1
        W_known[Z_rows[3]][col_offset + 0] += 1
        
        # M2 = A10(B01-B11)
        U_known[X_rows[2]][col_offset + 1] += 1
        U_known[X_rows[3]][col_offset + 1] += 1
        V_known[Y_rows[0]][col_offset + 1] += 1
        W_known[Z_rows[2]][col_offset + 1] += 1
        W_known[Z_rows[3]][col_offset + 1] -= 1
        
        # M3 = A00(B10-B11)
        U_known[X_rows[0]][col_offset + 2] += 1
        V_known[Y_rows[1]][col_offset + 2] += 1
        V_known[Y_rows[3]][col_offset + 2] -= 1
        W_known[Z_rows[1]][col_offset + 2] += 1
        W_known[Z_rows[3]][col_offset + 2] += 1
        
        # M4 = A11(B00+B10)
        U_known[X_rows[3]][col_offset + 3] += 1
        V_known[Y_rows[2]][col_offset + 3] += 1
        V_known[Y_rows[0]][col_offset + 3] -= 1
        W_known[Z_rows[0]][col_offset + 3] += 1
        W_known[Z_rows[2]][col_offset + 3] += 1
        
        # M5 = (A00+A01)B11
        U_known[X_rows[0]][col_offset + 4] += 1
        U_known[X_rows[1]][col_offset + 4] += 1
        V_known[Y_rows[3]][col_offset + 4] += 1
        W_known[Z_rows[0]][col_offset + 4] -= 1
        W_known[Z_rows[1]][col_offset + 4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U_known[X_rows[2]][col_offset + 5] += 1
        U_known[X_rows[0]][col_offset + 5] -= 1
        V_known[Y_rows[0]][col_offset + 5] += 1
        V_known[Y_rows[1]][col_offset + 5] += 1
        W_known[Z_rows[3]][col_offset + 5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U_known[X_rows[1]][col_offset + 6] += 1
        U_known[X_rows[3]][col_offset + 6] -= 1
        V_known[Y_rows[2]][col_offset + 6] += 1
        V_known[Y_rows[3]][col_offset + 6] += 1
        W_known[Z_rows[0]][col_offset + 6] += 1

    # Define the 4 blocks
    # A is 2x4, B is 4x4.
    # A1 (cols 0,1), A2 (cols 2,3)
    # B11 (rows 0,1; cols 0,1), B12 (rows 0,1; cols 2,3)
    # B21 (rows 2,3; cols 0,1), B22 (rows 2,3; cols 2,3)
    
    # X indices for A1: 0(i0k0), 1(i0k1), 4(i1k0), 5(i1k1)
    X1 = [0, 1, 4, 5]
    # X indices for A2: 2(i0k2), 3(i0k3), 6(i1k2), 7(i1k3)
    X2 = [2, 3, 6, 7]
    
    # Y indices for B11: 0(k0j0), 1(k0j1), 4(k1j0), 5(k1j1)
    Y11 = [0, 1, 4, 5]
    # Y indices for B12: 2(k0j2), 3(k0j3), 6(k1j2), 7(k1j3)
    Y12 = [2, 3, 6, 7]
    # Y indices for B21: 8(k2j0), 9(k2j1), 12(k3j0), 13(k3j1)
    Y21 = [8, 9, 12, 13]
    # Y indices for B22: 10(k2j2), 11(k2j3), 14(k3j2), 15(k3j3)
    Y22 = [10, 11, 14, 15]
    
    # Z indices for C
    # C is 2x4. z = i + 2j.
    # C11 (j=0,1): z(0,0)=0, z(0,1)=2, z(1,0)=1, z(1,1)=3 -> [0, 2, 1, 3]
    Z11 = [0, 2, 1, 3]
    # C12 (j=2,3): z(0,2)=4, z(0,3)=6, z(1,2)=5, z(1,3)=7 -> [4, 6, 5, 7]
    Z12 = [4, 6, 5, 7]
    
    # Block 1: A1 * B11 -> C11
    add_block(0, X1, Y11, Z11)
    # Block 2: A2 * B21 -> C11 (Wait, A2 is cols 2,3 of A. B21 is rows 2,3 of B. Product is 2x2 contributing to C cols 0,1? No.)
    # A is 2x4. B is 4x4.
    # C = A * B. C[i,j] = sum_k A[i,k] B[k,j].
    # A1 has k=0,1. A2 has k=2,3.
    # B11 has k=0,1; j=0,1.
    # A1 * B11 contributes to C[i, 0..1].
    # A1 * B12 contributes to C[i, 2..3].
    # A2 * B21 contributes to C[i, 0..1].
    # A2 * B22 contributes to C[i, 2..3].
    
    # So:
    # Block 1: A1 * B11 -> Z11
    # Block 2: A2 * B21 -> Z11
    # Block 3: A1 * B12 -> Z12
    # Block 4: A2 * B22 -> Z12
    
    add_block(7, X2, Y21, Z11)
    add_block(14, X1, Y12, Z12)
    add_block(21, X2, Y22, Z12)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    best_rank = 28
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 110

    # Try to find rank 27
    # We can try to merge two terms.
    # This is a hard problem. We'll use a heuristic search.
    
    # Initialize search from rank 28 solution
    # We'll try to find a linear combination that reduces rank by 1.
    
    def tensor_residual(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
        
        return np.sum((T_est - T) ** 2)

    # Local search for rank 27
    # Start from rank 28, zero out one term, and try to adjust others.
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        U = [row[:] for row in best_U]
        V = [row[:] for row in best_V]
        W = [row[:] for row in best_W]
        
        # Zero out the last term (index 27)
        for i in range(8):
            U[i][27] = 0
        for i in range(16):
            V[i][27] = 0
        for i in range(8):
            W[i][27] = 0
            
        rank = 27
        current_residual = tensor_residual(U, V, W, rank)
        best_residual = current_residual
        best_U_27 = [row[:] for row in U]
        best_V_27 = [row[:] for row in V]
        best_W_27 = [row[:] for row in W]
        
        for it in range(5000):
            if time.time() > deadline:
                break
                
            r = random.randint(0, rank - 1)
            mat = random.choice(['U', 'V', 'W'])
            
            if mat == 'U':
                row = random.randint(0, 7)
                old_val = U[row][r]
                delta = random.choice([-1, 1])
                U[row][r] += delta
            elif mat == 'V':
                row = random.randint(0, 15)
                old_val = V[row][r]
                delta = random.choice([-1, 1])
                V[row][r] += delta
            else:
                row = random.randint(0, 7)
                old_val = W[row][r]
                delta = random.choice([-1, 1])
                W[row][r] += delta
            
            new_residual = tensor_residual(U, V, W, rank)
            
            if new_residual < current_residual:
                current_residual = new_residual
                if new_residual < best_residual:
                    best_residual = new_residual
                    best_U_27 = [row[:] for row in U]
                    best_V_27 = [row[:] for row in V]
                    best_W_27 = [row[:] for row in W]
            else:
                if mat == 'U':
                    U[row][r] = old_val
                elif mat == 'V':
                    V[row][r] = old_val
                else:
                    W[row][r] = old_val
            
            if best_residual < 1e-9:
                if check_solution(best_U_27, best_V_27, best_W_27, rank):
                    best_rank = 27
                    best_U = best_U_27
                    best_V = best_V_27
                    best_W = best_W_27
                    break
        
        if best_rank == 27:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
