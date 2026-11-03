import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=4i+j
    # x in 0..7, y in 0..15, z in 0..7
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i  # Wait, the problem says W[k*n+i,r]?
                # "output factor follows the official trace convention W[k*n+i,r]"
                # This usually refers to the index mapping for the output matrix.
                # In the previous code, z = j * n + i was used.
                # Let's check the previous code's T definition.
                # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
                # Here n=2, so z = 2j + i.
                # My manual derivation: C is 2x4. Index w = i*4 + j.
                # Previous code: z = j*n + i = 2j + i.
                # This corresponds to column-major flattening of C?
                # C[i, j]. If flattened column-major: index = i + j*n.
                # Yes, z = i + 2j.
                # So I should use z = i + j * n.
                T[x, y, i + j * n] = 1

    # Current known solution rank 28
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Blocks for 2x4x4
    # Split A into two 2x2 (cols 0-1, 2-3)
    # Split B into two 2x4 (rows 0-1, 2-3)
    # A1 (cols 0-1) * B1 (rows 0-1) -> C part
    # A2 (cols 2-3) * B2 (rows 2-3) -> C part
    
    # X rows for A1: 0,1,4,5 (i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1)
    # X rows for A2: 2,3,6,7 (i=0,k=2; i=0,k=3; i=1,k=2; i=1,k=3)
    # Y rows for B1: 0,1,4,5 (k=0,j=0; k=0,j=1; k=1,j=0; k=1,j=1)
    # Y rows for B2: 8,9,12,13 (k=2,j=0; k=2,j=1; k=3,j=0; k=3,j=1)
    # Wait, B is 4x4. k is 0..3.
    # B1 uses k=0,1. Y indices: k*4+j.
    # k=0: 0,1,2,3. k=1: 4,5,6,7.
    # So B1 Y rows: 0,1,2,3,4,5,6,7.
    # B2 uses k=2,3. Y indices: 8,9,10,11,12,13,14,15.
    
    # Z rows for C
    # C is 2x4. z = i + 2j.
    # For A1*B1, we compute C[:, 0:2]? No.
    # A1 is 2x2 (cols 0,1). B1 is 2x4 (rows 0,1).
    # Product is 2x4.
    # So it contributes to all columns of C.
    # Z rows: 0..7.
    
    # The previous code used 4 blocks of 7.
    # It treated the problem as 4 independent 2x2x2 multiplications?
    # Let's stick to the previous code's block structure which worked for rank 28.
    
    X1 = [0, 1, 4, 5] # A cols 0,1
    Y1 = [0, 1, 4, 5] # B rows 0,1, cols 0,1?
    # The previous code's Y1 was [0,1,4,5]. This corresponds to k=0,j=0; k=0,j=1; k=1,j=0; k=1,j=1.
    # This is a 2x2 block of B.
    # So the previous code decomposed B into four 2x2 blocks.
    # B11 (k=0,1; j=0,1), B12 (k=0,1; j=2,3), B21 (k=2,3; j=0,1), B22 (k=2,3; j=2,3).
    # And A into two 2x2 blocks.
    # A1 (k=0,1), A2 (k=2,3).
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22.
    # Each term is 2x2 * 2x2 -> 2x2.
    # So 4 terms. Each rank 7. Total 28.
    
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3] # z = i + 2j. j=0,1. i=0,1.
    # z(0,0)=0, z(0,1)=2, z(1,0)=1, z(1,1)=3.
    # Order in X: 0(i0k0), 1(i0k1), 4(i1k0), 5(i1k1).
    # Order in Y: 0(k0j0), 1(k0j1), 4(k1j0), 5(k1j1).
    # Order in Z: 0(i0j0), 2(i0j1), 1(i1j0), 3(i1j1).
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7] # A cols 2,3
    Y2 = [8, 9, 12, 13] # B rows 2,3, cols 0,1
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5] # A cols 0,1
    Y3 = [2, 3, 6, 7] # B rows 0,1, cols 2,3
    Z3 = [4, 6, 5, 7] # j=2,3. z(0,2)=4, z(0,3)=6, z(1,2)=5, z(1,3)=7.
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7] # A cols 2,3
    Y4 = [10, 11, 14, 15] # B rows 2,3, cols 2,3
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

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

    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known
    
    start_time = time.time()
    deadline = start_time + 110

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

    def local_search_rank27(U_init, V_init, W_init, max_iter=5000):
        rank = 27
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        
        for i in range(8):
            U[i][27] = 0
        for i in range(16):
            V[i][27] = 0
        for i in range(8):
            W[i][27] = 0
        
        current_residual = tensor_residual(U, V, W, rank)
        best_residual = current_residual
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        for it in range(max_iter):
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
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                if mat == 'U':
                    U[row][r] = old_val
                elif mat == 'V':
                    V[row][r] = old_val
                else:
                    W[row][r] = old_val
            
            if best_residual < 1e-9:
                if check_solution(best_U, best_V, best_W, rank):
                    return best_U, best_V, best_W
        
        return None

    for restart in range(50):
        if time.time() > deadline:
            break
        
        result = local_search_rank27(U_known, V_known, W_known, max_iter=2000)
        if result is not None:
            U27, V27, W27 = result
            if check_solution(U27, V27, W27, 27):
                best_rank = 27
                best_U, best_V, best_W = U27, V27, W27
                break

    if best_rank == 28 and time.time() < deadline - 30:
        def local_search_rank26(U_init, V_init, W_init, max_iter=3000):
            rank = 26
            U = [row[:] for row in U_init]
            V = [row[:] for row in V_init]
            W = [row[:] for row in W_init]
            
            for i in range(8):
                U[i][26] = 0
                U[i][27] = 0
            for i in range(16):
                V[i][26] = 0
                V[i][27] = 0
            for i in range(8):
                W[i][26] = 0
                W[i][27] = 0
            
            current_residual = tensor_residual(U, V, W, rank)
            best_residual = current_residual
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            
            for it in range(max_iter):
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
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                else:
                    if mat == 'U':
                        U[row][r] = old_val
                    elif mat == 'V':
                        V[row][r] = old_val
                    else:
                        W[row][r] = old_val
                
                if best_residual < 1e-9:
                    if check_solution(best_U, best_V, best_W, rank):
                        return best_U, best_V, best_W
            
            return None
        
        for restart in range(20):
            if time.time() > deadline:
                break
            result = local_search_rank26(U_known, V_known, W_known, max_iter=1500)
            if result is not None:
                U26, V26, W26 = result
                if check_solution(U26, V26, W26, 26):
                    best_rank = 26
                    best_U, best_V, best_W = U26, V26, W26
                    break

    return {"U": best_U, "V": best_V, "W": best_W}
