import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1.0

    # Known rank 28 solution using Strassen's algorithm structure
    rank_known = 28
    U_known = np.zeros((8, rank_known))
    V_known = np.zeros((16, rank_known))
    W_known = np.zeros((8, rank_known))
    
    # Strassen's algorithm for 2x2 blocks
    # Split 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each block is 2x2, so we use Strassen's 7 multiplications for each pair
    
    def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
        """
        Add Strassen's 7 terms for multiplying 2x2 blocks
        X_rows: 4 rows corresponding to [A11, A12, A21, A22] entries
        Y_rows: 4 rows corresponding to [B11, B12, B21, B22] entries  
        Z_rows: 4 rows corresponding to [C11, C12, C21, C22] entries
        """
        # Strassen's formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 = (A11 + A22)(B11 + B22)
        U_known[X_rows[0]][col_offset + 0] += 1
        U_known[X_rows[3]][col_offset + 0] += 1
        V_known[Y_rows[0]][col_offset + 0] += 1
        V_known[Y_rows[2]][col_offset + 0] += 1
        W_known[Z_rows[0]][col_offset + 0] += 1
        W_known[Z_rows[3]][col_offset + 0] += 1
        
        # M2 = (A21 + A22)B11
        U_known[X_rows[2]][col_offset + 1] += 1
        U_known[X_rows[3]][col_offset + 1] += 1
        V_known[Y_rows[0]][col_offset + 1] += 1
        W_known[Z_rows[2]][col_offset + 1] += 1
        W_known[Z_rows[3]][col_offset + 1] -= 1
        
        # M3 = A11(B12 - B22)
        U_known[X_rows[0]][col_offset + 2] += 1
        V_known[Y_rows[1]][col_offset + 2] += 1
        V_known[Y_rows[2]][col_offset + 2] -= 1
        W_known[Z_rows[1]][col_offset + 2] += 1
        W_known[Z_rows[3]][col_offset + 2] += 1
        
        # M4 = A22(B21 - B11)
        U_known[X_rows[3]][col_offset + 3] += 1
        V_known[Y_rows[2]][col_offset + 3] += 1
        V_known[Y_rows[0]][col_offset + 3] -= 1
        W_known[Z_rows[0]][col_offset + 3] += 1
        W_known[Z_rows[2]][col_offset + 3] += 1
        
        # M5 = (A11 + A12)B22
        U_known[X_rows[0]][col_offset + 4] += 1
        U_known[X_rows[1]][col_offset + 4] += 1
        V_known[Y_rows[2]][col_offset + 4] += 1
        W_known[Z_rows[0]][col_offset + 4] -= 1
        W_known[Z_rows[1]][col_offset + 4] += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U_known[X_rows[2]][col_offset + 5] += 1
        U_known[X_rows[0]][col_offset + 5] -= 1
        V_known[Y_rows[0]][col_offset + 5] += 1
        V_known[Y_rows[1]][col_offset + 5] += 1
        W_known[Z_rows[3]][col_offset + 5] += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U_known[X_rows[1]][col_offset + 6] += 1
        U_known[X_rows[3]][col_offset + 6] -= 1
        V_known[Y_rows[2]][col_offset + 6] += 1
        V_known[Y_rows[3]][col_offset + 6] += 1
        W_known[Z_rows[0]][col_offset + 6] += 1

    # The 4x4 matrix A is split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # For 2x4 * 4x4: A is 2x4, so we can think of it as two 2x2 blocks side by side
    # A = [A_left, A_right] where A_left and A_right are 2x2
    
    # Actually, let me reconsider the block structure
    # A is 2x4, B is 4x4, C is 2x4
    # Split B into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # Split A into left and right 2x2: A = [A1, A2] where A1, A2 are 2x2
    # C = A1*B_left + A2*B_right where B_left = [B11, B21], B_right = [B12, B22]
    
    # Hmm, this is getting complicated. Let me use a different approach.
    # The known rank 28 solution from the original code should work.
    # Let me verify it first.
    
    X1 = [0, 1, 4, 5]
    X2 = [2, 3, 6, 7]
    Y11 = [0, 1, 4, 5]
    Y12 = [2, 3, 6, 7]
    Y21 = [8, 9, 12, 13]
    Y22 = [10, 11, 14, 15]
    Z11 = [0, 2, 1, 3]
    Z12 = [4, 6, 5, 7]
    
    add_strassen_block(0, X1, Y11, Z11)
    add_strassen_block(7, X2, Y21, Z11)
    add_strassen_block(14, X1, Y12, Z12)
    add_strassen_block(21, X2, Y22, Z12)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Verify the known solution
    if not check_solution(U_known.tolist(), V_known.tolist(), W_known.tolist(), rank_known):
        # If the known solution is wrong, fall back to naive rank 32
        rank_known = 32
        U_known = np.zeros((8, rank_known))
        V_known = np.zeros((16, rank_known))
        W_known = np.zeros((8, rank_known))
        for i in range(2):
            for k in range(4):
                for j in range(4):
                    x = i * 4 + k
                    y = k * 4 + j
                    z = i + j * 2
                    r = i * 16 + k * 4 + j
                    U_known[x, r] = 1
                    V_known[y, r] = 1
                    W_known[z, r] = 1
        if not check_solution(U_known.tolist(), V_known.tolist(), W_known.tolist(), rank_known):
            # Return a trivial valid solution
            return {"U": U_known.tolist(), "V": V_known.tolist(), "W": W_known.tolist()}

    best_rank = rank_known
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    def optimize_rank(U_init, V_init, W_init, rank, max_iters=3000):
        """
        Local search using incremental gradient updates with smooth surrogate.
        """
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        # Precompute outer products for fast T_est calculation
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += U[:, r][:, None, None] * np.outer(V[:, r], W[:, r])
            
        Residual = T_est - T
        current_loss = np.sum(Residual ** 2)
        
        best_loss = current_loss
        best_U_state = U.copy()
        best_V_state = V.copy()
        best_W_state = W.copy()
        
        # Precompute norms for Hessian approximation
        NormU_sq = np.sum(U**2, axis=0)
        NormV_sq = np.sum(V**2, axis=0)
        NormW_sq = np.sum(W**2, axis=0)
        
        def compute_gradients():
            Grad_U = np.zeros((8, rank), dtype=float)
            Grad_V = np.zeros((16, rank), dtype=float)
            Grad_W = np.zeros((8, rank), dtype=float)
            
            for r in range(rank):
                VW_outer = np.outer(V[:, r], W[:, r])
                Grad_U[:, r] = 2.0 * np.tensordot(Residual, VW_outer, axes=([1, 2], [0, 1]))
                
                RW = np.tensordot(Residual, W[:, r], axes=([2], [0]))
                Grad_V[:, r] = 2.0 * (RW.T @ U[:, r])
                
                RV = np.tensordot(Residual, V[:, r], axes=([1], [0]))
                Grad_W[:, r] = 2.0 * (RV.T @ U[:, r])
                
            return Grad_U, Grad_V, Grad_W
        
        Grad_U, Grad_V, Grad_W = compute_gradients()
        
        no_improvement_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            best_move = None
            best_new_loss = current_loss + 1e-12
            
            # Evaluate moves for U
            for r in range(rank):
                quad_U = NormV_sq[r] * NormW_sq[r]
                for i in range(8):
                    for delta in [-1, 1]:
                        new_loss = current_loss + delta * Grad_U[i, r] + delta * delta * quad_U
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('U', i, r, delta)
                            
            # Evaluate moves for V
            for r in range(rank):
                quad_V = NormU_sq[r] * NormW_sq[r]
                for y in range(16):
                    for delta in [-1, 1]:
                        new_loss = current_loss + delta * Grad_V[y, r] + delta * delta * quad_V
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('V', y, r, delta)
                            
            # Evaluate moves for W
            for r in range(rank):
                quad_W = NormU_sq[r] * NormV_sq[r]
                for z in range(8):
                    for delta in [-1, 1]:
                        new_loss = current_loss + delta * Grad_W[z, r] + delta * delta * quad_W
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('W', z, r, delta)
                            
            if best_move is None:
                no_improvement_count += 1
                # Try random perturbation to escape local minimum
                if no_improvement_count > 50 and random.random() < 0.1:
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = random.randint(0, 7)
                    elif mat == 'V':
                        row = random.randint(0, 15)
                    else:
                        row = random.randint(0, 7)
                    r = random.randint(0, rank - 1)
                    delta = random.choice([-1, 1])
                    best_move = (mat, row, r, delta)
                    best_new_loss = current_loss + 0.5
                else:
                    break
            else:
                no_improvement_count = 0
            
            mat, row, r, delta = best_move
            
            # Apply move
            if mat == 'U':
                U[row, r] += delta
                Delta_Res = np.zeros((8, 16, 8), dtype=float)
                Delta_Res[row, :, :] = delta * np.outer(V[:, r], W[:, r])
                Residual += Delta_Res
                T_est += Delta_Res
                
                # Update gradients
                for rp in range(rank):
                    DotV_rrp = np.dot(V[:, r], V[:, rp])
                    DotW_rrp = np.dot(W[:, r], W[:, rp])
                    Grad_U[row, rp] += 2 * delta * DotV_rrp * DotW_rrp
                    Grad_V[:, rp] += 2 * delta * V[:, r] * DotW_rrp * U[row, rp]
                    Grad_W[:, rp] += 2 * delta * W[:, r] * DotV_rrp * U[row, rp]
                    
                NormU_sq[r] = np.sum(U[:, r]**2)
                
            elif mat == 'V':
                V[row, r] += delta
                Delta_Res = np.zeros((8, 16, 8), dtype=float)
                Delta_Res[:, row, :] = delta * np.outer(U[:, r], W[:, r])
                Residual += Delta_Res
                T_est += Delta_Res
                
                for rp in range(rank):
                    DotU_rrp = np.dot(U[:, r], U[:, rp])
                    DotW_rrp = np.dot(W[:, r], W[:, rp])
                    Grad_V[row, rp] += 2 * delta * DotU_rrp * DotW_rrp
                    Grad_U[:, rp] += 2 * delta * U[:, r] * V[row, rp] * DotW_rrp
                    Grad_W[:, rp] += 2 * delta * W[:, r] * V[row, rp] * DotU_rrp
                    
                NormV_sq[r] = np.sum(V[:, r]**2)
                
            else:  # W
                W[row, r] += delta
                Delta_Res = np.zeros((8, 16, 8), dtype=float)
                Delta_Res[:, :, row] = delta * np.outer(U[:, r], V[:, r])
                Residual += Delta_Res
                T_est += Delta_Res
                
                for rp in range(rank):
                    DotU_rrp = np.dot(U[:, r], U[:, rp])
                    DotV_rrp = np.dot(V[:, r], V[:, rp])
                    Grad_W[row, rp] += 2 * delta * DotU_rrp * DotV_rrp
                    Grad_U[:, rp] += 2 * delta * U[:, r] * W[row, rp] * DotV_rrp
                    Grad_V[:, rp] += 2 * delta * V[:, r] * W[row, rp] * DotU_rrp
                    
                NormW_sq[r] = np.sum(W[:, r]**2)
                
            current_loss = best_new_loss
            if current_loss < best_loss - 1e-12:
                best_loss = current_loss
                best_U_state = U.copy()
                best_V_state = V.copy()
                best_W_state = W.copy()
                
            if best_loss < 1e-9:
                if check_solution(best_U_state, best_V_state, best_W_state, rank):
                    return best_U_state, best_V_state, best_W_state, True
                    
        return best_U_state, best_V_state, best_W_state, False

    # Try to find rank 27 by zeroing out one column and optimizing
    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        U_init = U_known.copy()
        V_init = V_known.copy()
        W_init = W_known.copy()
        
        # Zero out a random column
        col_to_zero = random.randint(0, rank_known - 1)
        U_init[:, col_to_zero] = 0
        V_init[:, col_to_zero] = 0
        W_init[:, col_to_zero] = 0
        
        # Remove the zeroed column to reduce rank
        mask = np.ones(rank_known, dtype=bool)
        mask[col_to_zero] = False
        U_opt = U_init[:, mask]
        V_opt = V_init[:, mask]
        W_opt = W_init[:, mask]
        
        U_res, V_res, W_res, success = optimize_rank(U_opt, V_opt, W_opt, rank_known - 1, max_iters=1500)
        
        if success:
            best_rank = 27
            best_U = U_res.tolist()
            best_V = V_res.tolist()
            best_W = W_res.tolist()
            break

    # Final verification
    if best_rank <= 27:
        if not check_solution(best_U, best_V, best_W, best_rank):
            best_rank = rank_known
            best_U = U_known.tolist()
            best_V = V_known.tolist()
            best_W = W_known.tolist()

    return {"U": best_U, "V": best_V, "W": best_W}
