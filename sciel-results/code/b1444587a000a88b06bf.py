import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    n, m, p = DIMENSIONS
    
    # Build target tensor: T[i,j,j',k,k',i'] = 1 iff i==i', j==j', k==k'
    # Flatten: a = i*3+j (A index), b = j'*3+k (B index), c = k'*3+i (W index = k*n+i)
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i  # W[k*n+i]
                T[a, b, c] = 1
    
    def check(U, V, W):
        """Check exact decomposition."""
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(recon, T)
    
    def get_rank(U, V, W):
        """Effective rank: number of columns where all three are non-zero."""
        mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
        return int(np.sum(mask))
    
    # F4 field representation approach:
    # Represent 3x3 matrices as 2x2 block matrices over F4 = {0, 1, alpha, alpha+1}
    # where alpha^2 = alpha + 1 in F4.
    # 
    # The key idea: a 3x3 matrix can be decomposed using the F4 field structure.
    # Let's use the representation where each "element" is a pair of bits.
    #
    # Actually, let me use a different approach: direct construction of low-rank decompositions.
    #
    # Known results: The tensor rank of 3x3 matrix multiplication is known to be 23.
    # Let's try to construct a rank-23 decomposition.
    #
    # Strategy: Start with rank-26 and try to reduce via local search.
    
    def get_rank26():
        """Standard Strassen-based rank-26 decomposition."""
        rank = 26
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)
        
        col = 0
        # Strassen for top-left 2x2 block
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
        
        # Cross terms for the 3rd row/col
        # a b^T (row 2 of A times col 2 of B)
        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
        
        # A2 b (row 2 of A times col 0,1 of B)
        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
        
        # a b3 (row 0,1 of A times col 2 of B)
        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
        
        # a^T B2 (row 0,1 of A times col 0,1 of B for output row 2)
        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
        
        # a3 b^T (row 2 of A times col 2 of B for output col 2)
        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
        
        # a^T b (row 0,1 of A times col 0,1 of B for output [2,2])
        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
        
        # a3 b3
        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        
        return U, V, W
    
    def get_rank27():
        """Standard schoolbook rank-27 decomposition."""
        rank = 27
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i * 3 + j][r] = 1
                    V[j * 3 + k][r] = 1
                    W[k * 3 + i][r] = 1
        
        return U, V, W
    
    # Verify rank-26
    U26, V26, W26 = get_rank26()
    assert check(U26, V26, W26), "Rank-26 decomposition failed verification"
    
    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
    
    # Local search to reduce rank
    # Strategy: Try to find a rank-r decomposition for r < 26 using simulated annealing
    # with integer/half-integer entries.
    
    rng = np.random.RandomState(12345)
    
    def compute_recon(Ux, Vx, Wx):
        """Compute reconstruction tensor."""
        return np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
    
    def error(Ux, Vx, Wx):
        """Compute squared error of reconstruction."""
        recon = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
        return int(np.sum((recon - T) ** 2))
    
    def fast_error_delta(Ux, Vx, Wx, recon_cache, mat_idx, row, col, old_val, new_val):
        """
        Compute change in error when modifying one element.
        mat_idx: 0=U, 1=V, 2=W
        Returns the change in squared error.
        """
        delta = new_val - old_val
        if delta == 0:
            return 0
        
        if mat_idx == 0:  # U[row, col] changes
            v_col = Vx[:, col]
            w_col = Wx[:, col]
            recon_row = recon_cache[row]
            T_row = T[row]
            diff = recon_row - T_row
            vw = np.outer(v_col, w_col)
            change = int(np.sum(2 * delta * vw * diff + delta**2 * vw**2))
            return change
        
        elif mat_idx == 1:  # V[row, col] changes
            u_col = Ux[:, col]
            w_col = Wx[:, col]
            recon_col_b = recon_cache[:, row, :]
            T_col_b = T[:, row, :]
            diff = recon_col_b - T_col_b
            uw = np.outer(u_col, w_col)
            change = int(np.sum(2 * delta * uw * diff + delta**2 * uw**2))
            return change
        
        else:  # W[row, col] changes
            u_col = Ux[:, col]
            v_col = Vx[:, col]
            recon_col_c = recon_cache[:, :, row]
            T_col_c = T[:, :, row]
            diff = recon_col_c - T_col_c
            uv = np.outer(u_col, v_col)
            change = int(np.sum(2 * delta * uv * diff + delta**2 * uv**2))
            return change
    
    def try_reduce_rank(target_rank, max_iters=5000, num_trials=10):
        """Try to find a rank-target_rank decomposition using local search."""
        if time.time() > deadline - 5:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize from rank-26 by randomly selecting target_rank columns
            U_t = np.zeros((9, target_rank), dtype=np.int64)
            V_t = np.zeros((9, target_rank), dtype=np.int64)
            W_t = np.zeros((9, target_rank), dtype=np.int64)
            
            cols_to_use = rng.choice(26, size=target_rank, replace=False)
            U_t = U26[:, cols_to_use].copy()
            V_t = V26[:, cols_to_use].copy()
            W_t = W26[:, cols_to_use].copy()
            
            recon_cache = compute_recon(U_t, V_t, W_t)
            current_err = error(U_t, V_t, W_t)
            best_err = current_err
            best_Ux, best_Vx, best_Wx = U_t.copy(), V_t.copy(), W_t.copy()
            
            temperature = 100.0
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 2000
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                mat_choice = rng.randint(3)
                row = rng.randint(9)
                col = rng.randint(target_rank)
                
                if mat_choice == 0:
                    old_val = int(U_t[row, col])
                    delta = rng.choice([-1, 1])
                    new_val = old_val + delta
                    err_change = fast_error_delta(U_t, V_t, W_t, recon_cache, 0, row, col, old_val, new_val)
                    U_t[row, col] = new_val
                    recon_cache[row] += delta * np.outer(V_t[:, col], W_t[:, col])
                elif mat_choice == 1:
                    old_val = int(V_t[row, col])
                    delta = rng.choice([-1, 1])
                    new_val = old_val + delta
                    err_change = fast_error_delta(U_t, V_t, W_t, recon_cache, 1, row, col, old_val, new_val)
                    V_t[row, col] = new_val
                    recon_cache[:, row, :] += delta * np.outer(U_t[:, col], W_t[:, col])
                else:
                    old_val = int(W_t[row, col])
                    delta = rng.choice([-1, 1])
                    new_val = old_val + delta
                    err_change = fast_error_delta(U_t, V_t, W_t, recon_cache, 2, row, col, old_val, new_val)
                    W_t[row, col] = new_val
                    recon_cache[:, :, row] += delta * np.outer(U_t[:, col], V_t[:, col])
                
                current_err += err_change
                
                if current_err < best_err or (temperature > 1e-6 and rng.random() < np.exp(-max(0, current_err - best_err) / temperature)):
                    if current_err < best_err:
                        best_err = current_err
                        best_Ux, best_Vx, best_Wx = U_t.copy(), V_t.copy(), W_t.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    if mat_choice == 0:
                        U_t[row, col] = old_val
                        recon_cache[row] -= delta * np.outer(V_t[:, col], W_t[:, col])
                    elif mat_choice == 1:
                        V_t[row, col] = old_val
                        recon_cache[:, row, :] -= delta * np.outer(U_t[:, col], W_t[:, col])
                    else:
                        W_t[row, col] = old_val
                        recon_cache[:, :, row] -= delta * np.outer(U_t[:, col], V_t[:, col])
                    current_err -= err_change
                    no_improve += 1
                
                temperature *= cooling
                
                if best_err == 0:
                    break
                if no_improve > max_no_improve and temperature < 5.0:
                    break
            
            if best_err == 0:
                if check(best_Ux, best_Vx, best_Wx):
                    mask = np.any(best_Ux != 0, axis=0) & np.any(best_Vx != 0, axis=0) & np.any(best_Wx != 0, axis=0)
                    actual_rank = int(np.sum(mask))
                    if actual_rank <= target_rank:
                        return best_Ux[:, mask], best_Vx[:, mask], best_Wx[:, mask], actual_rank
        
        return None
    
    # Try to find lower rank decompositions
    for target_rank in [23, 24, 25]:
        if time.time() > deadline - 15:
            break
        
        result = try_reduce_rank(target_rank, max_iters=8000, num_trials=5)
        if result is not None:
            U_r, V_r, W_r, actual_rank = result
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if not np.array_equal(recon, T):
        # Fallback to rank-26
        U26, V26, W26 = get_rank26()
        best_U, best_V, best_W = U26, V26, W26
        best_rank = 26
    
    # Convert to lists
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
