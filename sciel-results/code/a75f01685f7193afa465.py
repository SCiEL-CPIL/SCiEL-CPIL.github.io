import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    n, m, p = DIMENSIONS
    
    # Build target tensor: T[i,j,k] where i=(i_a,j_a), j=(j_b,k_b), k=(k_c,i_c)
    # T = 1 iff i_a==i_c and j_a==j_b and k_b==k_c
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i  # W index is k*n+i = k*3+i
                T[a, b, c] = 1
    
    # Precompute the non-zero positions for fast error computation
    T_nz_i, T_nz_j, T_nz_k = np.where(T != 0)
    T_nz_vals = T[T_nz_i, T_nz_j, T_nz_k]
    
    def get_rank26():
        rank = 26
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)
        
        col = 0
        # Strassen for top-left 2x2
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
        
        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
        
        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
        
        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
        
        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
        
        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
        
        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
        
        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        
        return U, V, W
    
    def fast_error(U, V, W):
        """Fast error computation using only non-zero T entries."""
        # For each non-zero T entry (i,j,k), compute sum_r U[i,r]*V[j,r]*W[k,r]
        # and compare to T[i,j,k]
        err = 0
        for idx in range(len(T_nz_i)):
            i, j, k = T_nz_i[idx], T_nz_j[idx], T_nz_k[idx]
            recon_val = np.dot(U[i, :], V[j, :] * W[k, :])
            err += (recon_val - T_nz_vals[idx]) ** 2
        return err
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces T."""
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(recon, T)
    
    def rank_of_decomposition(U, V, W):
        """Get the effective rank (number of non-zero columns)."""
        nonzero = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
        return int(np.sum(nonzero))
    
    # Start with rank 26
    U, V, W = get_rank26()
    best_rank = 26
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if check_decomposition(U, V, W):
        best_rank = rank_of_decomposition(U, V, W)
    
    rng = np.random.RandomState(42)
    
    def try_eliminate_column(U, V, W, col_idx):
        """
        Try to eliminate column col_idx by adjusting other columns.
        We need: sum_{r != col_idx} U[i,r]*V[j,r]*W[k,r] = T for all (i,j,k)
        This is a system of equations in the variables U[i,r], V[j,r], W[k,r] for r != col_idx.
        Since we're working with integers, this is hard. Instead, we'll use a heuristic:
        try to find small integer adjustments.
        """
        if time.time() > deadline - 20:
            return None
        
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Save the column to eliminate
        U_save = U_new[:, col_idx].copy()
        V_save = V_new[:, col_idx].copy()
        W_save = W_new[:, col_idx].copy()
        
        # Zero out the column
        U_new[:, col_idx] = 0
        V_new[:, col_idx] = 0
        W_new[:, col_idx] = 0
        
        # Current error after zeroing
        err = fast_error(U_new, V_new, W_new)
        
        if err == 0:
            return U_new, V_new, W_new
        
        # Try to fix by adjusting nearby columns
        # For each other column, try small adjustments
        other_cols = [c for c in range(U.shape[1]) if c != col_idx]
        
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            
            # Pick a random other column and a random row in one of the matrices
            col = other_cols[rng.randint(len(other_cols))]
            mat = rng.randint(3)
            row = rng.randint(9)
            delta = rng.choice([-2, -1, 1, 2])
            
            if mat == 0:
                U_new[row, col] += delta
            elif mat == 1:
                V_new[row, col] += delta
            else:
                W_new[row, col] += delta
            
            new_err = fast_error(U_new, V_new, W_new)
            
            if new_err < err:
                err = new_err
                if err == 0:
                    return U_new, V_new, W_new
            else:
                # Revert
                if mat == 0:
                    U_new[row, col] -= delta
                elif mat == 1:
                    V_new[row, col] -= delta
                else:
                    W_new[row, col] -= delta
        
        return None
    
    def try_full_search(target_rank, max_iter=2000):
        """Try to find a decomposition of exactly target_rank columns."""
        if time.time() > deadline - 30:
            return None
        
        for trial in range(30):
            if time.time() > deadline - 20:
                break
            
            # Initialize from rank-26 and zero out random columns
            U_t = U.copy()
            V_t = V.copy()
            W_t = W.copy()
            
            # Zero out columns to get to target_rank
            cols_to_zero = rng.choice(26, size=26 - target_rank, replace=False)
            for c in cols_to_zero:
                U_t[:, c] = 0
                V_t[:, c] = 0
                W_t[:, c] = 0
            
            err = fast_error(U_t, V_t, W_t)
            best_err = err
            best_Ut, best_Vt, best_Wt = U_t.copy(), V_t.copy(), W_t.copy()
            
            if err == 0:
                return best_Ut, best_Vt, best_Wt, target_rank
            
            # Simulated annealing
            temperature = 50.0
            cooling = 0.999
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Pick a random element to perturb
                mat_choice = rng.randint(3)
                row = rng.randint(9)
                col = rng.randint(target_rank)
                delta = rng.choice([-1, 1])
                
                if mat_choice == 0:
                    U_t[row, col] += delta
                elif mat_choice == 1:
                    V_t[row, col] += delta
                else:
                    W_t[row, col] += delta
                
                new_err = fast_error(U_t, V_t, W_t)
                
                if new_err < err or rng.random() < np.exp(-(new_err - err) / max(temperature, 1e-10)):
                    err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_Ut, best_Vt, best_Wt = U_t.copy(), V_t.copy(), W_t.copy()
                else:
                    if mat_choice == 0:
                        U_t[row, col] -= delta
                    elif mat_choice == 1:
                        V_t[row, col] -= delta
                    else:
                        W_t[row, col] -= delta
                
                temperature *= cooling
                
                if best_err == 0:
                    break
            
            if best_err == 0:
                mask = np.any(best_Ut != 0, axis=0) & np.any(best_Vt != 0, axis=0) & np.any(best_Wt != 0, axis=0)
                actual_rank = int(np.sum(mask))
                if actual_rank <= target_rank:
                    return best_Ut[:, mask], best_Vt[:, mask], best_Wt[:, mask], actual_rank
        
        return None
    
    # Strategy 1: Try to eliminate columns from the rank-26 solution
    for col_idx in range(26):
        if time.time() > deadline - 40:
            break
        
        result = try_eliminate_column(best_U, best_V, best_W, col_idx)
        if result is not None:
            U_r, V_r, W_r = result
            actual_rank = rank_of_decomposition(U_r, V_r, W_r)
            if actual_rank < best_rank and check_decomposition(U_r, V_r, W_r):
                best_rank = actual_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    # Strategy 2: Try direct search for lower ranks
    for r in range(23, min(26, best_rank)):
        if time.time() > deadline - 25:
            break
        
        result = try_full_search(r, max_iter=3000)
        if result is not None:
            U_r, V_r, W_r, actual_rank = result
            if actual_rank < best_rank and check_decomposition(U_r, V_r, W_r):
                best_rank = actual_rank
                best_U, best_V, best_W = U_r, V_r, W_r
    
    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if not np.array_equal(recon, T):
        # Fallback to known rank-26
        U_fb, V_fb, W_fb = get_rank26()
        best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Convert to lists
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
