import time
import numpy as np
from itertools import combinations

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

    # Verify rank-26
    U26, V26, W26 = get_rank26()
    assert check(U26, V26, W26), "Rank-26 decomposition failed verification"
    
    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
    
    def error(Ux, Vx, Wx):
        """Compute squared error of reconstruction."""
        recon = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
        return int(np.sum((recon - T) ** 2))

    def local_search(U_init, V_init, W_init, max_iters, time_limit, rng):
        """Perform local search to minimize error."""
        U_c = U_init.copy()
        V_c = V_init.copy()
        W_c = W_init.copy()
        
        current_err = error(U_c, V_c, W_c)
        if current_err == 0:
            return U_c, V_c, W_c, 0
        
        best_err = current_err
        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
        
        temperature = 5.0
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 1000
        rank = U_c.shape[1]
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random element to perturb
            # Focus on non-zero entries or near non-zero to keep sparsity
            mat_choice = rng.randint(3)
            row = rng.randint(9)
            col = rng.randint(rank)
            
            # Sometimes perturb a whole column slightly? No, keep element-wise for now but larger steps
            # Allow steps of -2, -1, 1, 2
            delta = rng.choice([-2, -1, 1, 2])
            
            if mat_choice == 0:
                U_c[row, col] += delta
            elif mat_choice == 1:
                V_c[row, col] += delta
            else:
                W_c[row, col] += delta
            
            new_err = error(U_c, V_c, W_c)
            
            # Accept if better, or with probability exp(-delta_err / temp)
            if new_err < current_err:
                current_err = new_err
                no_improve = 0
                if new_err < best_err:
                    best_err = new_err
                    best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
                    if best_err == 0:
                        break
            else:
                # Revert if not accepted
                if temperature > 1e-6 and rng.random() < np.exp(-(new_err - current_err) / temperature):
                    current_err = new_err
                    no_improve = 0
                    if new_err < best_err:
                        best_err = new_err
                        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
                        if best_err == 0:
                            break
                else:
                    # Revert
                    if mat_choice == 0:
                        U_c[row, col] -= delta
                    elif mat_choice == 1:
                        V_c[row, col] -= delta
                    else:
                        W_c[row, col] -= delta
            
            temperature *= cooling
            no_improve += 1
            
            if no_improve > max_no_improve and temperature < 0.5:
                break
        
        return best_Ux, best_Vx, best_Wx, best_err

    # Strategy: Sequentially try to reduce rank from 26 down to 23
    # Start with rank 26
    U_cur, V_cur, W_cur = U26.copy(), V26.copy(), W26.copy()
    current_rank = 26
    
    rng = np.random.RandomState(42)
    
    # Try to reduce to 25
    for target_rank in [25, 24, 23]:
        if time.time() > deadline - 10:
            break
            
        success = False
        # Try multiple random column deletions
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Pick a column to remove
            col_to_remove = rng.randint(current_rank)
            
            # Create new matrices without this column
            U_new = np.zeros((9, target_rank), dtype=np.int64)
            V_new = np.zeros((9, target_rank), dtype=np.int64)
            W_new = np.zeros((9, target_rank), dtype=np.int64)
            
            # Copy columns, skipping the one to remove
            idx = 0
            for c in range(current_rank):
                if c != col_to_remove:
                    U_new[:, idx] = U_cur[:, c]
                    V_new[:, idx] = V_cur[:, c]
                    W_new[:, idx] = W_cur[:, c]
                    idx += 1
            
            # Check if this simple removal works (unlikely)
            err = error(U_new, V_new, W_new)
            if err == 0:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                current_rank = target_rank
                success = True
                break
            
            # If not, try to repair
            # Use local search
            U_sol, V_sol, W_sol, err_sol = local_search(U_new, V_new, W_new, max_iters=1000, time_limit=deadline - 2, rng=rng)
            
            if err_sol == 0:
                U_cur, V_cur, W_cur = U_sol, V_sol, W_sol
                current_rank = target_rank
                success = True
                break
        
        if not success:
            # If we failed to reduce to target_rank, we might still be at a higher rank
            # We break here because reducing further is harder if we couldn't do the immediate step
            # However, we might have found a better configuration for the current rank?
            # The local search might have improved the current rank solution, but we didn't keep it.
            # Let's keep the best solution found at any rank.
            # Actually, the loop above only updates U_cur if success.
            # If we fail 25, we stay at 26.
            # We should probably try to improve the 26 solution too?
            # But the goal is rank reduction.
            break

    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', U_cur, V_cur, W_cur)
    if not np.array_equal(recon, T):
        # Fallback to rank-26
        U_cur, V_cur, W_cur = U26, V26, W26
        current_rank = 26
    
    # Convert to lists
    U_final = U_cur.tolist()
    V_final = V_cur.tolist()
    W_final = W_cur.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
