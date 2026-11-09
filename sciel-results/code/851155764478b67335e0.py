"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Schoolbook as fallback: rank = n*m*p = 48
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    def verify_fast(U_mat, V_mat, W_mat):
        """Faster verification using numpy einsum."""
        n, m, p = DIMENSIONS
        U_r = U_mat.reshape(n, m, U_mat.shape[1])
        V_r = V_mat.reshape(m, p, V_mat.shape[1])
        W_r = W_mat.reshape(p, n, W_mat.shape[1])
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def tensor_error(U_mat, V_mat, W_mat):
        """Return the max absolute error from target tensor of all 1s."""
        n, m, p = DIMENSIONS
        U_r = U_mat.reshape(n, m, U_mat.shape[1])
        V_r = V_mat.reshape(m, p, V_mat.shape[1])
        W_r = W_mat.reshape(p, n, W_mat.shape[1])
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(T - 1.0))

    def local_search(U_mat, V_mat, W_mat, target_rank, max_iters=5000):
        """Try to fix a near-solution via local entry adjustments."""
        n, m, p = DIMENSIONS
        entries = [-2, -1, -0.5, 0, 0.5, 1, 2]
        U = U_mat.copy()
        V = V_mat.copy()
        W = W_mat.copy()
        
        current_err = tensor_error(U, V, W)
        if current_err < 1e-9:
            return U, V, W, True
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random column and a random position in one of the matrices
            c = np.random.randint(target_rank)
            mat_choice = np.random.randint(3)
            
            if mat_choice == 0:
                row = np.random.randint(n * m)
                old_val = U[row, c]
            elif mat_choice == 1:
                row = np.random.randint(m * p)
                old_val = V[row, c]
            else:
                row = np.random.randint(p * n)
                old_val = W[row, c]
            
            # Try a new value
            new_val = entries[np.random.randint(len(entries))]
            if new_val == old_val:
                continue
            
            # Apply change
            if mat_choice == 0:
                U[row, c] = new_val
            elif mat_choice == 1:
                V[row, c] = new_val
            else:
                W[row, c] = new_val
            
            new_err = tensor_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-9:
                    return U, V, W, True
            else:
                # Revert
                if mat_choice == 0:
                    U[row, c] = old_val
                elif mat_choice == 1:
                    V[row, c] = old_val
                else:
                    W[row, c] = old_val
        
        return U, V, W, False

    # Strategy: structured random search with local refinement
    # Use entries from a smaller set for better convergence
    entries = [-1, -0.5, 0.5, 1]
    
    # Try ranks from 44 to 47
    for target_rank in [44, 45, 46, 47]:
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 8000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            for c in range(target_rank):
                # Exactly 3 nonzeros per column for balanced structure
                u_nonzero = np.random.choice(n*m, size=3, replace=False)
                v_nonzero = np.random.choice(m*p, size=3, replace=False)
                w_nonzero = np.random.choice(p*n, size=3, replace=False)
                
                for idx in u_nonzero:
                    U_mat[idx, c] = entries[np.random.randint(len(entries))]
                for idx in v_nonzero:
                    V_mat[idx, c] = entries[np.random.randint(len(entries))]
                for idx in w_nonzero:
                    W_mat[idx, c] = entries[np.random.randint(len(entries))]
            
            # Check exact solution
            if verify_fast(U_mat, V_mat, W_mat):
                best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                best_rank = target_rank
                break
            
            # Try local search if error is small
            err = tensor_error(U_mat, V_mat, W_mat)
            if err < 1.0:
                U_ls, V_ls, W_ls, found = local_search(U_mat, V_mat, W_mat, target_rank, max_iters=2000)
                if found:
                    best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                    best_rank = target_rank
                    break
        
        if best_rank < rank_school:
            break
    
    # If we found a better rank, try to optimize further
    if best_rank < rank_school and time.time() < deadline:
        current_rank = best_rank
        while current_rank > 40 and time.time() < deadline:
            new_rank = current_rank - 1
            found = False
            attempts = 0
            entries = [-1, -0.5, 0.5, 1]
            
            while attempts < 3000 and not found and time.time() < deadline:
                attempts += 1
                U_mat = np.zeros((n*m, new_rank))
                V_mat = np.zeros((m*p, new_rank))
                W_mat = np.zeros((p*n, new_rank))
                
                for c in range(new_rank):
                    u_nonzero = np.random.choice(n*m, size=3, replace=False)
                    v_nonzero = np.random.choice(m*p, size=3, replace=False)
                    w_nonzero = np.random.choice(p*n, size=3, replace=False)
                    
                    for idx in u_nonzero:
                        U_mat[idx, c] = entries[np.random.randint(len(entries))]
                    for idx in v_nonzero:
                        V_mat[idx, c] = entries[np.random.randint(len(entries))]
                    for idx in w_nonzero:
                        W_mat[idx, c] = entries[np.random.randint(len(entries))]
                
                if verify_fast(U_mat, V_mat, W_mat):
                    best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                    best_rank = new_rank
                    current_rank = new_rank
                    found = True
                else:
                    err = tensor_error(U_mat, V_mat, W_mat)
                    if err < 1.0:
                        U_ls, V_ls, W_ls, found_ls = local_search(U_mat, V_mat, W_mat, new_rank, max_iters=1500)
                        if found_ls:
                            best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                            best_rank = new_rank
                            current_rank = new_rank
                            found = True
            
            if not found:
                break
    
    return best_result
# EVOLVE-BLOCK-END
