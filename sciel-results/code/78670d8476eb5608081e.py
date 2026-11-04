import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check diagonal entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal entries (should be 0)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_w = c % n
                    k_w = c // n
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        return True
    
    # Build rank-42 Strassen decomposition with CORRECT W entries
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)
    
    def add_strassen_block(U_mat, V_mat, W_mat, i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        # M1 = (A00+A11)(B00+B11)
        U_mat[a00, offset] += 1.0; U_mat[a11, offset] += 1.0
        V_mat[b00, offset] += 1.0; V_mat[b11, offset] += 1.0
        W_mat[w00, offset] += 1.0; W_mat[w11, offset] += 1.0
        
        # M2 = (A10+A11)B00
        U_mat[a10, offset+1] += 1.0; U_mat[a11, offset+1] += 1.0
        V_mat[b00, offset+1] += 1.0
        W_mat[w10, offset+1] += 1.0; W_mat[w11, offset+1] -= 1.0
        
        # M3 = A00(B01-B11)
        U_mat[a00, offset+2] += 1.0
        V_mat[b01, offset+2] += 1.0; V_mat[b11, offset+2] -= 1.0
        W_mat[w01, offset+2] += 1.0; W_mat[w11, offset+2] -= 1.0
        
        # M4 = A11(B10-B00)
        U_mat[a11, offset+3] += 1.0
        V_mat[b10, offset+3] += 1.0; V_mat[b00, offset+3] -= 1.0
        W_mat[w00, offset+3] += 1.0; W_mat[w10, offset+3] -= 1.0
        
        # M5 = (A00+A01)B11
        U_mat[a00, offset+4] += 1.0; U_mat[a01, offset+4] += 1.0
        V_mat[b11, offset+4] += 1.0
        W_mat[w00, offset+4] -= 1.0; W_mat[w01, offset+4] += 1.0
        
        # M6 = (A10-A00)(B00+B01)
        U_mat[a10, offset+5] += 1.0; U_mat[a00, offset+5] -= 1.0
        V_mat[b00, offset+5] += 1.0; V_mat[b01, offset+5] += 1.0
        W_mat[w10, offset+5] += 1.0; W_mat[w11, offset+5] -= 1.0
        
        # M7 = (A01-A11)(B10+B11)
        U_mat[a01, offset+6] += 1.0; U_mat[a11, offset+6] -= 1.0
        V_mat[b10, offset+6] += 1.0; V_mat[b11, offset+6] += 1.0
        W_mat[w00, offset+6] += 1.0; W_mat[w01, offset+6] -= 1.0
    
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(U, V, W, i_start, j_start, k_start, offset)
                offset += 7
    
    best_rank = 42
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the rank-42 decomposition
    if verify(U, V, W):
        best_rank = 42
    else:
        # Fallback: schoolbook rank 48
        U_s = np.zeros((nm, 48), dtype=float)
        V_s = np.zeros((mp, 48), dtype=float)
        W_s = np.zeros((pn, 48), dtype=float)
        for idx in range(48):
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            U_s[i * m + j, idx] = 1.0
            V_s[j * p + k, idx] = 1.0
            W_s[k * n + i, idx] = 1.0
        if verify(U_s, V_s, W_s):
            best_rank = 48
            best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        else:
            return best_result
    
    # Try to reduce rank by finding linearly dependent columns
    # Use SVD-based approach: try to merge columns that are close
    # First, try simple column elimination via Gaussian elimination on the tensor
    
    # Attempt rank reduction by checking if any column can be expressed as combination of others
    # This is expensive, so we do it with a time budget
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try to find a lower rank decomposition using iterative refinement
        # Start from rank 42 and try to zero out one column at a time
        
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_rank = 42
        
        # Try removing columns one by one (greedy)
        # For each column, check if it can be represented as a linear combination of others
        # by solving a least-squares problem
        
        for attempt in range(10):  # Try up to 10 column removals
            if time.time() > deadline - 10:
                break
            
            removed = False
            for col_to_remove in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Try to express column col_to_remove as combination of others
                # Build the system: for each (a,b,c), the contribution from col_to_remove
                # should be matched by other columns
                
                # Get the "target" vector from column col_to_remove
                # We need: sum_t U[a,t]*V[b,t]*W[c,t] = delta_{a,b,c}
                # If we remove column r, we need to adjust other columns to compensate
                
                # Simple approach: try to find coefficients alpha_t such that
                # U[a,r]*V[b,r]*W[c,r] ≈ sum_{t≠r} alpha_t * U[a,t]*V[b,t]*W[c,t]
                
                # This is a nonlinear problem, so use a simpler heuristic:
                # Check if column r is nearly a linear combination of others in each matrix
                
                # Check U column
                U_others = np.delete(current_U, col_to_remove, axis=1)
                U_target = current_U[:, col_to_remove]
                
                # Check if U_target is in span of U_others
                if np.linalg.matrix_rank(U_others) < nm:
                    # Underdetermined, skip
                    continue
                
                # Least squares
                try:
                    coeffs_U, residuals_U, rank_U, _ = np.linalg.lstsq(U_others, U_target, rcond=None)
                except:
                    continue
                
                # Check V column
                V_others = np.delete(current_V, col_to_remove, axis=1)
                V_target = current_V[:, col_to_remove]
                
                try:
                    coeffs_V, residuals_V, rank_V, _ = np.linalg.lstsq(V_others, V_target, rcond=None)
                except:
                    continue
                
                # Check W column
                W_others = np.delete(current_W, col_to_remove, axis=1)
                W_target = current_W[:, col_to_remove]
                
                try:
                    coeffs_W, residuals_W, rank_W, _ = np.linalg.lstsq(W_others, W_target, rcond=None)
                except:
                    continue
                
                # Check if all three are well-approximated
                norm_U = np.linalg.norm(U_target)
                norm_V = np.linalg.norm(V_target)
                norm_W = np.linalg.norm(W_target)
                
                if norm_U < 1e-10 and norm_V < 1e-10 and norm_W < 1e-10:
                    # Column is zero, just remove it
                    current_U = np.delete(current_U, col_to_remove, axis=1)
                    current_V = np.delete(current_V, col_to_remove, axis=1)
                    current_W = np.delete(current_W, col_to_remove, axis=1)
                    current_rank -= 1
                    removed = True
                    break
                
                # Check if the residuals are small relative to norms
                res_U = np.linalg.norm(U_target - U_others @ coeffs_U) / (norm_U + 1e-15)
                res_V = np.linalg.norm(V_target - V_others @ coeffs_V) / (norm_V + 1e-15)
                res_W = np.linalg.norm(W_target - W_others @ coeffs_W) / (norm_W + 1e-15)
                
                if max(res_U, res_V, res_W) < 1e-6:
                    # Try to merge: adjust other columns
                    # For simplicity, just remove and verify
                    new_U = np.delete(current_U, col_to_remove, axis=1)
                    new_V = np.delete(current_V, col_to_remove, axis=1)
                    new_W = np.delete(current_W, col_to_remove, axis=1)
                    
                    # Add the contribution back
                    # new_U[:, t] += coeffs_U[t] * U_target  -- but this doesn't work directly
                    # We need a proper tensor decomposition update
                    
                    # For now, just check if removing it still gives valid decomposition
                    # (This won't work in general, but let's try)
                    if verify(new_U, new_V, new_W, tol=1e-6):
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank -= 1
                        removed = True
                        break
            if not removed:
                break
        
        if current_rank < best_rank:
            if verify(current_U, current_V, current_W):
                best_rank = current_rank
                best_result = {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
    
    return best_result
