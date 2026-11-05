import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor
    # T[a,b,c] = 1 if C[k*n+i] depends on A[i*m+j] * B[j*p+k]
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def _get_rank17_factors(n, m, p):
        """Return the known rank-17 factors."""
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val

        # M0: (A00 + A11)(B00 + B11)
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        # M1: (A10 + A11)B00
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        # M2: A00(B01 - B11)
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        # M3: A11(B10 - B00)
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        # M4: (A00 + A01)B11
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        # M5: (A10 - A00)(B00 + B01)
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        # M6: (A11 - A01)(B10 + B11)
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)

        # Term 7: A[0,0] * B[0,2]
        set_U(0, 0, 7, 1)
        set_V(0, 2, 7, 1)
        set_W(2, 0, 7, 1)
        
        # Term 8: A[0,1] * B[1,2]
        set_U(0, 1, 8, 1)
        set_V(1, 2, 8, 1)
        set_W(2, 0, 8, 1)
        
        # Term 9: A[1,0] * B[0,2]
        set_U(1, 0, 9, 1)
        set_V(0, 2, 9, 1)
        set_W(2, 1, 9, 1)
        
        # Term 10: A[1,1] * B[1,2]
        set_U(1, 1, 10, 1)
        set_V(1, 2, 10, 1)
        set_W(2, 1, 10, 1)
        
        # Term 11: A[0,2] * B[2,0]
        set_U(0, 2, 11, 1)
        set_V(2, 0, 11, 1)
        set_W(0, 0, 11, 1)
        
        # Term 12: A[0,2] * B[2,1]
        set_U(0, 2, 12, 1)
        set_V(2, 1, 12, 1)
        set_W(1, 0, 12, 1)
        
        # Term 13: A[0,2] * B[2,2]
        set_U(0, 2, 13, 1)
        set_V(2, 2, 13, 1)
        set_W(2, 0, 13, 1)
        
        # Term 14: A[1,2] * B[2,0]
        set_U(1, 2, 14, 1)
        set_V(2, 0, 14, 1)
        set_W(0, 1, 14, 1)
        
        # Term 15: A[1,2] * B[2,1]
        set_U(1, 2, 15, 1)
        set_V(2, 1, 15, 1)
        set_W(1, 1, 15, 1)
        
        # Term 16: A[1,2] * B[2,2]
        set_U(1, 2, 16, 1)
        set_V(2, 2, 16, 1)
        set_W(2, 1, 16, 1)
        
        return U, V, W
    
    def check_exact(U, V, W, tol=1e-8):
        """Check if the decomposition is exact and has half-integer entries."""
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err > tol:
            return False, err
        
        def is_half_integer(arr):
            scaled = 2 * arr
            rounded = np.round(scaled)
            return np.max(np.abs(scaled - rounded)) < tol
        
        return (is_half_integer(U) and is_half_integer(V) and is_half_integer(W)), err
    
    def try_rank16_hybrid():
        """Try to find rank-16 using structured search from rank-17."""
        if time.time() > deadline:
            return None
        
        U17, V17, W17 = _get_rank17_factors(n, m, p)
        r17 = 17
        r16 = 16
        
        # Strategy: The rank-17 solution has 7 "complex" terms (0-6) and 9 "simple" terms (7-16)
        # The 9 simple terms correspond to the 3rd column of A times 3rd column of B contributions
        # Let's try to find a rank-16 by modifying the structure
        
        # Approach 1: Try all pairs of terms to merge/eliminate
        # For each pair (i,j), try to find alpha such that
        # term_i + alpha * term_j can be absorbed into other terms
        
        # Approach 2: Use the fact that we need T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # with only 16 terms. Start from rank-17 and try to set one column to be
        # a linear combination of others, then re-solve.
        
        # Let's try: remove term 16 (last simple term) and see if we can adjust
        # the remaining 16 terms to still represent T exactly.
        
        # Actually, let's use a more systematic approach:
        # Fix the first 16 columns of U, V, W from rank-17, and try to find
        # a 16th column that, combined with adjusted versions of the first 15,
        # gives the exact tensor.
        
        # Better: Use alternating minimization with rank 16, initialized from
        # rank-17 with one term removed and the rest adjusted.
        
        # Initialize rank-16 factors from rank-17 by dropping one term
        # and trying to compensate
        
        best_result = None
        best_err = np.inf
        
        # Try dropping each of the 17 terms and re-solving
        for drop_idx in range(r17):
            if time.time() > deadline:
                break
            
            U_init = np.delete(U17, drop_idx, axis=1)
            V_init = np.delete(V17, drop_idx, axis=1)
            W_init = np.delete(W17, drop_idx, axis=1)
            
            # Now do alternating least squares to refine
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Alternating minimization with multiple restarts
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                # Add small perturbation for restarts > 0
                if restart > 0:
                    U_curr = U_init + 0.01 * np.random.randn(n*m, r16)
                    V_curr = V_init + 0.01 * np.random.randn(m*p, r16)
                    W_curr = W_init + 0.01 * np.random.randn(p*n, r16)
                
                # Alternating least squares
                for iteration in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Update U
                    Y = T.reshape(n*m, m*p * p*n)
                    Z3 = V_curr[:, None, :] * W_curr[None, :, :]  # (m*p, p*n, r16)
                    Z2 = Z3.reshape(m*p*p*n, r16)
                    try:
                        U_new, _, _, _ = np.linalg.lstsq(Z2.T, Y, rcond=None)
                    except:
                        U_new = U_curr.copy()
                    U_curr = U_new
                    
                    # Update V
                    Y2 = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m * p*n)
                    Z3b = (U_curr[:, None, :] * W_curr[None, :, :]).reshape(n*m*p*n, r16)
                    try:
                        V_new, _, _, _ = np.linalg.lstsq(Z3b.T, Y2, rcond=None)
                    except:
                        V_new = V_curr.copy()
                    V_curr = V_new
                    
                    # Update W
                    Y3 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m * m*p)
                    Z3c = (U_curr[:, None, :] * V_curr[None, :, :]).reshape(n*m*m*p, r16)
                    try:
                        W_new, _, _, _ = np.linalg.lstsq(Z3c.T, Y3, rcond=None)
                    except:
                        W_new = W_curr.copy()
                    W_curr = W_new
                
                # Check result
                valid, err = check_exact(U_curr, V_curr, W_curr)
                if valid:
                    # Round to half-integers
                    U_round = np.round(2 * U_curr) / 2
                    V_round = np.round(2 * V_curr) / 2
                    W_round = np.round(2 * W_curr) / 2
                    valid_round, err_round = check_exact(U_round, V_round, W_round)
                    if valid_round:
                        return U_round, V_round, W_round
                
                if err < best_err:
                    best_err = err
                    best_result = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        # If we found a near-exact solution, try to polish it
        if best_result is not None and best_err < 1e-4:
            U_curr, V_curr, W_curr = best_result
            # High-precision polish
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                Y = T.reshape(n*m, m*p * p*n)
                Z3 = V_curr[:, None, :] * W_curr[None, :, :]
                Z2 = Z3.reshape(m*p*p*n, r16)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(Z2.T, Y, rcond=None)
                except:
                    break
                U_curr = U_new
                
                Y2 = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m * p*n)
                Z3b = (U_curr[:, None, :] * W_curr[None, :, :]).reshape(n*m*p*n, r16)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(Z3b.T, Y2, rcond=None)
                except:
                    break
                V_curr = V_new
                
                Y3 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m * m*p)
                Z3c = (U_curr[:, None, :] * V_curr[None, :, :]).reshape(n*m*m*p, r16)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(Z3c.T, Y3, rcond=None)
                except:
                    break
                W_curr = W_new
            
            valid, err = check_exact(U_curr, V_curr, W_curr)
            if valid:
                U_round = np.round(2 * U_curr) / 2
                V_round = np.round(2 * V_curr) / 2
                W_round = np.round(2 * W_curr) / 2
                valid_round, err_round = check_exact(U_round, V_round, W_round)
                if valid_round:
                    return U_round, V_round, W_round
        
        return None
    
    def try_rank16_structured():
        """Try a structured rank-16 approach exploiting the problem structure."""
        if time.time() > deadline:
            return None
        
        # For 2x3x3, let's try a specific structured form.
        # The key insight: we can group terms by which output column they affect.
        # C[0,0], C[0,1], C[0,2] depend on A[0,:], C[1,0], C[1,1], C[1,2] depend on A[1,:]
        
        # Try to find a rank-16 decomposition with a specific sparsity pattern.
        # Use the rank-17 as a guide and try to combine terms.
        
        U17, V17, W17 = _get_rank17_factors(n, m, p)
        r16 = 16
        
        # Try: keep terms 0-14 (15 terms) and find a 16th term that replaces 15 and 16
        # Or: keep 16 terms and adjust them
        
        # Let's try a different initialization: use the HOSVD of T directly
        T1 = T.reshape(n*m, m*p*p*n)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        T2 = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m*p*n)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        T3 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m*m*p)
        U3, S3, Vt3 = np.linalg.svd(T3, full_matrices=False)
        
        # Initialize from truncated SVD
        U_curr = U1[:, :r16].copy()
        V_curr = U2[:, :r16].copy()
        W_curr = U3[:, :r16].copy()
        
        # Alternating minimization
        for iteration in range(300):
            if time.time() > deadline:
                break
            
            Y = T.reshape(n*m, m*p * p*n)
            Z3 = V_curr[:, None, :] * W_curr[None, :, :]
            Z2 = Z3.reshape(m*p*p*n, r16)
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z2.T, Y, rcond=None)
            except:
                break
            U_curr = U_new
            
            Y2 = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m * p*n)
            Z3b = (U_curr[:, None, :] * W_curr[None, :, :]).reshape(n*m*p*n, r16)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z3b.T, Y2, rcond=None)
            except:
                break
            V_curr = V_new
            
            Y3 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m * m*p)
            Z3c = (U_curr[:, None, :] * V_curr[None, :, :]).reshape(n*m*m*p, r16)
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z3c.T, Y3, rcond=None)
            except:
                break
            W_curr = W_new
        
        valid, err = check_exact(U_curr, V_curr, W_curr)
        if valid:
            U_round = np.round(2 * U_curr) / 2
            V_round = np.round(2 * V_curr) / 2
            W_round = np.round(2 * W_curr) / 2
            valid_round, err_round = check_exact(U_round, V_round, W_round)
            if valid_round:
                return U_round, V_round, W_round
        
        return None
    
    # Try rank-16 approaches
    result = try_rank16_hybrid()
    if result is not None:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    result = try_rank16_structured()
    if result is not None:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fall back to rank 17
    U17, V17, W17 = _get_rank17_factors(n, m, p)
    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }
