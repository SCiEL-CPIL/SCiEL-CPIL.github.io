import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n  # 6, 27, 18
    deadline = time.time() + 140

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def verify(U, V, W, tol=1e-10):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def verify_error(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_rec - T) ** 2)

    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)

    def snap_half_integer(mat):
        return np.round(2.0 * mat) / 2.0

    def als_step(T, U, V, W, nm, mp, pn, r):
        # Update U
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, r)
        T_flat = T.reshape(nm, mp * pn)
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_VW, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
        # Update V
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, r)
        T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_UW, T_flat_V.T, rcond=None)
            V = V_new.T
        except:
            pass
        # Update W
        M_UV = np.einsum('ar,br->abr', U, V).reshape(nm * mp, r)
        T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_UV, T_flat_W.T, rcond=None)
            W = W_new.T
        except:
            pass
        return U, V, W

    # Strategy 1: Integer-structured search
    # Fix V and W to small integers, solve for U, check if U is half-integer
    # This exploits the fact that the tensor has 0/1 entries
    
    def integer_search(r, max_trials=200):
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                return None, None, None, False
            
            np.random.seed(42 + r * 1000 + trial)
            
            # Try different integer ranges
            if trial < 60:
                v_range = (-1, 2)  # -1, 0, 1
                w_range = (-1, 2)
            elif trial < 120:
                v_range = (-2, 3)  # -2, -1, 0, 1, 2
                w_range = (-2, 3)
            else:
                v_range = (-3, 4)  # -3 to 3
                w_range = (-3, 4)
            
            V = np.random.randint(v_range[0], v_range[1], size=(mp, r))
            W = np.random.randint(w_range[0], w_range[1], size=(pn, r))
            
            # Build M: M[b*pn+c, r] = V[b, r] * W[c, r]
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            
            # Check rank
            if np.linalg.matrix_rank(M) < r:
                continue
            
            # Solve for U
            U = np.zeros((nm, r))
            valid = True
            for a in range(nm):
                target = T[a, :, :].flatten()
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                    U[a, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if U is half-integer
            if is_half_integer(U):
                U_snapped = snap_half_integer(U)
                if verify(U_snapped, V, W):
                    return U_snapped, V.astype(float), W.astype(float), True
        
        return None, None, None, False

    # Strategy 2: ALS with random initialization
    def als_search(r, max_trials=50):
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                return None, None, None, False
            
            np.random.seed(999 + r * 1000 + trial)
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            prev_error = np.inf
            for iteration in range(500):
                if time.time() > deadline - 15:
                    return None, None, None, False
                
                U, V, W = als_step(T, U, V, W, nm, mp, pn, r)
                error = verify_error(U, V, W)
                
                if error < 1e-12:
                    return U, V, W, True
                elif error > prev_error * 1.5 and iteration > 10:
                    break
                prev_error = error
        
        return None, None, None, False

    # Search from low ranks upward
    # Lower bound: max of slice ranks
    # For 2x3x9, the rank is at least max(rank of slices)
    # Each slice in the first mode is a 27x18 matrix
    # T[0, j, k] = 1 only when i=0, so j*p+k must correspond to valid (j,k)
    # Actually, let's just start from a reasonable lower bound
    
    # The rank of 2x3x3 is known to be 8 (or 16 depending on source)
    # For 2x3x9, we can use 3 blocks of 2x3x3 giving rank 24
    # But there might be a better decomposition
    
    # Try ranks from 20 to 53
    for r in range(20, rank_sb):
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        # Try integer search first (more likely to find exact solutions)
        U, V, W, found = integer_search(r, max_trials=150)
        if found:
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
        
        # Try ALS
        U, V, W, found = als_search(r, max_trials=30)
        if found:
            # Check if we can snap to half-integers
            if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                U_snapped = snap_half_integer(U)
                V_snapped = snap_half_integer(V)
                W_snapped = snap_half_integer(W)
                if verify(U_snapped, V_snapped, W_snapped):
                    best_rank = r
                    best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                    break
            else:
                # Keep as is if it's exact
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break

    # If we haven't found a good solution, try structured initialization
    if best_rank >= 24 and time.time() < deadline - 20:
        # Use block structure: 2x3x9 = 3 copies of 2x3x3
        # Initialize with block-diagonal structure
        r_struct = 24
        U_struct = np.zeros((nm, r_struct))
        V_struct = np.zeros((mp, r_struct))
        W_struct = np.zeros((pn, r_struct))
        
        for b in range(3):
            col_start = b * 8
            np.random.seed(777 + b)
            U_block = np.random.randn(nm, 8) * 0.1
            V_block = np.random.randn(m * 3, 8) * 0.1
            W_block = np.random.randn(3 * n, 8) * 0.1
            
            U_struct[:, col_start:col_start + 8] = U_block
            
            for j in range(m):
                for k_local in range(3):
                    k = 3 * b + k_local
                    v_row = j * p + k
                    v2_row = j * 3 + k_local
                    V_struct[v_row, col_start:col_start + 8] = V_block[v2_row, :]
            
            for k_local in range(3):
                k = 3 * b + k_local
                for i in range(n):
                    w_row = k * n + i
                    w2_row = k_local * n + i
                    W_struct[w_row, col_start:col_start + 8] = W_block[w2_row, :]
        
        # Refine with ALS
        U_ref, V_ref, W_ref = U_struct, V_struct, W_struct
        prev_error = np.inf
        for iteration in range(300):
            if time.time() > deadline - 10:
                break
            U_ref, V_ref, W_ref = als_step(T, U_ref, V_ref, W_ref, nm, mp, pn, r_struct)
            error = verify_error(U_ref, V_ref, W_ref)
            if error < 1e-12:
                if r_struct < best_rank:
                    best_rank = r_struct
                    best_U, best_V, best_W = U_ref, V_ref, W_ref
                break
            elif error > prev_error * 1.5 and iteration > 10:
                break
            prev_error = error

    # Final polish: try to snap to half-integers if close
    if time.time() < deadline - 5:
        if is_half_integer(best_U, tol=1e-4) and is_half_integer(best_V, tol=1e-4) and is_half_integer(best_W, tol=1e-4):
            U_snapped = snap_half_integer(best_U)
            V_snapped = snap_half_integer(best_V)
            W_snapped = snap_half_integer(best_W)
            if verify(U_snapped, V_snapped, W_snapped, tol=1e-8):
                best_U, best_V, best_W = U_snapped, V_snapped, W_snapped

    if time.time() > deadline - 3:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
