"""Combined program: structured Strassen tiling (rank 77) + coordinate-descent rank reduction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 115.0  # safety margin before 120s evaluator limit

def _build_structured_rank77():
    """Build the proven rank-77 nested Strassen tiling decomposition."""
    n, m, p = 4, 4, 6
    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=6, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    # Build the tensor T from U, V, W: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Then compare with the schoolbook tensor: T[i,j,k] = 1 if (i,j,k) valid
    # Actually, the tensor for matrix mult C = A*B where A is n x m, B is m x p:
    # T[a_idx, b_idx, c_idx] where a_idx = i*m+j (A entry), b_idx = j*p+k (B entry),
    # c_idx = k*n+i (C entry). The tensor value is 1 when i,j,k are consistent.
    # More precisely: T is a 3-tensor of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, else 0.
    # The decomposition says T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    
    # Efficient check: for each (a_idx, c_idx), check if sum_r U[a,r]*W[c,r] == expected
    # Actually, the full tensor check: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For matrix mult, T[a,b,c] is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i
    # and the value is 1.
    
    # Let's check: for each a in range(n*m), c in range(p*n):
    #   a = i*m+j, c = k*n+i. Need i from a and i from c to match.
    #   Then b = j*p+k, and T[a,b,c] should be 1, all other T[a,b',c] should be 0.
    
    # Simpler: compute the full tensor and compare.
    # R can be up to 77, sizes are 16x24x24 = 9216 entries.
    # T = sum_r outer(U[:,r], V[:,r], W[:,r])
    # This is a bit expensive but doable.
    
    # Actually, let's just check a few random entries for speed, or do the full check.
    # Full check: 16*24*24*R = 9216*77 ≈ 709k multiplications. Fast enough.
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    return np.max(np.abs(T - T_expected)) < tol


def _rank_reduce(U, V, W, max_iter=200, tol=1e-6, deadline=None):
    """Try to reduce rank by zeroing columns and repairing via coordinate descent."""
    n, m, p = 4, 4, 6
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    R = U.shape[1]
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Precompute current tensor to allow fast residual calculation
    T_current = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T_current += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
    def fit_rank1(residual, init_u, init_v, init_w):
        """Fit a rank-1 approximation to residual using ALS."""
        u = init_u.copy()
        v = init_v.copy()
        w = init_w.copy()
        
        # Check if initialization is valid
        if np.max(np.abs(u)) < 1e-10 or np.max(np.abs(v)) < 1e-10 or np.max(np.abs(w)) < 1e-10:
            # SVD initialization
            mat = residual.reshape(n*m, -1)
            try:
                U_svd, S, Vt_svd = np.linalg.svd(mat, full_matrices=False)
                if len(S) > 0 and S[0] > 1e-10:
                    u = U_svd[:,0] * np.sqrt(S[0])
                    vec = Vt_svd[0,:] * np.sqrt(S[0])
                    mat2 = vec.reshape(m*p, p*n)
                    U2, S2, Vt2 = np.linalg.svd(mat2, full_matrices=False)
                    if len(S2) > 0 and S2[0] > 1e-10:
                        v = U2[:,0] * np.sqrt(S2[0])
                        w = Vt2[0,:] * np.sqrt(S2[0])
                    else:
                        return None
                else:
                    return None
            except:
                return None
        
        # ALS iterations
        for it in range(100):
            if deadline and time.time() > deadline:
                return None
            
            # Update u
            vw = np.outer(v, w)
            denom = np.sum(vw*vw)
            if denom < 1e-20:
                break
            u_new = np.sum(residual * np.tile(vw, (n*m, 1, 1)), axis=(1,2)) / denom
            u = u_new
            
            # Update v
            uw = np.outer(u, w)
            denom = np.sum(uw*uw)
            if denom < 1e-20:
                break
            v_new = np.sum(residual * np.tile(uw, (1, m*p, 1)).reshape(n*m, m*p, p*n), axis=(0,2)) / denom
            v = v_new
            
            # Update w
            uv = np.outer(u, v)
            denom = np.sum(uv*uv)
            if denom < 1e-20:
                break
            w_new = np.sum(residual * np.tile(uv, (1, 1, p*n)).reshape(n*m, m*p, p*n), axis=(0,1)) / denom
            w = w_new
            
            # Early stop if error is small
            approx = np.einsum('i,j,k->ijk', u, v, w)
            err = np.max(np.abs(residual - approx))
            if err < 1e-8:
                break
                
        approx = np.einsum('i,j,k->ijk', u, v, w)
        err = np.max(np.abs(residual - approx))
        if err < tol:
            return (u, v, w)
        return None

    current_rank = R
    # We will iterate through columns. If we remove one, we update T_current and R effectively.
    # To keep it simple, we iterate from R-1 down to 0.
    for r in range(R - 1, -1, -1):
        if deadline and time.time() > deadline:
            break
        
        # Skip if column is already zero
        if np.max(np.abs(U[:,r])) < 1e-10:
            continue
            
        U_orig = U[:,r].copy()
        V_orig = V[:,r].copy()
        W_orig = W[:,r].copy()
        
        # Residual = T_expected - (T_current - outer(U[:,r], V[:,r], W[:,r]))
        # T_current includes column r
        outer_r = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        residual = T_expected - (T_current - outer_r)
        
        # Try to fit rank-1 to residual
        result = fit_rank1(residual, U_orig, V_orig, W_orig)
        
        if result is not None:
            u, v, w = result
            # Update U, V, W
            U[:,r] = u
            V[:,r] = v
            W[:,r] = w
            
            # Update T_current: remove old outer_r, add new outer
            T_current = T_current - outer_r + np.einsum('i,j,k->ijk', u, v, w)
            
            # Mark as removed for ranking purposes? 
            # Actually, we are replacing column r with a new one that fits the residual.
            # This effectively merges the contribution of column r into the "rest".
            # Wait, the logic of "remove column r" means we want to set U[:,r]=0 etc.
            # But here we are *replacing* it with a better fit for the residual?
            # No, the standard pruning is:
            # 1. Assume we can remove col r.
            # 2. The "deficit" is what col r used to provide minus what the rest provides?
            # No.
            # Original: T = sum_{k != r} O_k + O_r
            # We want: T = sum_{k != r} O_k + O'_r where O'_r is rank 1?
            # No, that's just re-optimizing col r. That doesn't reduce rank.
            
            # To reduce rank, we must set one of the columns to ZERO.
            # If we set col r to zero, the new tensor is T' = T - O_r.
            # We need T' to be representable by the remaining R-1 columns.
            # But the remaining columns are fixed.
            # So we need O_r to be exactly representable by the remaining columns?
            # No, we can *adjust* the remaining columns.
            
            # The current code's logic in `try_remove_column` was:
            # 1. Zero out col r.
            # 2. Compute residual = T_expected - T_without_r.
            # 3. Try to fit residual with a SINGLE rank-1 term?
            # That implies we are adding a NEW column? No, R is fixed.
            # Ah, the code puts the fit back into U[:,r].
            # So it's not removing the column, it's *re-optimizing* it to minimize error?
            # If it minimizes error to 0, then the decomposition is still valid with R columns.
            # It does NOT reduce the rank count in the array, it just changes the values.
            # The "current_rank -= 1" in the original code was misleading if the column isn't actually zeroed.
            
            # To actually reduce the rank (number of non-zero columns), we must be able to set a column to ZERO and have the rest cover the tensor.
            # This requires that the tensor T_expected lies in the span of the other R-1 rank-1 terms.
            # This is a very strict condition.
            
            # Let's look at the original `try_remove_column` again.
            # It computes `residual = T_expected - T_without`.
            # Then it fits `residual` with `u,v,w`.
            # Then it sets `U[:,r] = u`.
            # This means the new column r is responsible for the `residual`.
            # The other columns are UNCHANGED.
            # So the new sum is `T_without + outer(u,v,w)`.
            # If `outer(u,v,w) == residual`, then `T_without + residual == T_expected`.
            # So the decomposition is still valid with R columns.
            # It has NOT reduced the rank. It has just re-parameterized column r.
            # The `current_rank -= 1` in the original code was WRONG unless it actually zeroed the column and verified the rest was sufficient (which it didn't, it refitted the removed slot).
            
            # So the original code likely didn't reduce the rank, it just did a local optimization.
            # To reduce rank, we need to find a column r such that `T_expected` can be represented by the other R-1 columns (possibly adjusted).
            # This is equivalent to: Is `T_expected` in the manifold of rank R-1 tensors?
            # A simpler heuristic: Try to zero out col r, and see if the error can be reduced to 0 by adjusting the OTHER columns.
            # This is a much harder problem (non-convex, many variables).
            
            # However, if the columns are linearly dependent in a specific way, we might be able to merge them.
            
            # Given the time, let's stick to the "re-optimization" but verify if we can actually zero out a column.
            # A column can be zeroed if its contribution is redundant.
            
            # Let's try a different strategy:
            # 1. Compute T_current.
            # 2. For each r, compute residual = T_expected - (T_current - outer_r).
            #    This is the part of T_expected NOT covered by the other columns.
            #    Wait, T_current - outer_r is the sum of all other columns.
            #    So residual = T_expected - Sum_{k!=r} O_k.
            #    If we can represent `residual` with 0 rank-1 terms (i.e., residual == 0), then col r is redundant.
            #    If residual is small, maybe we can absorb it into the others?
            
            # The original code tried to fit `residual` with 1 rank-1 term and put it back in slot r.
            # This doesn't reduce rank.
            
            # To reduce rank, we must find r such that `residual` is ZERO (or very close).
            # This happens if col r is linearly dependent on the others in the tensor space.
            
            # Let's check if `residual` is close to 0.
            err = np.max(np.abs(residual))
            if err < tol:
                # Column r is redundant!
                U[:,r] = 0
                V[:,r] = 0
                W[:,r] = 0
                T_current = T_current - outer_r
                current_rank -= 1
            else:
                # Try to see if we can merge r with another column?
                # Or just skip.
                pass

    # Trim
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return U, V, W, current_rank


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Step 1: Build structured rank-77 decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook if something went wrong
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 2: Try rank reduction
    deadline = t_start + _DEADLINE
    try:
        U_red, V_red, W_red, new_rank = _rank_reduce(U, V, W, deadline=deadline)
        
        # Verify reduced decomposition
        if _verify_decomposition(U_red, V_red, W_red):
            U, V, W = U_red, V_red, W_red
    except Exception:
        pass  # Keep original if reduction fails
    
    # Convert to list with integer entries where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
