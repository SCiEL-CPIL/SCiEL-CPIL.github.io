import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
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

    # Block 1: A_L B_TL -> C_LL
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_R B_BL -> C_LR
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_L B_TR -> C_RL
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R B_BR -> C_RR
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    # Build the multiplication tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if c corresponds to the same (row,col) as implied by a,b
    # a = i*4+l (A[i,l]), b = l*4+k (B[l,k]), c = k*2+i (C[i,k] via trace convention)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l       # index in U
                b = l * p + k       # index in V
                c = k * n + i       # index in W (trace convention: k*n+i)
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        """Check that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        # Efficient check using einsum
        # Compute S[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-mode tensor product
        # Use: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify rank-28 solution
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}

    # Now search for rank reduction
    # Strategy: check if any of the 28 terms is linearly dependent on the others
    # For term r to be eliminable: there exist coefficients alpha_s (s != r) such that
    # for all (a,b,c): U[a,r]*V[b,r]*W[c,r] = sum_{s!=r} alpha_s * U[a,s]*V[b,s]*W[c,s]
    # This is a system of nm*mp*pn = 1024 equations in 27 unknowns
    # We can set up: for each (a,b,c), the equation is:
    # sum_s alpha_s * (U[a,s]*V[b,s]*W[c,s]) = U[a,r]*V[b,r]*W[c,r]
    # where alpha_r = 0 (we're eliminating term r)
    
    # Build the linear system for each candidate term to eliminate
    # M[a,b,c, s] = U[a,s]*V[b,s]*W[c,s] for s in 0..27
    # RHS[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    
    # Flatten to matrix form: M_flat (1024 x 27) * alpha = RHS_flat (1024,)
    # where M_flat[i, s] = U[a,s]*V[b,s]*W[c,s] for the i-th (a,b,c) triple
    
    # Build M matrix (1024 x 28)
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()
    
    # For each candidate r, check if M[:, r] is in the span of M[:, s] for s != r
    # Use QR or SVD to check linear dependence
    # More efficient: check rank of M vs rank of M without column r
    
    # First, check the rank of M (should be <= 28)
    # If rank(M) < 28, then some columns are linearly dependent
    # We need to find which columns can be removed while maintaining the decomposition
    
    # Compute SVD of M to find the rank
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
        # Count singular values above tolerance
        tol = 1e-10
        num_nonzero = np.sum(S_svd > tol)
        if num_nonzero < rank:
            # There are linear dependencies!
            # Find which columns are in the null space
            # The right singular vectors corresponding to zero singular values give null space
            # But we need to find which individual columns can be removed
            
            # Alternative: for each column r, check if it's in the span of the others
            # by solving the linear system
            best_rank = rank
            best_U = U_arr
            best_V = V_arr
            best_W = W_arr
            
            for r_elim in range(rank):
                if time.time() > deadline - 5:
                    break
                # Build system: find alpha such that M[:, r_elim] = sum_{s != r_elim} alpha_s * M[:, s]
                # This is a least squares problem
                other_cols = [s for s in range(rank) if s != r_elim]
                M_sub = M[:, other_cols]  # 1024 x 27
                rhs = M[:, r_elim]  # 1024
                
                # Solve least squares
                try:
                    alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=tol)
                    # Check residual
                    residual = np.max(np.abs(M_sub @ alpha - rhs))
                    if residual < 1e-8:
                        # Term r_elim is redundant! Remove it
                        new_rank = rank - 1
                        new_U = np.delete(U_arr, r_elim, axis=1)
                        new_V = np.delete(V_arr, r_elim, axis=1)
                        new_W = np.delete(W_arr, r_elim, axis=1)
                        
                        # Verify
                        err = verify_decomp(new_U, new_V, new_W, new_rank)
                        if err < 1e-10:
                            best_rank = new_rank
                            best_U = new_U
                            best_V = new_V
                            best_W = new_W
                            # Try to remove more terms
                            # Rebuild M with the reduced set
                            M = np.zeros((nm * mp * pn, new_rank), dtype=float)
                            for r in range(new_rank):
                                M[:, r] = np.einsum('a,b,c->abc', new_U[:, r], new_V[:, r], new_W[:, r]).flatten()
                            rank = new_rank
                            U_arr = new_U
                            V_arr = new_V
                            W_arr = new_W
                            # Continue searching for more redundancies
                            break
                except np.linalg.LinAlgError:
                    continue
            else:
                # No break, so no more reductions found
                pass
        else:
            # No linear dependencies in the tensor flattening
            pass
    except Exception:
        pass

    # Convert back to lists with integer/rational values
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    U_final = to_list(best_U if 'best_U' in dir() else U_arr)
    V_final = to_list(best_V if 'best_V' in dir() else V_arr)
    W_final = to_list(best_W if 'best_W' in dir() else W_arr)

    return {"U": U_final, "V": V_final, "W": W_final}
