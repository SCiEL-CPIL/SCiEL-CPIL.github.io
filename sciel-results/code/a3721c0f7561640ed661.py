import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0  # safety margin before 150s kill

    # Target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    # ---- Strassen 2x2x2 block (rank 7) ----
    def strassen_block(n2, m2, p2):
        U = np.zeros((n2*m2, 7))
        V = np.zeros((m2*p2, 7))
        W = np.zeros((p2*n2, 7))
        a00, a01, a10, a11 = 0, 1, 2, 3
        b00, b01, b10, b11 = 0, 1, 2, 3
        c00, c01, c10, c11 = 0, 1, 2, 3
        # M1 = (A00+A11)(B00+B11)
        U[a00,0]+=1; U[a11,0]+=1; V[b00,0]+=1; V[b11,0]+=1
        W[c00,0]+=1; W[c11,0]+=1
        # M2 = (A10+A11)B00
        U[a10,1]+=1; U[a11,1]+=1; V[b00,1]+=1
        W[c10,1]+=1; W[c11,1]-=1
        # M3 = A00(B01-B11)
        U[a00,2]+=1; V[b01,2]+=1; V[b11,2]-=1
        W[c01,2]+=1; W[c11,2]+=1
        # M4 = A11(B10-B00)
        U[a11,3]+=1; V[b10,3]+=1; V[b00,3]-=1
        W[c00,3]+=1; W[c10,3]+=1
        # M5 = (A00+A01)B11
        U[a00,4]+=1; U[a01,4]+=1; V[b11,4]+=1
        W[c00,4]-=1; W[c01,4]+=1
        # M6 = (A10-A00)(B00+B01)
        U[a10,5]+=1; U[a00,5]-=1; V[b00,5]+=1; V[b01,5]+=1
        W[c11,5]+=1
        # M7 = (A01-A11)(B10+B11)
        U[a01,6]+=1; U[a11,6]-=1; V[b10,6]+=1; V[b11,6]+=1
        W[c00,6]+=1
        return U, V, W

    # ---- Schoolbook block for nx1xp (rank n*p) ----
    def schoolbook_block(n2, p2):
        rank = n2*p2
        U = np.zeros((n2*1, rank))
        V = np.zeros((1*p2, rank))
        W = np.zeros((p2*n2, rank))
        c = 0
        for i in range(n2):
            for k in range(p2):
                U[i, c] = 1.0
                V[k, c] = 1.0
                W[k*n2+i, c] = 1.0
                c += 1
        return U, V, W

    # ---- Block-diagonal decomposition: rank 7 + 10 = 17 ----
    # j in {0,1} handled by Strassen (2x2x5), j in {2} by schoolbook (2x1x5)
    U1, V1, W1 = strassen_block(n, 2, p)
    U2, V2, W2 = schoolbook_block(n, p)

    U = np.zeros((n*m, 17))
    V = np.zeros((m*p, 17))
    W = np.zeros((p*n, 17))
    # Strassen block occupies j indices 0,1
    U[:2*2, :7] = U1
    V[:2*p, :7] = V1
    W[:, :7] = W1
    # Schoolbook block occupies j index 2
    U[2*2:3*2, 7:] = U2
    V[2*p:3*p, 7:] = V2
    W[:, 7:] = W2

    # Verify the block construction is exact
    if not verify(U, V, W):
        # Fallback to pure schoolbook (rank 30)
        rank = n*m*p
        Us = np.zeros((n*m, rank))
        Vs = np.zeros((m*p, rank))
        Ws = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c] = 1.0
                    Vs[j*p+k, c] = 1.0
                    Ws[k*n+i, c] = 1.0
                    c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 17

    # ---- Try to reduce rank by zeroing columns (polynomial solve) ----
    # For a given set of "free" columns F and "solve" columns S, check if S can be
    # zeroed out by adjusting F so the tensor still equals T.
    def try_reduce(U0, V0, W0, target_rank):
        """Try to achieve target_rank by zeroing (17-target_rank) columns."""
        r0 = U0.shape[1]
        r_solve = r0 - target_rank
        if r_solve <= 0:
            return U0, V0, W0
        # Indices to solve for (last r_solve columns)
        solve_idx = list(range(r0 - r_solve, r0))
        free_idx = list(range(0, r0 - r_solve))

        # We want to find delta (r0 - r_solve) such that
        # sum_f U[:,f] V[:,f] W[:,f] + sum_s U[:,s] V[:,s] W[:,s] = T
        # => sum_f U[:,f] V[:,f] W[:,f] = T - sum_s U[:,s]V[:,s]W[:,s]
        # The LHS is linear in the free columns (we can scale them). Actually we
        # can adjust free columns arbitrarily. Let's set up a linear system.
        #
        # Let the free columns be variables. The contribution of free columns is
        # linear in each free column's entries. So we have a linear system in the
        # (n*m + m*p + p*n) * (r0 - r_solve) unknowns... that's large.
        #
        # Alternative: use the fact that we can set solve columns to zero and
        # solve for free columns via least squares on the tensor entries.
        # But the tensor is bilinear in U,V,W, not linear in all at once.
        #
        # Simpler approach: use gradient descent to zero out the solve columns
        # while keeping the tensor equal to T.
        # Actually, the cleanest approach: treat this as a rank-reduction via
        # null-space. If the current decomposition has rank r0, we can try to
        # find a linear combination of columns that gives zero tensor, then
        # use it to eliminate a column.
        #
        # For now, use a simple approach: try random linear combinations to
        # zero out columns.
        start = time.time()
        for attempt in range(200):
            if time.time() > deadline or time.time() - start > 20.0:
                break
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            # Try to zero out solve columns by absorbing their contribution
            # into free columns via gradient descent on the constraint that
            # the tensor equals T and solve columns are zero.
            # This is complex; instead, just try to find if the current
            # decomposition already has linearly dependent columns.
            # Check rank of U, V, W
            rank_U = np.linalg.matrix_rank(U)
            rank_V = np.linalg.matrix_rank(V)
            rank_W = np.linalg.matrix_rank(W)
            if max(rank_U, rank_V, rank_W) < r0:
                # There's a linear dependency; we can reduce rank
                # Find null vector and eliminate
                for mat, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
                    if np.linalg.matrix_rank(mat) < r0:
                        # Find null vector
                        _, _, Vt = np.linalg.svd(mat)
                        null_vec = Vt[-1]
                        # Eliminate the last column using null vector
                        # This is a simplification; proper elimination is complex
                        break
            # If we can't easily reduce, break
            break
        return None

    # The rank-reduction via linear algebra is complex; instead, use a
    # gradient-based approach to try to find a lower-rank decomposition.
    # But given the block structure is already rank 17, and the known lower
    # bound for 2x3x5 is likely around 15-17, we'll try a few optimization
    # passes to see if we can do better.

    # Try gradient descent to find rank-16 decomposition
    def try_optimize(target_rank, max_time=25.0):
        start = time.time()
        for restart in range(3):
            if time.time() > deadline or time.time() - start > max_time:
                break
            # Initialize near the block decomposition
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            # Copy first target_rank columns from block decomposition
            if target_rank <= 17:
                U[:, :target_rank] = U0[:, :target_rank] if 'U0' in dir() else best_U[:, :target_rank]
                V[:, :target_rank] = V0[:, :target_rank] if 'V0' in dir() else best_V[:, :target_rank]
                W[:, :target_rank] = W0[:, :target_rank] if 'W0' in dir() else best_W[:, :target_rank]
            # Add small perturbation
            U += np.random.randn(*U.shape) * 0.01
            V += np.random.randn(*V.shape) * 0.01
            W += np.random.randn(*W.shape) * 0.01

            lr = 0.05
            for step in range(300):
                if time.time() > deadline or time.time() - start > max_time:
                    break
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T_recon - T
                err = np.sum(R**2)
                if err < 1e-14:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_r, V_r, W_r
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_r, V_r, W_r
                    break
                grad_U = 2 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', R, U, V)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                lr *= 0.98
        return None

    # Try to find rank 16 or lower
    for target_rank in [16, 15]:
        if time.time() > deadline:
            break
        result = try_optimize(target_rank, max_time=20.0)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break

    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to the verified block decomposition (rank 17)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = 17
        if not verify(best_U, best_V, best_W):
            # Ultimate fallback: schoolbook
            rank = n*m*p
            Us = np.zeros((n*m, rank))
            Vs = np.zeros((m*p, rank))
            Ws = np.zeros((p*n, rank))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Us[i*m+j, c] = 1.0
                        Vs[j*p+k, c] = 1.0
                        Ws[k*n+i, c] = 1.0
                        c += 1
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
