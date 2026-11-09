import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    def build_strassen_decomposition():
        rank = 92
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        blocks = []
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2]:
                    blocks.append(([i_base, i_base + 1],
                                   [j_base, j_base + 1],
                                   [k_base, k_base + 1]))

        skip_set = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        skip_set.add((i, j, k))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in skip_set:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        for I, J, K in blocks:
            i1, i2 = I
            j1, j2 = J
            k1, k2 = K
            # M1 = (a11+a22)(b11+b22)
            U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
            V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
            W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
            col += 1
            # M2 = (a21+a22)b11
            U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
            V[j1*p+k1][col] += 1
            W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1
            col += 1
            # M3 = a11(b12-b22)
            U[i1*m+j1][col] += 1
            V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
            W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1
            col += 1
            # M4 = a22(b21-b11)
            U[i2*m+j2][col] += 1
            V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
            W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1
            col += 1
            # M5 = (a11+a12)b22
            U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
            V[j2*p+k2][col] += 1
            W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1
            col += 1
            # M6 = (a21-a11)(b11+b12)
            U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
            V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
            W[k2*n+i2][col] += 1
            col += 1
            # M7 = (a12-a22)(b21+b22)
            U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
            V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
            W[k1*n+i1][col] += 1
            col += 1

        return U, V, W, rank

    def verify(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1) > tol:
                        return False
        return True

    # Build the base rank-92 decomposition
    U, V, W, rank = build_strassen_decomposition()
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    if not verify(U_np, V_np, W_np):
        # Fallback to standard rank-100 decomposition
        Uf = np.zeros((nm, n*m*p)); Vf = np.zeros((mp, n*m*p)); Wf = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j, col] = 1; Vf[j*p+k, col] = 1; Wf[k*n+i, col] = 1
                    col += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    current_rank = rank

    # Rank-reduction search:
    # For each candidate term t, solve a least-squares problem to express the
    # tensor-contribution of t as a linear combination of the other terms.
    # If the residual is ~0, term t is redundant and can be dropped.
    #
    # Tensor contribution of term r: c_r(i,j,k) = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We want: for all (i,j,k), c_t = sum_{r != t} alpha_r * c_r
    # This is a linear system in alpha. Build a matrix A where A[(i,j,k), r] = c_r(i,j,k)
    # and a vector b[(i,j,k)] = c_t(i,j,k). Solve A[cols != t] alpha = b.

    def tensor_matrix(U, V, W, r):
        """Return matrix of shape (100, r) where row (i,j,k) is the contribution vector."""
        M = np.zeros((n*m*p, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
        return M

    # Build the full tensor matrix once
    A_full = tensor_matrix(U_np, V_np, W_np, current_rank)
    # Row index for (i,j,k)
    def row_idx(i, j, k):
        return i*m*p + j*p + k

    # Try dropping each term
    # Order: try standard (single-index) terms first, they're most likely redundant
    term_order = list(range(current_rank))
    # Standard terms are the first 100-56=... let's identify: standard terms are columns
    # that have exactly one nonzero in each of U, V, W (a single (i,j,k) index).
    # In our build, standard terms come first (col 0..), then Strassen terms.
    # Count standard terms:
    n_std = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # count if not in skip
                in_skip = False
                for ib in [0,2]:
                    for jb in [0,2]:
                        for kb in [0,2]:
                            if i in [ib,ib+1] and j in [jb,jb+1] and k in [kb,kb+1]:
                                in_skip = True
                if not in_skip:
                    n_std += 1
    # So standard terms are columns 0..n_std-1, Strassen terms are n_std..91
    # Prioritize standard terms
    term_order = list(range(n_std)) + list(range(n_std, current_rank))

    improved = False
    for t in term_order:
        if time.time() > deadline - 5:
            break
        if t >= current_rank:
            continue
        # Build A with column t removed
        cols = [c for c in range(current_rank) if c != t]
        A = A_full[:, cols]
        b = A_full[:, t]
        # Solve least squares
        try:
            alpha, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Check residual
        recon = A @ alpha
        resid = np.linalg.norm(b - recon)
        if resid < 1e-7:
            # Term t is redundant! Drop it.
            # Rebuild matrices without column t
            U_np = np.delete(U_np, t, axis=1)
            V_np = np.delete(V_np, t, axis=1)
            W_np = np.delete(W_np, t, axis=1)
            A_full = np.delete(A_full, t, axis=1)
            current_rank -= 1
            improved = True
            # Re-verify
            if not verify(U_np, V_np, W_np):
                # Shouldn't happen, but restore
                U_np = np.array(U, dtype=np.float64)
                V_np = np.array(V, dtype=np.float64)
                W_np = np.array(W, dtype=np.float64)
                A_full = tensor_matrix(U_np, V_np, W_np, 92)
                current_rank = 92
                break

    # Final verification
    if not verify(U_np, V_np, W_np):
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # Remove all-zero columns
    for r in range(current_rank - 1, -1, -1):
        if (np.all(U_np[:, r] == 0) and np.all(V_np[:, r] == 0) and np.all(W_np[:, r] == 0)):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1

    # Snap to integer/half-integer
    def snap(mat):
        out = []
        for i in range(mat.shape[0]):
            row = []
            for r in range(mat.shape[1]):
                val = mat[i, r]
                if abs(val - round(val)) < 1e-9:
                    row.append(int(round(val)))
                elif abs(val - round(val*2)/2) < 1e-9:
                    row.append(round(val*2)/2)
                else:
                    row.append(float(val))
            out.append(row)
        return out

    return {"U": snap(U_np), "V": snap(V_np), "W": snap(W_np)}
