import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44

    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]

    col = 0

    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col

    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)

    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def _verify(Ua, Va, Wa):
        T = np.zeros((n, m, p), dtype=float)
        for r in range(Ua.shape[1]):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=1e-8, rtol=0)

    if not _verify(U_arr, V_arr, W_arr):
        # Fallback: schoolbook rank 48
        rank_s = n * m * p
        Us = [[0] * rank_s for _ in range(dim_u)]
        Vs = [[0] * rank_s for _ in range(dim_v)]
        Ws = [[0] * rank_s for _ in range(dim_w)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j][c] = 1
                    Vs[j * p + k][c] = 1
                    Ws[k * n + i][c] = 1
                    c += 1
        return {"U": Us, "V": Vs, "W": Ws}

    # Rank-reduction via linear dependency detection
    # Each rank-1 term contributes U_r ⊗ V_r ⊗ W_r to the tensor.
    # If sum_r alpha_r (U_r ⊗ V_r ⊗ W_r) = 0, then any term can be dropped
    # by expressing it as a combination of others.
    # We form the 192x44 matrix M where M[:,r] = vec(U_r ⊗ V_r ⊗ W_r) flattened
    # but that's 12*16*12 = 2304 entries. Instead, we check linear dependence
    # by building the tensor contribution matrix and checking rank.
    # 
    # Better approach: the 44 vectors in R^2304. If rank < 44, some are dependent.
    # Build M (2304 x 44) and find null space.
    
    # Flatten each term: term_r = U_r ⊗ V_r ⊗ W_r as a vector of length 2304
    # M[i, r] = U_r[i_u] * V_r[i_v] * W_r[i_w] for flattened index i
    # i = i_u * 16*12 + i_v * 12 + i_w
    
    # Build M efficiently
    M = np.zeros((dim_u * dim_v * dim_w, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()
    
    # Find null space of M (vectors alpha such that M @ alpha = 0)
    # Use SVD
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"U": U, "V": V, "W": W}
    
    # Null space dimension = rank - number of singular values > tol
    tol = 1e-10 * (S[0] if len(S) > 0 else 1.0)
    num_indep = np.sum(S > tol)
    null_dim = rank - num_indep
    
    if null_dim <= 0:
        return {"U": U, "V": V, "W": W}
    
    # Null space vectors are rows of Vt[num_indep:]
    # Vt shape: (rank, rank), null space: Vt[num_indep:, :]
    null_space = Vt[num_indep:, :]  # (null_dim, rank)
    
    # For each null space vector, we can eliminate one term.
    # Strategy: greedily find terms that can be eliminated.
    # A null vector alpha with alpha[r] != 0 means term r can be expressed as 
    # sum_{r' != r} (alpha[r']/alpha[r]) * term_{r'}.
    # But we need the combination to actually work on the tensor.
    
    # Simpler: find columns of M that are linear combinations of others.
    # Use QR with column pivoting to identify independent columns.
    
    # Alternative: iteratively try to remove each column and check if the 
    # remaining columns still span the same tensor.
    
    # Most direct: for each null vector, pick the term with largest |alpha| 
    # and try to remove it. Verify after each removal.
    
    removed = set()
    for alpha in null_space:
        if len(removed) >= null_dim:
            break
        # Find the term with largest coefficient that hasn't been removed
        # and is not the only term contributing to some tensor entry
        best_r = -1
        best_val = 0
        for r in range(rank):
            if r in removed:
                continue
            if abs(alpha[r]) > best_val:
                best_val = abs(alpha[r])
                best_r = r
        
        if best_r < 0 or best_val < tol:
            continue
        
        # Try removing best_r
        test_removed = removed | {best_r}
        keep = [r for r in range(rank) if r not in test_removed]
        if len(keep) == 0:
            continue
        
        U_test = U_arr[:, keep]
        V_test = V_arr[:, keep]
        W_test = W_arr[:, keep]
        
        if _verify(U_test, V_test, W_test):
            removed = test_removed
    
    if removed:
        keep = [r for r in range(rank) if r not in removed]
        U_new = U_arr[:, keep].tolist()
        V_new = V_arr[:, keep].tolist()
        W_new = W_arr[:, keep].tolist()
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U, "V": V, "W": W}
