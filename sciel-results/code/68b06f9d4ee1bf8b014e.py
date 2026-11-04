import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 120.0  # seconds; evaluator kills at 150s

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
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

    # Check for linear dependencies in the rank-1 tensors
    if time.time() - t_start < _DEADLINE - 10:
        # Flatten the tensors
        dim_u, dim_v, dim_w = 16, 24, 24
        flat_dim = dim_u * dim_v * dim_w
        tensors = np.zeros((R, flat_dim))
        
        for r in range(R):
            # u_r is U[:, r], v_r is V[:, r], w_r is W[:, r]
            # Tensor T_r = u_r * v_r * w_r
            # Flatten: T_r[i, j, k] = U[i, r] * V[j, r] * W[k, r]
            # Vector index: (i * dim_v + j) * dim_w + k
            # We can compute this efficiently using outer products
            # But R=77, dims are small.
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            # Create the tensor
            # T[i,j,k] = u_r[i] * v_r[j] * w_r[k]
            # We can do this with broadcasting
            T = np.einsum('i,j,k->ijk', u_r, v_r, w_r)
            tensors[r, :] = T.flatten()
        
        # Compute rank
        # Use SVD to find effective rank
        try:
            _, s_vals, vh = np.linalg.svd(tensors, full_matrices=False)
            # Threshold for rank
            tol = 1e-10
            rank_eff = np.sum(s_vals > tol * s_vals[0])
            
            if rank_eff < R:
                # We can potentially reduce rank
                # For now, just log it. A full reduction is complex.
                # But if rank is significantly lower, we might be able to project.
                # However, projecting onto the basis of the tensors does not necessarily
                # yield a valid rank-1 decomposition of the same form.
                # We will stick with R=77 for safety, but this check is a good diagnostic.
                pass
        except Exception:
            pass

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
