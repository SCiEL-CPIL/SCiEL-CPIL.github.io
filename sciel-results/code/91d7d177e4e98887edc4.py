import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _build_rank21_strassen():
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_blocks = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                if l == 0: v_idx = 0 * p + 2 * b
                elif l == 1: v_idx = 0 * p + 2 * b + 1
                elif l == 2: v_idx = 1 * p + 2 * b
                elif l == 3: v_idx = 1 * p + 2 * b + 1
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    
    return U, V, W

def _find_linear_dependency(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    A = np.zeros((nm * mp * pn, rank))
    idx = 0
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                for r in range(rank):
                    A[idx, r] = U[i, r] * V[j, r] * W[k, r]
                idx += 1
    
    try:
        _, s, vh = np.linalg.svd(A, full_matrices=True)
        tol = 1e-10
        num_nonzero = np.sum(s > tol)
        if num_nonzero < rank:
            return vh[num_nonzero:rank, :]
    except Exception:
        pass
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    U, V, W = _build_rank21_strassen()
    rank = U.shape[1]
    
    null_vecs = _find_linear_dependency(U, V, W)
    
    if null_vecs is not None and null_vecs.shape[0] > 0:
        c = null_vecs[0]
        drop_idx = int(np.argmax(np.abs(c)))
        mask = [i for i in range(rank) if i != drop_idx]
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
