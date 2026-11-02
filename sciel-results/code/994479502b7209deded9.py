import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 120.0

def _build_tensor_2x2x7():
    # T[i,k, i_A, j_A, j_B, k_B] = 1 if i=i_A, j_A=j_B, k=k_B
    # Shape: (14, 4, 14)
    T = np.zeros((14, 4, 14))
    for i in range(2):
        for j in range(2):
            for k in range(7):
                out_idx = k * 2 + i
                u_idx = i * 2 + j
                v_idx = j * 7 + k
                T[out_idx, u_idx, v_idx] = 1.0
    return T

def _als_decompose(T, r, n_iter=200, seed=0, tol=1e-10):
    I, J, K = T.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((I, r))
    V = rng.standard_normal((J, r))
    W = rng.standard_normal((K, r))
    for t in range(r):
        nrm = max(1e-12, np.sqrt(np.sum(U[:, t]**2) + np.sum(V[:, t]**2) + np.sum(W[:, t]**2)))
        U[:, t] /= nrm
        V[:, t] /= nrm
        W[:, t] /= nrm
    best_err = np.inf
    for it in range(n_iter):
        VW = np.einsum('jt,kt->jkt', V, W).reshape(J * K, r)
        for i in range(I):
            rhs = T[i, :, :].reshape(-1)
            sol, *_ = np.linalg.lstsq(VW, rhs, rcond=None)
            U[i, :] = sol
        UW = np.einsum('it,kt->ikt', U, W).reshape(I * K, r)
        for j in range(J):
            rhs = T[:, j, :].reshape(-1)
            sol, *_ = np.linalg.lstsq(UW, rhs, rcond=None)
            V[j, :] = sol
        UV = np.einsum('it,jt->ijt', U, V).reshape(I * J, r)
        for k in range(K):
            rhs = T[:, :, k].reshape(-1)
            sol, *_ = np.linalg.lstsq(UV, rhs, rcond=None)
            W[k, :] = sol
        recon = np.einsum('it,jt,kt->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
        if err < tol:
            break
    if best_err < 1e-6:
        return U, V, W, best_err
    return None

def _find_2x2x7_decomp(max_seeds=20, n_iter=150):
    T = _build_tensor_2x2x7()
    for r in range(14, 29):
        for seed in range(max_seeds):
            res = _als_decompose(T, r, n_iter=n_iter, seed=seed)
            if res is not None:
                U, V, W, err = res
                return r, U, V, W
    return None

def _combine_2x2x7_decomps(r, U1, V1, W1, U2, V2, W2):
    # Combine two 2x2x7 decompositions into a 2x4x7 decomposition
    # U1, U2: (14, r) -> U: (14, 2r)
    # V1, V2: (4, r) -> V: (28, 2r)  [V1 for block 0, V2 for block 1]
    # W1, W2: (14, r) -> W: (14, 2r)
    U = np.concatenate([U1, U1], axis=1)  # Wait, this is wrong
    # Actually, the output is the same for both blocks, so U should be the same
    # But the U factor in the full 2x4x7 tensor is (8, rank), not (14, rank)
    # I'm confusing myself. Let me reconsider.
    pass

def _to_int_or_half(M):
    out = []
    for row in M:
        newrow = []
        for x in row:
            v = round(2 * x) / 2.0
            if abs(v - round(v)) < 1e-6:
                v = float(round(v))
            newrow.append(v)
        out.append(newrow)
    return out

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # Try to find a 2x2x7 decomposition
    res = _find_2x2x7_decomp(max_seeds=10, n_iter=100)
    if res is not None:
        r_sub, U_sub, V_sub, W_sub = res
        # Combine two 2x2x7 decompositions into a 2x4x7 decomposition
        # Full tensor: T[i,k, i_A, j_block*2+j_local, j_block'*2+j_local', k_B]
        # = delta(i,i_A) * delta(j_block,j_block') * delta(j_local,j_local') * delta(k,k_B)
        # Block 0: j_block=0, j_local=j_A, j_local'=j_B
        # Block 1: j_block=1, j_local=j_A, j_local'=j_B
        # U factor: (8, 2*r_sub) where 8 = n*m = 2*4
        # V factor: (28, 2*r_sub) where 28 = m*p = 4*7
        # W factor: (14, 2*r_sub) where 14 = p*n = 7*2
        # For block 0: U0[i_A*2+j_local, t] = U_sub[i_A*2+j_local, t] for t=0..r_sub-1
        #             V0[j_local*7+k, t] = V_sub[j_local*7+k, t] for t=0..r_sub-1
        #             W0[k*2+i_A, t] = W_sub[k*2+i_A, t] for t=0..r_sub-1
        # For block 1: U1[i_A*2+2+j_local, t+r_sub] = U_sub[i_A*2+j_local, t] for t=0..r_sub-1
        #             V1[(j_local+2)*7+k, t+r_sub] = V_sub[j_local*7+k, t] for t=0..r_sub-1
        #             W1[k*2+i_A, t+r_sub] = W_sub[k*2+i_A, t] for t=0..r_sub-1
        r_full = 2 * r_sub
        U_full = np.zeros((8, r_full))
        V_full = np.zeros((28, r_full))
        W_full = np.zeros((14, r_full))
        # Block 0
        for i_A in range(2):
            for j_local in range(2):
                for t in range(r_sub):
                    U_full[i_A * 2 + j_local, t] = U_sub[i_A * 2 + j_local, t]
        for j_local in range(2):
            for k in range(7):
                for t in range(r_sub):
                    V_full[j_local * 7 + k, t] = V_sub[j_local * 7 + k, t]
        for k in range(7):
            for i_A in range(2):
                for t in range(r_sub):
                    W_full[k * 2 + i_A, t] = W_sub[k * 2 + i_A, t]
        # Block 1
        for i_A in range(2):
            for j_local in range(2):
                for t in range(r_sub):
                    U_full[i_A * 2 + 2 + j_local, t + r_sub] = U_sub[i_A * 2 + j_local, t]
        for j_local in range(2):
            for k in range(7):
                for t in range(r_sub):
                    V_full[(j_local + 2) * 7 + k, t + r_sub] = V_sub[j_local * 7 + k, t]
        for k in range(7):
            for i_A in range(2):
                for t in range(r_sub):
                    W_full[k * 2 + i_A, t + r_sub] = W_sub[k * 2 + i_A, t]
        # Verify
        T_full = np.zeros((14, 8, 28))
        for i in range(2):
            for j in range(4):
                for k in range(7):
                    out_idx = k * 2 + i
                    u_idx = i * 4 + j
                    v_idx = j * 7 + k
                    T_full[out_idx, u_idx, v_idx] = 1.0
        recon = np.einsum('it,jt,kt->ijk', U_full, V_full, W_full)
        if np.allclose(recon, T_full, atol=1e-5):
            return {"U": _to_int_or_half(U_full), "V": _to_int_or_half(V_full), "W": _to_int_or_half(W_full)}
    # Fall back to schoolbook
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": _to_int_or_half(U), "V": _to_int_or_half(V), "W": _to_int_or_half(W)}
