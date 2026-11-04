import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication.
    # U2 indexed by (i,j) -> i*2+j ; V2 by (j,k) -> j*2+k ; W2 by (k,i) -> k*2+i
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=float)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1,  0,  1,  0,  1]
    ], dtype=float)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=float)

    # Kronecker product gives 4x4 from 2x2: rank 49.
    U_kron = np.kron(U2, U2)  # shape (16, 49), row index (a_out*2+a_in)*4 + (b_out*2+b_in)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)

    # Map standard row-major index S -> Kronecker row index K.
    # For U: S = i*4+j -> i=i1*2+i2, j=j1*2+j2 -> K=(i1*2+j1)*4+(i2*2+j2)
    # For V: S = j*4+k -> j=j1*2+j2, k=k1*2+k2 -> K=(j1*2+k1)*4+(j2*2+k2)
    # For W: S = k*4+i -> k=k1*2+k2, i=i1*2+i2 -> K=(k1*2+i1)*4+(k2*2+i2)
    def std_to_kron(S, first, second):
        a = S // 4
        b = S % 4
        a1, a2 = a // 2, a % 2
        b1, b2 = b // 2, b % 2
        return (a1 * 2 + b1) * 4 + (a2 * 2 + b2)

    perm = np.array([std_to_kron(S, 'f', 's') for S in range(16)], dtype=int)

    U = U_kron[perm, :].copy()  # shape (16, 49)
    V = V_kron[perm, :].copy()
    W = W_kron[perm, :].copy()

    # Verification helper: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] must equal 1.
    def verify(Um, Vm, Wm, tol=1e-9):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    ui = Um[i * 4 + j, :]
                    vj = Vm[j * 4 + k, :]
                    wk = Wm[k * 4 + i, :]
                    if abs(np.dot(ui, vj * wk) - 1.0) > tol:
                        return False
        return True

    # Guarantee the base rank-49 decomposition is correct before doing anything else.
    base_ok = verify(U, V, W)

    # Schoolbook rank-64 fallback (always valid).
    def schoolbook():
        rank_fb = 64
        Uf = np.zeros((16, rank_fb), dtype=float)
        Vf = np.zeros((16, rank_fb), dtype=float)
        Wf = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    Uf[i * 4 + j, col] = 1.0
                    Vf[j * 4 + k, col] = 1.0
                    Wf[k * 4 + i, col] = 1.0
                    col += 1
        return Uf, Vf, Wf

    if not base_ok:
        Uf, Vf, Wf = schoolbook()
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    def best_result():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the constraint matrix for rank reduction.
    # For each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1  (64 equations).
    A_full = np.zeros((64, 49))
    b_vec = np.ones(64)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    A_full[idx, r] = U[i * 4 + j, r] * V[j * 4 + k, r] * W[k * 4 + i, r]
                idx += 1

    active = set(range(49))

    # Fast pre-pass: drop any column that is entirely zero (trivially removable).
    for r in range(49):
        if time.time() - t0 > deadline - 25.0:
            break
        if not np.any(A_full[:, r]):
            active.remove(r)

    # Systematic greedy column elimination: each round try to drop every active column.
    for iteration in range(49):
        if time.time() - t0 > deadline - 25.0:
            break
        progress = False
        for r in sorted(active):
            if time.time() - t0 > deadline - 25.0:
                break
            test_active = list(active - {r})
            A_sub = A_full[:, test_active]
            try:
                c, _, _, _ = np.linalg.lstsq(A_sub, b_vec, rcond=None)
                if np.linalg.norm(A_sub @ c - b_vec) < 1e-9:
                    active.remove(r)
                    progress = True
                    break
            except Exception:
                pass
        if not progress:
            break

    if time.time() - t0 > deadline - 15.0:
        return best_result()

    if len(active) < 49:
        active_list = sorted(active)
        U_red = U[:, active_list]
        V_red = V[:, active_list]
        W_red = W[:, active_list]
        if verify(U_red, V_red, W_red):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    return best_result()

if __name__ == "__main__":
    construct()
