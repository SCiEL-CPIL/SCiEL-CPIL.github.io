import numpy as np
import time

def construct():
    t0 = time.time()
    HARD = 146.0

    def remaining():
        return HARD - time.time()

    # ---- 6x6 MM tensor T[i,k,j] = [A*B]_{i,k} as a function of A(i,j),B(j,k) ----
    # T = kron(T22, T33); flat a=i*6+j, b=j*6+k, c=i*6+k
    def mm_tensor(n):
        T = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    T[i, k, j] = 1.0
        return T

    T22 = mm_tensor(2)
    T33 = mm_tensor(3)
    T66 = np.zeros((6, 6, 6))
    for i2 in range(2):
        for k2 in range(2):
            for j2 in range(2):
                v = T22[i2, k2, j2]
                if v != 0:
                    for i3 in range(3):
                        for k3 in range(3):
                            for j3 in range(3):
                                T66[i2 * 3 + i3, k2 * 3 + k3, j2 * 3 + j3] += v * T33[i3, k3, j3]

    E = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    def mk(d):
        vec = np.zeros(4)
        for a, c in d.items():
            vec[a] = c
        return vec
    S22 = [
        (mk({E[0]: 1, E[3]: 1}),   mk({E[0]: 1, E[3]: 1}),   mk({E[0]: 1, E[3]: 1})),
        (mk({E[2]: 1, E[3]: 1}),   mk({E[0]: 1}),            mk({E[2]: 1, E[3]: -1})),
        (mk({E[0]: 1}),            mk({E[1]: 1, E[3]: -1}),  mk({E[1]: 1, E[3]: 1})),
        (mk({E[3]: 1}),            mk({E[2]: 1, E[0]: -1}),  mk({E[0]: 1, E[2]: 1})),
        (mk({E[0]: 1, E[1]: 1}),   mk({E[3]: 1}),            mk({E[0]: -1, E[1]: 1})),
        (mk({E[2]: 1, E[0]: -1}),  mk({E[0]: 1, E[1]: 1}),   mk({E[3]: 1})),
        (mk({E[1]: 1, E[3]: -1}),  mk({E[2]: 1, E[3]: 1}),   mk({E[0]: 1})),
    ]

    # Compose 2x2 (rank7) x 3x3 terms -> 6x6 U,V,W (36 x R)
    # flat: A index = (i2*3+i3)*6 + (j2*3+j3); B = (j2*3+j3)*6 + (k2*3+k3); C = (i2*3+i3)*6 + (k2*3+k3)
    def compose(terms33):
        R = 7 * len(terms33)
        U = np.zeros((36, R)); V = np.zeros((36, R)); W = np.zeros((36, R))
        for r3, (u3, v3, w3) in enumerate(terms33):
            for r2, (u2, v2, w2) in enumerate(S22):
                c = r3 * 7 + r2
                for a2 in range(4):
                    if u2[a2] != 0.0:
                        i2, j2 = E[a2]
                        for a3 in range(9):
                            if u3[a3] != 0.0:
                                i3, j3 = a3 // 3, a3 % 3
                                U[(i2 * 3 + i3) * 6 + (j2 * 3 + j3), c] += u2[a2] * u3[a3]
                for b2 in range(4):
                    if v2[b2] != 0.0:
                        j2, k2 = E[b2]
                        for b3 in range(9):
                            if v3[b3] != 0.0:
                                j3, k3 = b3 // 3, b3 % 3
                                V[(j2 * 3 + j3) * 6 + (k2 * 3 + k3), c] += v2[b2] * v3[b3]
                for c2 in range(4):
                    if w2[c2] != 0.0:
                        i2, k2 = E[c2]
                        for c3 in range(9):
                            if w3[c3] != 0.0:
                                i3, k3 = c3 // 3, c3 % 3
                                W[(i2 * 3 + i3) * 6 + (k2 * 3 + k3), c] += w2[c2] * w3[c3]
        return U, V, W

    # Correct verification: T[i,k,j] = sum_t U[i,t] V[k,t] W[j,t]
    def verify(U, V, W, tol=1e-6):
        R = U.shape[1]
        for j in range(6):
            # slice = sum_t W[j,t] * (U[:,t] outer V[:,t])
            sl = np.zeros((6, 6))
            for t in range(R):
                wt = W[j, t]
                if wt != 0.0:
                    sl += wt * np.outer(U[:, t], V[:, t])
            if np.linalg.norm(sl - T66[:, :, j]) > tol:
                return False
        return True

    # rank of a composed decomposition = rank of the U@V^T structure (with W free)
    def composed_rank(U, V, W):
        R = U.shape[1]
        M = U @ V.T  # 6x6, must equal T66 @ W
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > 1e-6))

    # schoolbook 3x3 terms (rank27) -> baseline rank 189
    school33 = []
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                u3 = np.zeros(9); v3 = np.zeros(9); w3 = np.zeros(9)
                u3[i3 * 3 + j3] = 1; v3[j3 * 3 + k3] = 1; w3[k3 * 3 + i3] = 1
                school33.append((u3, v3, w3))

    U, V, W = compose(school33)
    assert verify(U, V, W), "baseline must verify"
    best_rank = composed_rank(U, V, W)  # 189
    best_terms = school33

    # ---- Search: find C (9x9) so that T33c @ C has low rank; T33c[c_idx, a_idx] ----
    T33c = T33.reshape(9, 9)  # row c=i3*3+k3, col a=i3*3+j3
    rng = np.random.default_rng(20240606)

    def rank_of(M, tol=1e-6):
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > tol))

    def build_terms33_from_C(C):
        M = T33c @ C  # 9x9 = U33 V33^T
        Uf, Sf, Vtf = np.linalg.svd(M, full_matrices=False)
        r2 = int(np.sum(Sf > 1e-6))
        U33 = Uf[:, :r2] * np.sqrt(Sf[:r2])
        V33 = Vtf[:r2, :].T * np.sqrt(Sf[:r2])
        terms = [(U33[:, t], V33[:, t], C[:, t]) for t in range(r2)]
        return terms

    def try_target(rank_target, n_iter):
        for _ in range(n_iter):
            if remaining() < 12.0:
                return None
            C = rng.integers(-1, 2, size=(9, 9)).astype(float)
            if rank_of(T33c @ C) > rank_target:
                continue
            terms33 = build_terms33_from_C(C)
            U3, V3, W3 = compose(terms33)
            if not verify(U3, V3, W3, 1e-5):
                continue
            r = composed_rank(U3, V3, W3)
            if r <= rank_target:
                return (terms33, r)
        return None

    for target in [49, 56, 63]:
        if best_rank <= target:
            continue
        if remaining() < 15.0:
            break
        res = try_target(target, n_iter=6000)
        if res is not None:
            terms33, r = res
            if r < best_rank:
                best_rank = r
                best_terms = terms33

    # final build
    U, V, W = compose(best_terms)
    if not verify(U, V, W, 1e-5):
        U, V, W = compose(school33)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
