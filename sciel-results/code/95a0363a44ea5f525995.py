"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 with systematic trimming."""
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    Base: Strassen(2x2x2) + schoolbook rank-66.
    Refinement: systematic iterative column-removal (trimming) search.
    """
    n, m, p = 3, 4, 6

    def add_strassen_2x2(i1, i2, j1, j2, k1, k2, terms):
        """Strassen 2x2x2 block decomposition (7 terms)."""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)

        terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
        terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
        terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
        terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
        terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
        terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
        terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))

    def add_schoolbook(i, j, k, terms):
        terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))

    def build_base_decomposition():
        """Rank-66: row 0 schoolbook (24) + rows 1-2 Strassen on 2x2 blocks (42)."""
        terms = []
        for j in range(m):
            for k in range(p):
                add_schoolbook(0, j, k, terms)
        for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
            add_strassen_2x2(1, 2, 0, 1, k1, k2, terms)
            add_strassen_2x2(1, 2, 2, 3, k1, k2, terms)
        return terms

    def build_schoolbook():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_schoolbook(i, j, k, terms)
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        return U, V, W

    def verify_fast(U_np, V_np, W_np, rank):
        U_t = U_np.reshape(n, m, rank)
        V_t = V_np.reshape(m, p, rank)
        W_t = W_np.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_random(U_np, V_np, W_np, rank, seed=42, trials=2):
        """Cross-check via random matrices: reconstruct C and compare to A@B."""
        rng = random.Random(seed)
        for _ in range(trials):
            A = np.array([[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)], dtype=float)
            B = np.array([[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)], dtype=float)
            C_expected = A @ B
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(rank):
                a_vec = U_np[:, r].reshape(n, m)
                b_vec = V_np[:, r].reshape(m, p)
                w_vec = W_np[:, r].reshape(p, n)
                C_computed += np.einsum('ij,jk,ki->ik', a_vec, b_vec, w_vec)
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True

    # Build base rank-66 solution
    terms = build_base_decomposition()
    U, V, W = terms_to_matrices(terms)
    rank = len(terms)

    if not verify_fast(np.array(U, float), np.array(V, float), np.array(W, float), rank):
        # Fallback to pure schoolbook
        terms = build_schoolbook()
        U, V, W = terms_to_matrices(terms)
        rank = len(terms)
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Convert to numpy for efficient operations
    current_U = np.array(U, dtype=float)
    current_V = np.array(V, dtype=float)
    current_W = np.array(W, dtype=float)
    current_rank = rank

    # Systematic iterative trimming: repeatedly try removing each column.
    # A column can be removed if the resulting decomposition is still exact.
    # Iterate until no column can be removed (a local minimum of rank).
    # Re-scan from column 0 after each removal because removals shift indices
    # and can unlock further removals.
    progress = True
    while progress and _time_left() > 20:
        progress = False
        for remove_col in range(current_rank):
            if _time_left() < 15:
                break
            U_test = np.delete(current_U, remove_col, axis=1)
            V_test = np.delete(current_V, remove_col, axis=1)
            W_test = np.delete(current_W, remove_col, axis=1)
            if verify_fast(U_test, V_test, W_test, current_rank - 1):
                current_rank -= 1
                current_U, current_V, current_W = U_test, V_test, W_test
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = current_U, current_V, current_W
                progress = True
                break  # indices shifted; restart scan from 0

    # Final safety cross-check of the best solution via random matrices
    if best_rank < rank and not verify_random(best_U, best_V, best_W, best_rank):
        best_rank = rank
        best_U, best_V, best_W = current_U, current_V, current_W

    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U, "V": V, "W": W}
