import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 135.0


def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def build_decomposition_54():
    """
    Build a rank-54 decomposition using Strassen for 2x2x2 blocks.
    Layout: m=5 as [2,2,1], p=6 as [2,2,2]
    """
    n, m, p = 2, 5, 6
    U = np.zeros((n*m, 54))
    V = np.zeros((m*p, 54))
    W = np.zeros((p*n, 54))

    def strassen_block(j0, k0, base_col):
        """
        Add a Strassen block for rows i in [0,1], cols j in [j0, j0+2), k in [k0, k0+2).
        """
        # Term 0: M1 = (A00+A11)(B00+B11)
        U[0, base_col] += 1; U[3, base_col] += 1
        V[j0*6+k0, base_col] += 1; V[(j0+1)*6+(k0+1), base_col] += 1
        W[k0*2+0, base_col] += 1; W[(k0+1)*2+1, base_col] += 1

        # Term 1: M2 = (A10+A11)B00
        U[2, base_col+1] += 1; U[3, base_col+1] += 1
        V[j0*6+k0, base_col+1] += 1
        W[(k0+1)*2+0, base_col+1] += 1

        # Term 2: M3 = A00(B01-B11)
        U[0, base_col+2] += 1
        V[j0*6+(k0+1), base_col+2] += 1; V[(j0+1)*6+(k0+1), base_col+2] -= 1
        W[k0*2+1, base_col+2] += 1

        # Term 3: M4 = A11(B10-B00)
        U[3, base_col+3] += 1
        V[(j0+1)*6+k0, base_col+3] += 1; V[j0*6+k0, base_col+3] -= 1
        W[k0*2+0, base_col+3] += 1; W[(k0+1)*2+0, base_col+3] += 1

        # Term 4: M5 = (A00+A10)B11
        U[0, base_col+4] += 1; U[2, base_col+4] += 1
        V[(j0+1)*6+(k0+1), base_col+4] += 1
        W[k0*2+0, base_col+4] -= 1; W[k0*2+1, base_col+4] += 1

        # Term 5: M6 = (A10-A00)(B00+B01)
        U[2, base_col+5] += 1; U[0, base_col+5] -= 1
        V[j0*6+k0, base_col+5] += 1; V[j0*6+(k0+1), base_col+5] += 1
        W[(k0+1)*2+1, base_col+5] += 1

        # Term 6: M7 = (A01-A11)(B10+B11)
        U[1, base_col+6] += 1; U[3, base_col+6] -= 1
        V[(j0+1)*6+k0, base_col+6] += 1; V[(j0+1)*6+(k0+1), base_col+6] += 1
        W[k0*2+0, base_col+6] += 1

    def schoolblock_2x1x2(j0, k0, base_col):
        """
        Add a schoolbook block for rows i in [0,1], cols j=j0, k in [k0, k0+2).
        """
        # c=0: i=0, j=j0, k=k0
        U[0, base_col] += 1
        V[j0*6+k0, base_col] += 1
        W[k0*2+0, base_col] += 1

        # c=1: i=0, j=j0, k=k0+1
        U[0, base_col+1] += 1
        V[j0*6+(k0+1), base_col+1] += 1
        W[(k0+1)*2+0, base_col+1] += 1

        # c=2: i=1, j=j0, k=k0
        U[1, base_col+2] += 1
        V[j0*6+k0, base_col+2] += 1
        W[k0*2+1, base_col+2] += 1

        # c=3: i=1, j=j0, k=k0+1
        U[1, base_col+3] += 1
        V[j0*6+(k0+1), base_col+3] += 1
        W[(k0+1)*2+1, base_col+3] += 1

    # Strassen blocks for m=0,1 and m=2,3 (p=0,1 and p=2,3 and p=4,5)
    strassen_block(0, 0, 0)
    strassen_block(0, 2, 7)
    strassen_block(0, 4, 14)
    strassen_block(2, 0, 21)
    strassen_block(2, 2, 28)
    strassen_block(2, 4, 35)

    # Schoolbook blocks for m=4 (p=0,1 and p=2,3 and p=4,5)
    schoolblock_2x1x2(4, 0, 42)
    schoolblock_2x1x2(4, 2, 46)
    schoolblock_2x1x2(4, 4, 50)

    U = U[:, :54]
    V = V[:, :54]
    W = W[:, :54]

    if not check_decomposition(U, V, W, n, m, p):
        raise ValueError("Block decomposition failed verification")

    return U, V, W


def try_column_reduction(U, V, W, n, m, p, max_rounds=100):
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_fast(Uc, Vc, Wc):
        result = np.einsum('ac,bc,dc->abd', Uc, Vc, Wc)
        return np.allclose(result, target, atol=1e-9)

    rounds = 0
    while rounds < max_rounds:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        rounds += 1
        removed = False
        for c in range(rank):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            if check_fast(U_test, V_test, W_test):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                removed = True
                break
        if not removed:
            break

    return best_U, best_V, best_W, best_rank


def try_linear_combination_removal(U, V, W, n, m, p):
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_fast(Uc, Vc, Wc):
        result = np.einsum('ac,bc,dc->abd', Uc, Vc, Wc)
        return np.allclose(result, target, atol=1e-9)

    rounds = 0
    while rounds < 50:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        rounds += 1
        removed = False
        for c in range(rank):
            if time.time() - START_TIME > TIME_LIMIT:
                break

            other_cols = [d for d in range(rank) if d != c]
            if len(other_cols) < 3:
                continue

            U_others = U[:, other_cols]
            V_others = V[:, other_cols]
            W_others = W[:, other_cols]

            U_c = U[:, c]
            V_c = V[:, c]
            W_c = W[:, c]

            try:
                alpha_U, _, _, _ = np.linalg.lstsq(U_others, U_c, rcond=None)
                residual_U = np.linalg.norm(U_c - U_others @ alpha_U)
            except:
                continue

            if residual_U > 1e-8:
                continue

            try:
                V_pred = V_others @ alpha_U
                residual_V = np.linalg.norm(V_c - V_pred)
            except:
                continue

            if residual_V > 1e-8:
                continue

            try:
                W_pred = W_others @ alpha_U
                residual_W = np.linalg.norm(W_c - W_pred)
            except:
                continue

            if residual_W > 1e-8:
                continue

            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            rank -= 1
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            removed = True
            break
        if not removed:
            break

    return best_U, best_V, best_W, best_rank


def construct():
    global START_TIME
    n, m, p = DIMENSIONS

    U, V, W = build_decomposition_54()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if time.time() - START_TIME < TIME_LIMIT - 30:
        U2, V2, W2, rank2 = try_column_reduction(best_U, best_V, best_W, n, m, p)
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2

    if time.time() - START_TIME < TIME_LIMIT - 20:
        U3, V3, W3, rank3 = try_linear_combination_removal(best_U, best_V, best_W, n, m, p)
        if rank3 < best_rank:
            best_rank = rank3
            best_U, best_V, best_W = U3, V3, W3

    if not check_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = build_decomposition_54()
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
