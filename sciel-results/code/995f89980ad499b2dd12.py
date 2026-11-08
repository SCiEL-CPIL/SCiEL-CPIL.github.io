import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    DEADLINE = t_start + 140.0

    # ---- Build the target matrix-multiplication tensor ----
    # i = row_A*m + col_A (0..5); j = row_B*p + col_B (0..11); k = col_C*n + row_C (0..7)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # ---- Rank-22 base decomposition (Strassen x2 + outer product) ----
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx, col] += val
        for idx, val in v_indices:
            V[idx, col] += val
        for idx, val in w_indices:
            W[idx, col] += val

    # Block 1: A[0:2,0:2] * B[0:2,0:2]  (Strassen)
    add_term([(0,1),(4,1)], [(0,1),(5,1)], [(0,1),(3,1)], 0)
    add_term([(3,1),(4,1)], [(0,1)],        [(1,1),(3,-1)], 1)
    add_term([(0,1)],        [(1,1),(5,-1)], [(2,1),(3,1)], 2)
    add_term([(4,1)],        [(4,1),(0,-1)], [(0,1),(1,1)], 3)
    add_term([(0,1),(1,1)],  [(5,1)],        [(0,-1),(2,1)], 4)
    add_term([(3,1),(0,-1)], [(0,1),(1,1)],  [(3,1)], 5)
    add_term([(1,1),(4,-1)], [(4,1),(5,1)],  [(0,1)], 6)
    # Block 2: A[0:2,0:2] * B[0:2,2:4]  (Strassen)
    add_term([(0,1),(4,1)], [(2,1),(7,1)], [(4,1),(7,1)], 7)
    add_term([(3,1),(4,1)], [(2,1)],        [(5,1),(7,-1)], 8)
    add_term([(0,1)],        [(3,1),(7,-1)], [(6,1),(7,1)], 9)
    add_term([(4,1)],        [(6,1),(2,-1)], [(4,1),(5,1)], 10)
    add_term([(0,1),(1,1)],  [(7,1)],        [(4,-1),(6,1)], 11)
    add_term([(3,1),(0,-1)], [(2,1),(3,1)],  [(7,1)], 12)
    add_term([(1,1),(4,-1)], [(6,1),(7,1)],  [(4,1)], 13)
    # Block 3: A[0:2,2:3] * B[2:3,0:4]  (outer product)
    add_term([(2,1)], [(8,1)],  [(0,1)], 14)
    add_term([(2,1)], [(9,1)],  [(2,1)], 15)
    add_term([(2,1)], [(10,1)], [(4,1)], 16)
    add_term([(2,1)], [(11,1)], [(6,1)], 17)
    add_term([(5,1)], [(8,1)],  [(1,1)], 18)
    add_term([(5,1)], [(9,1)],  [(3,1)], 19)
    add_term([(5,1)], [(10,1)], [(5,1)], 20)
    add_term([(5,1)], [(11,1)], [(7,1)], 21)

    # ---- Fast sparse tensor reconstruction from a subset of columns ----
    nnz = 6 * 12 * 8  # full size
    Tflat = T.reshape(nnz)

    def term_indices(col):
        ui = np.nonzero(U[:, col])[0]
        vi = np.nonzero(V[:, col])[0]
        wi = np.nonzero(W[:, col])[0]
        out = []
        for a in ui:
            for b in vi:
                for c in wi:
                    out.append((a, b, c, U[a, col] * V[b, col] * W[c, col]))
        return out

    def reconstruct(cols):
        acc = np.zeros(nnz)
        for col in cols:
            for a, b, c, val in term_indices(col):
                acc[a * (m * p) * (p * n) + b * (p * n) + c] += val
        return acc

    def is_valid(cols, tol=1e-9):
        return np.linalg.norm(reconstruct(cols) - Tflat) <= tol

    # ---- Redundancy test: is rank-1 tensor of `col` in span of the others? ----
    def redundant(col, tol=1e-9):
        others = [c for c in range(rank) if c != col]
        basis = []
        for c in others:
            vec = reconstruct([c])
            piv = None
            for i in range(nnz):
                if abs(vec[i]) > tol:
                    piv = i
                    break
            if piv is None:
                continue
            for b in basis:
                if abs(vec[b[0]]) > tol:
                    f = vec[b[0]] / b[1][b[0]]
                    vec = vec - f * b[1]
            if np.max(np.abs(vec)) > tol:
                basis.append((piv, vec.copy()))
        target = reconstruct([col])
        for b in basis:
            if abs(target[b[0]]) > tol:
                f = target[b[0]] / b[1][b[0]]
                target = target - f * b[1]
        return np.max(np.abs(target)) <= tol

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # verify base
    if not is_valid(list(range(rank))):
        print("WARNING: base decomposition invalid")
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # ---- Search 1: single-column redundancy ----
    for col in range(rank):
        if time.time() > DEADLINE:
            break
        if redundant(col):
            cols = [c for c in range(rank) if c != col]
            if is_valid(cols):
                best_rank = rank - 1
                best_U, best_V, best_W = np.delete(U, col, 1), np.delete(V, col, 1), np.delete(W, col, 1)
                break

    # ---- Search 2: pairwise column redundancy (needs a shared dependence) ----
    # If dropping col a and col b together is valid, rank reduces by 2.
    for a in range(rank):
        if time.time() > DEADLINE:
            break
        if best_rank < rank:
            break
        for b in range(a + 1, rank):
            if time.time() > DEADLINE:
                break
            cols = [c for c in range(rank) if c != a and c != b]
            if is_valid(cols):
                best_rank = rank - 2
                best_U = np.delete(np.delete(U, a, 1), b, 1)
                best_V = np.delete(np.delete(V, a, 1), b, 1)
                best_W = np.delete(np.delete(W, a, 1), b, 1)
                break
        if best_rank < rank:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
