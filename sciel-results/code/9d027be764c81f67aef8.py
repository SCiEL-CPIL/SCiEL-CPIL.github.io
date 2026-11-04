"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24

    # Build the tensor T: T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i
    tensor_ones = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_ones.add((a, b, c))

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Precompute rhs vectors
    rhs_vectors = []
    for c in range(pn):
        rhs = np.zeros(nm * mp, dtype=float)
        for a in range(nm):
            for b in range(mp):
                if (a, b, c) in tensor_ones:
                    rhs[a * mp + b] = 1.0
        rhs_vectors.append(rhs)

    def verify_solution(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using integer arithmetic."""
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0
                    for t in range(rank):
                        s += U_mat[a][t] * V_mat[b][t] * W_mat[c][t]
                    expected = 1 if (a, b, c) in tensor_ones else 0
                    if s != expected:
                        return False
        return True

    # Try structured sparse patterns for lower rank
    # Strategy: group (i,j,k) triples by shared indices to create sparse U,V
    for rank in range(90, best_rank):
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue

        for attempt in range(80):
            if time.time() > deadline:
                break

            U = np.zeros((nm, rank), dtype=float)
            V = np.zeros((mp, rank), dtype=float)

            # Create structured assignment: group triples by (i,j) or (j,k) or (i,k)
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(triples)

            # Assign triples to columns with some structure
            col_to_triples = {t: [] for t in range(rank)}
            for (i, j, k) in triples:
                # Try to find a column where (i,j) or (j,k) already appears
                best_col = None
                for t in range(rank):
                    for (i2, j2, k2) in col_to_triples[t]:
                        if (i == i2 and j == j2) or (j == j2 and k == k2):
                            best_col = t
                            break
                    if best_col is not None:
                        break
                if best_col is None:
                    best_col = min(range(rank), key=lambda t: len(col_to_triples[t]))
                col_to_triples[best_col].append((i, j, k))

            # Build U and V from assignment
            for t in range(rank):
                for (i, j, k) in col_to_triples[t]:
                    a = i * m + j
                    b = j * p + k
                    U[a, t] = 1.0
                    V[b, t] = 1.0

            # Build M matrix: M[a*mp+b, t] = U[a,t]*V[b,t]
            M = np.zeros((nm * mp, rank), dtype=float)
            for a in range(nm):
                for b in range(mp):
                    M[a * mp + b, :] = U[a, :] * V[b, :]

            # Solve for W using least squares
            W = np.zeros((pn, rank), dtype=float)
            valid = True
            for c in range(pn):
                try:
                    W_c, res, _, _ = np.linalg.lstsq(M, rhs_vectors[c], rcond=None)
                    if len(res) > 0 and res[0] > 1e-8:
                        valid = False
                        break
                    W[c, :] = W_c
                except:
                    valid = False
                    break

            if valid:
                # Round to integers and verify exactly
                U_int = [[int(round(U[a, t])) for t in range(rank)] for a in range(nm)]
                V_int = [[int(round(V[b, t])) for t in range(rank)] for b in range(mp)]
                W_int = [[int(round(W[c, t])) for t in range(rank)] for c in range(pn)]

                if verify_solution(U_int, V_int, W_int, rank):
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
                    best_rank = rank
                    break

        if best_rank <= rank:
            break

    return best_result
# EVOLVE-BLOCK-END
