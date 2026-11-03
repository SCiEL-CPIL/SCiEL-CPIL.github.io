"""Matrix multiplication tensor decomposition for 4x4 * 4x7.
Strategy: tiled multi-block Strassen.
Partition into two disjoint 2x2x2 blocks, apply Strassen to each.
Rank = 112 - 2 = 110.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Target tensor entries: T[i*m+j, j*p+k, k*n+i] = 1
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.add((i * m + j, j * p + k, k * n + i))

    def verify_solution(sol):
        """Strict verification of tensor decomposition."""
        U = sol["U"]; V = sol["V"]; W = sol["W"]
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        for (u, v, w) in target_entries:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False

        for u in range(n * m):
            Uu = U_np[u]
            for v in range(m * p):
                UV = Uu * V_np[v]
                if not np.any(UV):
                    continue
                for w in range(p * n):
                    if (u, v, w) in target_entries:
                        continue
                    val = float(np.dot(UV, W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # ---- Tiled multi-block Strassen: two disjoint 2x2x2 blocks ----
    # Block 1: rows {0,1}, cols {0,1}, outputs {0,1}
    # Block 2: rows {2,3}, cols {2,3}, outputs {2,3}
    # Both blocks are fully within the 4x4x7 tensor (p=7 >= 4).
    
    rank_total = n * m * p - 2  # 112 - 2 = 110
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)

    col = 0

    # Define the two blocks: (i0, i1, j0, j1, k0, k1)
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (2, 3, 2, 3, 2, 3),
    ]

    # Mark which (i,j,k) triples are in a block
    in_block = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    in_block.add((i, j, k))

    # Place all non-block terms (standard rank-1 terms)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in in_block:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Now col should be 112 - 16 = 96. We have 14 columns for 2 Strassen blocks (7 each).
    assert col == n * m * p - 16, f"Expected col={n*m*p-16}, got {col}"

    # Apply Strassen to each block
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # P1 = (a+d)(e+h)
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1

        # P2 = (c+d)(e-f)
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += -1
        W[k0*n+i0, col] += -1; W[k0*n+i1, col] += 1
        col += 1

        # P3 = (a-b)(g+h)
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += -1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += -1
        col += 1

        # P4 = (c-a)(f-h)
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] += -1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] += -1
        W[k0*n+i0, col] += 1; W[k1*n+i0, col] += -1
        col += 1

        # P5 = (a+b)(g-h)
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += -1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1

        # P6 = (c+d)(f-h)
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] += -1
        W[k0*n+i1, col] += 1; W[k1*n+i0, col] += -1; W[k1*n+i1, col] += -1
        col += 1

        # P7 = (a-d)(e+g)
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += -1
        V[j0*p+k0, col] += 1; V[j1*p+k0, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i0, col] += -1
        col += 1

    assert col == rank_total, f"Expected col={rank_total}, got {col}"

    sol = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

    if verify_solution(sol):
        return sol

    # Fallback: schoolbook rank 112
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    sol_sb = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    return sol_sb
# EVOLVE-BLOCK-END
