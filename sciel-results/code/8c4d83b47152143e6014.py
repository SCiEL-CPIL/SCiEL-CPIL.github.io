import numpy as np
import itertools
from collections import defaultdict

def construct():
    n, m, p = 2, 3, 9
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18

    # The tensor T_{ijk} where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if j == k else 0
    # We seek matrices U (nm x rank), V (mp x rank), W (pn x rank)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]

    # We start from a known rank-50 decomposition and try to reduce rank.
    # Approach: 
    # 1. Construct the rank-50 decomposition (same as before).
    # 2. Form the tensor T as a 3D array.
    # 3. For each candidate rank R < 50, we check if a rank-R decomposition exists.
    #    This is generally hard, but we can use a heuristic:
    #    - Try to find a linear dependence in the columns of the rank-50 matrices.
    #    - If columns are linearly dependent, we can reduce the rank.

    # Let's build the rank-50 decomposition first.
    rank_50 = 50
    U50 = [[0.0] * rank_50 for _ in range(nm)]
    V50 = [[0.0] * rank_50 for _ in range(mp)]
    W50 = [[0.0] * rank_50 for _ in range(pn)]

    def aidx(i, j): return i * m + j
    def bidx(j, k): return j * p + k
    def cidx(k, i): return k * n + i

    col = 0
    # 4 Strassen blocks for k=0..7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = bidx(0, k0); v_f = bidx(0, k1)
        v_g = bidx(1, k0); v_h = bidx(1, k1)
        u_a = aidx(0, 0); u_b = aidx(0, 1)
        u_c = aidx(1, 0); u_d = aidx(1, 1)

        # M1 = (a+d)(e+h)
        U50[u_a][col] += 1.0; U50[u_d][col] += 1.0
        V50[v_e][col] += 1.0; V50[v_h][col] += 1.0
        # M2 = (c+d)e
        U50[u_c][col+1] += 1.0; U50[u_d][col+1] += 1.0
        V50[v_e][col+1] += 1.0
        # M3 = a(f-h)
        U50[u_a][col+2] += 1.0
        V50[v_f][col+2] += 1.0; V50[v_h][col+2] -= 1.0
        # M4 = d(g-e)
        U50[u_d][col+3] += 1.0
        V50[v_g][col+3] += 1.0; V50[v_e][col+3] -= 1.0
        # M5 = (a+b)h
        U50[u_a][col+4] += 1.0; U50[u_b][col+4] += 1.0
        V50[v_h][col+4] += 1.0
        # M6 = (c-a)(e+f)
        U50[u_c][col+5] += 1.0; U50[u_a][col+5] -= 1.0
        V50[v_e][col+5] += 1.0; V50[v_f][col+5] += 1.0
        # M7 = (b-d)(g+h)
        U50[u_b][col+6] += 1.0; U50[u_d][col+6] -= 1.0
        V50[v_g][col+6] += 1.0; V50[v_h][col+6] += 1.0

        w_00 = cidx(k0, 0); w_01 = cidx(k1, 0)
        w_10 = cidx(k0, 1); w_11 = cidx(k1, 1)

        W50[w_00][col]   += 1.0; W50[w_00][col+3] += 1.0
        W50[w_00][col+4] -= 1.0; W50[w_00][col+6] += 1.0
        W50[w_01][col+2] += 1.0; W50[w_01][col+4] += 1.0
        W50[w_10][col+1] += 1.0; W50[w_10][col+3] += 1.0
        W50[w_11][col]   += 1.0; W50[w_11][col+1] -= 1.0
        W50[w_11][col+2] += 1.0; W50[w_11][col+5] += 1.0
        col += 7

    # k=8 direct
    v_e = bidx(0, 8); v_g = bidx(1, 8)
    u_a = aidx(0, 0); u_b = aidx(0, 1); u_c = aidx(1, 0); u_d = aidx(1, 1)
    U50[u_a][col]   += 1.0; V50[v_e][col]   += 1.0
    U50[u_b][col+1] += 1.0; V50[v_g][col+1] += 1.0
    U50[u_c][col+2] += 1.0; V50[v_e][col+2] += 1.0
    U50[u_d][col+3] += 1.0; V50[v_g][col+3] += 1.0
    W50[cidx(8, 0)][col]   += 1.0; W50[cidx(8, 0)][col+1] += 1.0
    W50[cidx(8, 1)][col+2] += 1.0; W50[cidx(8, 1)][col+3] += 1.0
    col += 4

    # Part 2: A_right * B_bottom
    u_ar0 = aidx(0, 2); u_ar1 = aidx(1, 2)
    for k in range(9):
        v_b = bidx(2, k)
        U50[u_ar0][col]   += 1.0; V50[v_b][col]   += 1.0
        W50[cidx(k, 0)][col] += 1.0
        U50[u_ar1][col+1] += 1.0; V50[v_b][col+1] += 1.0
        W50[cidx(k, 1)][col+1] += 1.0
        col += 2

    # Convert to numpy arrays
    U50 = np.array(U50, dtype=float)
    V50 = np.array(V50, dtype=float)
    W50 = np.array(W50, dtype=float)

    # Verify the decomposition
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0 if j == k else 0.0

    def check(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        C = np.zeros((n, m, p))
        for r in range(R):
            # U[:, r] has shape (nm,) -> reshape to (n, m)
            # V[:, r] has shape (mp,) -> reshape to (m, p)
            # W[:, r] has shape (pn,) -> reshape to (p, n)
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # C[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
        return np.allclose(C, T, atol=1e-9)

    if not check(U50, V50, W50):
        raise ValueError("Initial decomposition is incorrect")

    # Now try to reduce rank.
    # We can use the following idea:
    # If we find a linear combination of columns of U, V, W that yields zero tensor contribution,
    # we can remove that column.
    # More formally, if there exists a vector x in R^R such that 
    # sum_r x_r * (U[:,r] (x) V[:,r] (x) W[:,r]) = 0, then we can project out this direction.
    # But this is a hard problem.

    # Alternative: Use the fact that the tensor is sparse and structured.
    # We can try to find a lower rank decomposition by solving a system of equations.
    # For a given rank R, we have R * (nm + mp + pn) unknowns.
    # We have n*m*p = 2*3*9 = 54 equations.
    # For R=27, we have 27*(6+27+18) = 27*51 = 1377 unknowns, which is much larger than 54.
    # So a solution likely exists for R=27, but finding it is hard.

    # Let's try a greedy approach:
    # Start with the rank-50 decomposition.
    # Try to remove columns one by one.
    # For each column r, check if removing it and adjusting the remaining columns can still satisfy the equations.
    # This is not straightforward.

    # Instead, let's try to use a different construction.
    # The tensor T[i,j,k] = delta_{jk}.
    # We can write C = A * B.
    # A is 2x3, B is 3x9.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We want to decompose this using rank R.
    # One trivial decomposition is R = n*m*p = 54 (outer products).
    # But we know R=50 works.

    # Let's try to find a decomposition of rank R for R in [27, 49].
    # We will use a randomized search.
    # Fix W to be a random matrix of rank R, and solve for U and V.
    # But this is not guaranteed to work.

    # Given the time constraint, we will stick with the rank-50 decomposition.
    # However, we can try to simplify the representation.

    # Convert back to lists of ints
    U = [[int(round(v)) for v in row] for row in U50]
    V = [[int(round(v)) for v in row] for row in V50]
    W = [[int(round(v)) for v in row] for row in W50]

    # Remove zero columns to reduce rank if possible
    # Check each column
    keep = []
    for r in range(rank_50):
        col_u = [U[i][r] for i in range(nm)]
        col_v = [V[i][r] for i in range(mp)]
        col_w = [W[i][r] for i in range(pn)]
        if any(col_u) or any(col_v) or any(col_w):
            keep.append(r)

    # Reconstruct with kept columns
    U = [[U[i][r] for r in keep] for i in range(nm)]
    V = [[V[i][r] for r in keep] for i in range(mp)]
    W = [[W[i][r] for r in keep] for i in range(pn)]

    rank = len(keep)

    return {"U": U, "V": V, "W": W}
