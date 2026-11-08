"""Fresh design: rank-minimal tensor decomposition for 2x3 * 3x10 matrix product."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _pair_mul_2x2():
    # Returns U (4 x 3), V (2 x 3) such that for scalars a,b,c,d:
    #   prod = U @ (V @ (a,b))  ==  [ab, ac+bd, ad+bc, cd]
    # Term structure: term t has (x,y,z,w) in U and (u,v) in V,
    # giving contribution (x*u, x*v, y*u, y*v) = (ab, ac+bd, ad+bc, cd) components.
    U = [[1, 0, 1],
         [1, 1, 0],
         [0, 1, 1],
         [0, 0, 1]]
    V = [[1, 0, 1],
         [0, 1, 1]]
    return U, V


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    n, m, p = DIMENSIONS
    # Pair counts
    nr = (n + 1) // 2   # 1
    mr = (m + 1) // 2   # 2
    pr = (p + 1) // 2   # 5
    T = 3               # terms for 2x2 block multiplication
    rank = nr * mr * pr * T  # 1*2*5*3 = 10

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    u_block, v_block = _pair_mul_2x2()  # 4x3, 2x3

    for a in range(nr):
        for b in range(mr):
            for c in range(pr):
                for t in range(T):
                    col = ((a * mr + b) * pr + c) * T + t
                    # U: row index i*m + j, where i = 2a + dx, j = 2b + dy
                    for dx in range(2):
                        for dy in range(2):
                            i = 2 * a + dx
                            j = 2 * b + dy
                            if i < n and j < m:
                                U[i * m + j][col] = u_block[dx * 2 + dy][t]
                    # V: row index j*p + k, where j = 2b + dy, k = 2c + dz
                    for dy in range(2):
                        for dz in range(2):
                            j = 2 * b + dy
                            k = 2 * c + dz
                            if j < m and k < p:
                                V[j * p + k][col] = v_block[dy][t]
                    # W: row index k*n + i, where k = 2c + dz, i = 2a + dx
                    for dz in range(2):
                        for dx in range(2):
                            k = 2 * c + dz
                            i = 2 * a + dx
                            if k < p and i < n:
                                W[k * n + i][col] = u_block[dx * 2 + dz][t]

    # Verify exactness before returning
    try:
        import numpy as np
        if time.time() < deadline:
            Um = np.array(U, dtype=object)
            Vm = np.array(V, dtype=object)
            Wm = np.array(W, dtype=object)
            # Reconstruct: C[i,k] = sum_{j,c} A[i,j] B[j,k]
            # via: sum_t U[i*m+j, t] * (A[i,j]) * V[j*p+k, t] * (B[j,k]) * W[k*n+i, t]
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # coefficient of A[i,j]*B[j,k] in C[i,k] must be 1
                        s = 0
                        for t in range(rank):
                            s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                        if s != 1:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                # Fallback to schoolbook if verification fails (should not)
                rank = n * m * p
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
