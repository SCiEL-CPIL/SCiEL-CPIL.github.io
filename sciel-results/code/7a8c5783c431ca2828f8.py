"""Matrix multiplication tensor decomposition for (2,3,10).

Strategy:
  * Block decomposition exploiting the middle dimension m=3.
  * Split A into A0 (2x2, cols 0-1) and A1 (2x1, col 2).
  * Split B into B0 (2x10, rows 0-1) and B1 (1x10, row 2).
  * C = A0 @ B0 + A1 @ B1.
  * For A0 @ B0: split B0 into 5 blocks of width 2. Each 2x2x2 block
    uses Strassen's rank-7 algorithm. Total: 5 * 7 = 35 terms.
  * For A1 @ B1: outer product of 2x1 and 1x10. Rank 2 * 10 = 20 terms.
  * Total rank: 35 + 20 = 55 (vs 60 schoolbook).
"""
import time
import random


# Strassen rank-7 coefficients for 2x2x2 multiplication.
# For matrices X (2x2), Y (2x2), Z = X @ Y:
# X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
# Strassen's 7 products:
# M1 = (x00+x11)*(y00+y11)
# M2 = (x10+x11)*y00
# M3 = x00*(y01-y11)
# M4 = x11*(y10-y00)
# M5 = (x00+x01)*y11
# M6 = (x10-x00)*(y00+y10)
# M7 = (x01-x11)*(y01+y11)
# Z00 = M1 + M4 - M5 + M7
# Z01 = M3 + M5
# Z10 = M2 + M4
# Z11 = M1 - M2 + M3 + M6

def _strassen_2x2x2():
    """Return (U, V, W) lists for one 2x2x2 Strassen block.
    U: 4 rows (x00,x01,x10,x11), V: 4 rows (y00,y01,y10,y11), W: 4 rows (z00,z01,z10,z11)
    Each is a list of 7 coefficients.
    """
    U = [
        [1, 0, 1, 0, 0, 0, 0],  # x00
        [0, 1, 0, 0, 1, 0, 0],  # x01
        [0, 0, 1, 1, 0, 1, 0],  # x10
        [1, 0, 0, 1, 0, 0, 1],  # x11
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],  # y00
        [0, 0, 1, 0, 1, 0, 1],  # y01
        [0, 1, -1, -1, 0, 1, 0],  # y10
        [1, 0, -1, 0, 1, 0, 1],  # y11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # z00
        [0, 0, 1, 0, 1, 0, 0],  # z01
        [0, 1, 0, 1, 0, 0, 0],  # z10
        [1, 0, 1, 0, 0, 1, 0],  # z11
    ]
    return U, V, W


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = 2, 3, 10

    # Build Strassen block once
    sU, sV, sW = _strassen_2x2x2()

    # Total rank: 5 Strassen blocks (35) + schoolbook for A1@B1 (20) = 55
    total_rank = 5 * 7 + 2 * 10

    U = [[0] * total_rank for _ in range(n * m)]  # 6 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 30 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 20 rows

    # --- Part 1: A0 @ B0 using Strassen on 5 blocks of width 2 ---
    # A0 uses rows 0,1 of A (indices 0,1 in U: i*m+j for i in 0,1; j in 0,1)
    #   U row 0 = A[0,0], U row 1 = A[0,1], U row 2 = A[1,0], U row 3 = A[1,1]
    # B0 uses rows 0,1 of B (indices 0..19 in V: j*p+k for j in 0,1; k in 0..9)
    #   For block b (k=2b, 2b+1): V rows 2b, 2b+1, 2b+p, 2b+p+1
    # C block: W rows for output (i,k) and (i,k+1) -> indices k*n+i and (k+1)*n+i

    for b in range(5):
        base = b * 7
        k0 = 2 * b
        k1 = 2 * b + 1
        # V row indices for this 2x2 block of B:
        #   y00 = B[0,k0] -> V[0*p + k0] = V[k0]
        #   y01 = B[0,k1] -> V[0*p + k1] = V[k1]
        #   y10 = B[1,k0] -> V[1*p + k0] = V[p + k0]
        #   y11 = B[1,k1] -> V[1*p + k1] = V[p + k1]
        v_rows = [k0, k1, p + k0, p + k1]
        # W row indices for this 2x2 block of C:
        #   z00 = C[0,k0] -> W[k0*n + 0] = W[k0*2]
        #   z01 = C[0,k1] -> W[k1*n + 0] = W[k1*2]
        #   z10 = C[1,k0] -> W[k0*n + 1] = W[k0*2 + 1]
        #   z11 = C[1,k1] -> W[k1*n + 1] = W[k1*2 + 1]
        w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
        # U rows for A0 (2x2): x00=A[0,0], x01=A[0,1], x10=A[1,0], x11=A[1,1]
        u_rows = [0, 1, 2, 3]

        for r in range(7):
            for lr in range(4):
                U[u_rows[lr]][base + r] = sU[lr][r]
                V[v_rows[lr]][base + r] = sV[lr][r]
                W[w_rows[lr]][base + r] = sW[lr][r]

    # --- Part 2: A1 @ B1 (schoolbook) ---
    # A1 is 2x1: A[0,2] (U row 5) and A[1,2] (U row 5? no)
    # U row index: i*m+j. A[0,2] -> 0*3+2=2? No, m=3.
    # Wait: n=2, m=3. U has n*m=6 rows.
    #   U[0] = A[0,0], U[1] = A[0,1], U[2] = A[0,2]
    #   U[3] = A[1,0], U[4] = A[1,1], U[5] = A[1,2]
    # A1 = [[A[0,2]], [A[1,2]]] -> U rows 2 and 5
    # B1 is 1x10: B[2,0..9] -> V rows 2*10+0 .. 2*10+9 = V[20..29]
    # C1 = A1 @ B1 is 2x10: C[i,k] = A[i,2]*B[2,k]
    #   W[k*n + i] for i in {0,1}, k in 0..9

    sb_base = 5 * 7  # start of schoolbook block
    for i in range(n):
        for k in range(p):
            r = sb_base + i * p + k
            U[i * m + 2][r] = 1  # A[i,2]
            V[2 * p + k][r] = 1  # B[2,k]
            W[k * n + i][r] = 1

    # --- Verify with random inputs ---
    rng = random.Random(7)
    valid = True
    for trial in range(20):
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C_direct = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                C_direct[i][k] = sum(A[i][j] * B[j][k] for j in range(m))
        C_decomp = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(total_rank):
                    ua = sum(U[i * m + j][r] * A[i][j] for j in range(m))
                    vb = sum(V[j * p + k][r] * B[j][k] for j in range(m))
                    acc += W[k * n + i][r] * ua * vb
                C_decomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(C_direct[i][k] - C_decomp[i][k]) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if time.time() > deadline - 5.0:
            break

    if not valid:
        # Fallback to schoolbook if Strassen verification fails
        total_rank_sb = n * m * p
        U_sb = [[0] * total_rank_sb for _ in range(n * m)]
        V_sb = [[0] * total_rank_sb for _ in range(m * p)]
        W_sb = [[0] * total_rank_sb for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][idx] = 1
                    V_sb[j * p + k][idx] = 1
                    W_sb[k * n + i][idx] = 1
                    idx += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
