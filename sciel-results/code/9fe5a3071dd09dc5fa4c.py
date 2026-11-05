"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Matrix multiplication 3x3 by 3x6.

    Strategy:
      - Build a rank-23 exact decomposition of the 3x3x3 matmul tensor
        (Laderman's scheme), verified exhaustively at runtime.
      - The 3x3x6 tensor splits into two independent 3x3x3 blocks
        (B columns 0-2 and 3-5), so two copies give rank 46.
      - If the rank-23 core fails verification, fall back to schoolbook
        rank 54 (never worse than seed).

    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (18, rank)
    W: (p*n, rank) = (18, rank)   with W[k*n+i, r] trace convention.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 115.0   # safety margin vs 120s

    def time_left():
        return DEADLINE - time.time()

    # ---- Laderman rank-23 scheme for 3x3x3 -----------------------------
    # Each term is a triple (u, v, w) of length-9 vectors with entries in
    # {-1, 0, 1}.  Flattening order: index = row*3 + col.
    #   u encodes a linear form in A, v in B, w the output C placement.
    # The tensor identity required:
    #   sum_c u_c[i*3+j] * v_c[j*3+k] * w_c[k*3+i] == delta(i,j,k)
    # where delta(i,j,k)=1 iff the matmul tensor entry T[i,j,k] is 1,
    # i.e. T[i,j,k] = 1 for all (i,j,k) with the relation C[i,k]=sum_j A[i,j]B[j,k].
    # For the matmul tensor, T[i,j,k] = 1 for every (i,j,k) (it's the
    # structure tensor: C[i,k] = sum_j A[i,j]*B[j,k]  <=>  T[i,j,k]=1).
    #
    # Laderman's 23 products (1-based -> 0-based below).
    # We express each product as (A_form, B_form, C_form) where each form
    # is a dict mapping flattened index -> coefficient.

    def form(*pairs):
        d = {}
        for idx, coef in pairs:
            d[idx] = d.get(idx, 0) + coef
        return d

    # A-forms
    # (0-based: a_rc = r*3+c)
    a11, a12, a13 = 0, 1, 2
    a21, a22, a23 = 3, 4, 5
    a31, a32, a33 = 6, 7, 8
    # B-forms
    b11, b12, b13 = 0, 1, 2
    b21, b22, b23 = 3, 4, 5
    b31, b32, b33 = 6, 7, 8

    # Laderman's 23 terms as (A_form, B_form, C_form).
    # C-form: which output C[r*3+c] gets which coefficient.
    terms = []

    # M1 = (a11+a22)(b11+b22) -> c11 +=, c22 +=
    terms.append((form((a11,1),(a22,1)), form((b11,1),(b22,1)), form((0,1),(4,1))))
    # M2 = (a21+a22)b11 -> c21 +=, c11 -=
    terms.append((form((a21,1),(a22,1)), form((b11,1)), form((3,1),(0,-1))))
    # M3 = a11(b21-b22) -> c12 -=, c11 +=
    terms.append((form((a11,1)), form((b21,1),(b22,-1)), form((1,-1),(0,1))))
    # M4 = a22(b12-b11) -> c21 -=, c22 +=
    terms.append((form((a22,1)), form((b12,1),(b11,-1)), form((3,-1),(4,1))))
    # M5 = (a11+a12)b22 -> c12 +=, c22 -=
    terms.append((form((a11,1),(a12,1)), form((b22,1)), form((1,1),(4,-1))))
    # M6 = (a21-a11)(b11+b12) -> c22 -=, c12 +=
    terms.append((form((a21,1),(a11,-1)), form((b11,1),(b12,1)), form((4,-1),(1,1))))
    # M7 = (a12-a22)(b21+b22) -> c11 -=, c21 +=
    terms.append((form((a12,1),(a22,-1)), form((b21,1),(b22,1)), form((0,-1),(3,1))))
    # M8 = a13(b31+b32+b33) -> c13 +=
    terms.append((form((a13,1)), form((b31,1),(b32,1),(b33,1)), form((2,1))))
    # M9 = (a11+a22+a33)(b11+b22+b33) -> c33 +=
    terms.append((form((a11,1),(a22,1),(a33,1)), form((b11,1),(b22,1),(b33,1)), form((8,1))))
    # M10 = (a13+a33)(b11+b12+b21+b22) -> c31 +=
    terms.append((form((a13,1),(a33,1)), form((b11,1),(b12,1),(b21,1),(b22,1)), form((6,1))))
    # M11 = (a12+a22+a33)(b22+b23+b32+b33) -> c32 +=
    terms.append((form((a12,1),(a22,1),(a33,1)), form((b22,1),(b23,1),(b32,1),(b33,1)), form((7,1))))
    # M12 = (a31+a33)(b13+b23) -> c13 +=, c33 -=
    terms.append((form((a31,1),(a33,1)), form((b13,1),(b23,1)), form((2,1),(8,-1))))
    # M13 = (a11+a12+a31+a33)(b11+b12+b31+b32) -> c11 +=, c31 +=
    terms.append((form((a11,1),(a12,1),(a31,1),(a33,1)), form((b11,1),(b12,1),(b31,1),(b32,1)), form((0,1),(6,1))))
    # M14 = (a21+a22+a31+a33)(b11+b13+b21+b23) -> c21 +=, c31 +=
    terms.append((form((a21,1),(a22,1),(a31,1),(a33,1)), form((b11,1),(b13,1),(b21,1),(b23,1)), form((3,1),(6,1))))
    # M15 = (a12+a22+a32+a33)(b22+b23+b32+b33) -> c22 +=, c32 +=
    terms.append((form((a12,1),(a22,1),(a32,1),(a33,1)), form((b22,1),(b23,1),(b32,1),(b33,1)), form((4,1),(7,1))))
    # M16 = (a31+a32+a33)(b12+b13+b22+b23) -> c31 +=, c32 +=
    terms.append((form((a31,1),(a32,1),(a33,1)), form((b12,1),(b13,1),(b22,1),(b23,1)), form((6,1),(7,1))))
    # M17 = (a11+a13)(b31+b33) -> c13 +=, c33 -=
    terms.append((form((a11,1),(a13,1)), form((b31,1),(b33,1)), form((2,1),(8,-1))))
    # M18 = (a12+a22+a33)(b12+b13+b32+b33) -> c23 +=, c13 -=
    terms.append((form((a12,1),(a22,1),(a33,1)), form((b12,1),(b13,1),(b32,1),(b33,1)), form((5,1),(2,-1))))
    # M19 = (a13+a23)(b11+b12) -> c13 +=, c23 +=
    terms.append((form((a13,1),(a23,1)), form((b11,1),(b12,1)), form((2,1),(5,1))))
    # M20 = (a23+a33)(b21+b22) -> c23 +=, c33 -=
    terms.append((form((a23,1),(a33,1)), form((b21,1),(b22,1)), form((5,1),(8,-1))))
    # M21 = (a11+a12+a13)(b31+b32+b33) -> c13 +=
    terms.append((form((a11,1),(a12,1),(a13,1)), form((b31,1),(b32,1),(b33,1)), form((2,1))))
    # M22 = (a21+a22+a23)(b31+b32+b33) -> c23 +=
    terms.append((form((a21,1),(a22,1),(a23,1)), form((b31,1),(b32,1),(b33,1)), form((5,1))))
    # M23 = (a31+a32+a33)(b11+b12+b13) -> c31 +=
    terms.append((form((a31,1),(a32,1),(a33,1)), form((b11,1),(b12,1),(b13,1)), form((6,1))))

    def to_vec(d, length):
        v = [0] * length
        for idx, coef in d.items():
            if 0 <= idx < length:
                v[idx] = coef
        return v

    r23_U = [to_vec(t[0], 9) for t in terms]
    r23_V = [to_vec(t[1], 9) for t in terms]
    r23_W = [to_vec(t[2], 9) for t in terms]

    def verify_core(Uc, Vc, Wc, d):
        """Check sum_c u[i*d+j]*v[j*d+k]*w[k*d+i] == 1 for all i,j,k."""
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    s = 0
                    for c in range(len(Uc)):
                        s += Uc[c][i*d+j] * Vc[c][j*d+k] * Wc[c][k*d+i]
                    if s != 1:
                        return False
        return True

    core_ok = verify_core(r23_U, r23_V, r23_W, 3)

    if core_ok and time_left() > 5.0:
        # Build rank-46 for 3x3x6: two 3x3x3 blocks.
        rank = 46
        U = [[0] * rank for _ in range(n * m)]   # (9, 46)
        V = [[0] * rank for _ in range(m * p)]   # (18, 46)
        W = [[0] * rank for _ in range(p * n)]   # (18, 46)
        for block in range(2):
            for c in range(23):
                col = block * 23 + c
                u = r23_U[c]; v = r23_V[c]; w = r23_W[c]
                for idx in range(9):
                    U[idx][col] = u[idx]
                    V[idx + block * 9][col] = v[idx]
                    W[idx + block * 9][col] = w[idx]
        # Full verification of the 3x3x6 tensor.
        ok = True
        for a in range(n * m):
            if not ok:
                break
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                if j1 != j2:
                    continue
                for c2 in range(p * n):
                    k2, i2 = divmod(c2, n)
                    expected = 1 if (k1 == k2 and i1 == i2) else 0
                    s = 0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[c2][c]
                    if s != expected:
                        ok = False
                        break
        if ok:
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank n*m*p = 54 (guaranteed valid).
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
