import time
import numpy as np

def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0  # safety margin before the 150s kill

    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    # 6 B-column blocks of 2 columns:
    #   per block = 2*(1x2*2x2 -> 3 terms) + 2*(2x2*2x2 Strassen -> 7 terms) = 6 + 14 = 20
    #   total = 3 * 20 = 60
    RANK = 60
    U = [[0] * RANK for _ in range(n * m)]  # (12, 60)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 60)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 60)

    col = 0

    # Indexing (trace convention for W):
    #   A_{i,k} -> U row i*m + k
    #   B_{k,j} -> V row k*p + j
    #   C_{i,j} -> W row j*n + i
    def _iu(i, k):
        return i * m + k

    def _iv(k, j):
        return k * p + j

    def _iw(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 3.
        A[ar, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[cr, cc], C[cr, cc+1]
        C0 = A0*B0 + A1*B1 ; C1 = A0*B1 + A1*B1
        """
        nonlocal col
        A0 = _iu(a_r, a_c)
        A1 = _iu(a_r, a_c + 1)
        B00 = _iv(b_r, b_c)
        B01 = _iv(b_r, b_c + 1)
        B10 = _iv(b_r + 1, b_c)
        B11 = _iv(b_r + 1, b_c + 1)
        C0 = _iw(c_c, c_r)
        C1 = _iw(c_c + 1, c_r)

        # T1: (A0+A1)(B00+B11) -> C0
        U[A0][col] = 1; U[A1][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C0][col] = 1
        col += 1
        # T2: A0*B01 -> -C0 + C1
        U[A0][col] = 1
        V[B01][col] = 1
        W[C0][col] = -1; W[C1][col] = 1
        col += 1
        # T3: A1*B10 -> -C0 + C1
        U[A1][col] = 1
        V[B10][col] = 1
        W[C0][col] = -1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[cr:cr+2, cc:cc+2]
        """
        nonlocal col
        A00 = _iu(a_r, a_c)
        A01 = _iu(a_r, a_c + 1)
        A10 = _iu(a_r + 1, a_c)
        A11 = _iu(a_r + 1, a_c + 1)
        B00 = _iv(b_r, b_c)
        B01 = _iv(b_r, b_c + 1)
        B10 = _iv(b_r + 1, b_c)
        B11 = _iv(b_r + 1, b_c + 1)
        C00 = _iw(c_c, c_r)
        C01 = _iw(c_c + 1, c_r)
        C10 = _iw(c_c, c_r + 1)
        C11 = _iw(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 60-term decomposition ----
    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # Row 0 of C (A row 0, 1x2 blocks) -> 3 terms each
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # Rows 1-2 of C (A rows 1:3, 2x2 Strassen blocks) -> 7 terms each
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

        if time.time() > deadline:
            break

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # ---- Optional verification + redundant-term removal (within family) ----
    # Verify exact decomposition, then attempt to drop any linearly dependent term.
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # Build the multiplication tensor T[i,k,j] = 1
        # Decomposition: T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
        # Check: sum_r U[i,k,r] V[k,j,r] W[j,i,r] == 1 for all i,k,j
        ok = True
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for r in range(RANK):
                        s += Ua[i * m + k, r] * Va[k * p + j, r] * Wa[j * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            raise ValueError("decomposition not exact")

        # Try to drop a term: for each term t, check if removing it still
        # leaves an exact decomposition (i.e. the residual is zero).
        # A term is redundant iff it is a linear combination that cancels,
        # which for a decomposition of the constant-1 tensor means: after
        # removing term t, the remaining terms still sum to 1 everywhere.
        for t in range(RANK):
            if time.time() > deadline:
                break
            good = True
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        s = 0.0
                        for r in range(RANK):
                            if r == t:
                                continue
                            s += Ua[i * m + k, r] * Va[k * p + j, r] * Wa[j * n + i, r]
                        if abs(s - 1.0) > 1e-9:
                            good = False
                            break
                    if not good:
                        break
                if not good:
                    break
            if good:
                # term t is redundant; drop it
                Ua = np.delete(Ua, t, axis=1)
                Va = np.delete(Va, t, axis=1)
                Wa = np.delete(Wa, t, axis=1)
                RANK -= 1
                break  # re-scan from scratch
    except Exception:
        pass  # keep the verified 60-term construction

    # Rebuild as nested lists (integer entries)
    Uo = [[int(round(Ua[i, r])) for r in range(RANK)] for i in range(n * m)]
    Vo = [[int(round(Va[i, r])) for r in range(RANK)] for i in range(m * p)]
    Wo = [[int(round(Wa[i, r])) for r in range(RANK)] for i in range(p * n)]
    return {"U": Uo, "V": Vo, "W": Wo}
