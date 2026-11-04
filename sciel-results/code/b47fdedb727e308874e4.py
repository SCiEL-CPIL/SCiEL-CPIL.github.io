"""Matrix multiplication 3x4x8 - low-rank exact bilinear decomposition."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 3, 4, 8

    def verify(U, V, W, trials=30, atol=1e-8):
        """Verify C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for _ in range(trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(U.shape[1]):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                # L_c(A)[i] = sum_j Uc[i,j]*A[i,j]
                Lc = Uc * A          # elementwise, shape (n,m) -> but we need sum over j per i
                Lc = Lc.sum(axis=1)  # shape (n)
                Mc = Vc * B
                Mc = Mc.sum(axis=0)  # shape (p)
                # outer Lc (n) x Mc (p), weighted by Wc (p,n)
                C_computed += Wc.T * np.outer(Lc, Mc)
            if not np.allclose(C_computed, C_expected, atol=atol):
                return False
        return True

    # Schoolbook baseline (rank 96) - guaranteed valid
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Build a Strassen-based decomposition.
    # Strategy: split middle index j into two groups {0,1} and {2,3}.
    # For each group, we have a 3x2 (A-block) times 2x8 (B-block) = 3x8.
    # Further split the 8 output columns into four 2-column blocks.
    # For each (j-group, k-block): a 3x2 * 2x2 = 3x2 sub-problem.
    # The 3x2*2x2 tensor: Z[i,k] = X[i,0]*Y[0,k] + X[i,1]*Y[1,k].
    # This is M(3,2,2). Its rank is 12 (schoolbook) but we can use the
    # "Strassen on the 2x2 part shared across i".
    #
    # Actually M(3,2,2) = sum over i of (e_i * (row i of X)) * (Y) .
    # We can write: for the 2x2 block Y, use Strassen rank 7 to compute
    # X_row * Y for a single row, but X has 3 rows.
    #
    # Cleaner: M(3,2,2) has rank at most 3 * 4 = 12 (schoolbook).
    # But note M(3,2,2) <= M(3,2,2). We can use: rank(M(3,2,2)) <= 3*rank(M(1,2,2)).
    # rank(M(1,2,2)) = 2 (it's just two products x0*y0k + x1*y1k, two terms).
    # So 3*2 = 6? No: M(1,2,2) tensor T[a,k] = X[a]*Y[k]... wait X is 1x2, Y is 2x2.
    # Z[k] = X[0]*Y[0,k] + X[1]*Y[1,k]. That's 2 bilinear terms. rank = 2.
    # So M(3,2,2) <= 3 * 2 = 6? Let's check: each i gives Z[i,k] = X[i,0]*Y[0,k]+X[i,1]*Y[1,k].
    # For fixed i, that's 2 terms (rank-2 in the (k) output with the 2-dim input).
    # The 3 rows are independent, so total rank = 3*2 = 6. YES!
    #
    # So M(3,2,2) has rank 6, not 12! Because the "2" middle dim gives only 2 terms per row.
    #
    # Wait, that contradicts schoolbook 3*2*2=12. The schoolbook counts (i, a, k) triples = 12.
    # But Z[i,k] = sum_a X[i,a]*Y[a,k]. For fixed i, sum over a (2 values) and output k (2 values).
    # The bilinear terms: X[i,0]*Y[0,k] and X[i,1]*Y[1,k] for each k. For k=0: X[i,0]*Y[0,0]+X[i,1]*Y[1,0].
    # For k=1: X[i,0]*Y[0,1]+X[i,1]*Y[1,1].
    # Total distinct bilinear monomials: 4 per i, 12 total. But can we share?
    # X[i,0] appears in both k=0 and k=1 terms: X[i,0]*Y[0,0] and X[i,0]*Y[0,1].
    # These are different because Y[0,0] != Y[0,1]. So no sharing across k.
    # Hence rank(M(3,2,2)) = 12. My "6" was wrong.
    #
    # OK so M(3,2,2) = 12. Total for 3x4x8 = 4 j-groups? No, 2 j-groups * 4 k-blocks = 8 sub-problems * 12 = 96.
    # Same as schoolbook. No gain from this split.
    #
    # The real gain must come from Strassen on 2x2x2.
    # Let me split: n=3 -> keep, m=4 -> 2+2, p=8 -> 4 blocks of 2.
    # For a 2x2x2 sub-block (j in {a,a+1}, k in {2b,2b+1}), we have a 3x2 * 2x2 = 3x2.
    # That's M(3,2,2)=12. No Strassen gain because the "2" is in m and p but n=3.
    #
    # To get Strassen, we need a 2x2x2 where ALL three dims are 2.
    # n=3 is the blocker. Split n=3 into 2+1.
    #
    # Let me try: n = 2 + 1, m = 2 + 2, p = 4 blocks of 2.
    # For the n=2 part: 2x2 * 2x2 = 2x2, use Strassen rank 7.
    # For the n=1 part: 1x2 * 2x2 = 1x2, rank 4.
    # Per (j-group, k-block): 7 + 4 = 11.
    # Number of (j-group, k-block) pairs: 2 j-groups * 4 k-blocks = 8.
    # Total: 8 * 11 = 88. Better than 96!
    #
    # Let me construct this.

    def strassen_2x2x2():
        """Return (U,V,W) lists for 2x2 * 2x2 = 2x2 with rank 7.
        U shape (2*2,7), V shape (2*2,7), W shape (2*2,7).
        Convention: C[i,k] = sum_c W[k*2+i,c]*(sum_j U[i*2+j,c]*A[i,j])*(sum_j V[j*2+k,c]*B[j,k]).
        """
        # Strassen:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # A indices: A[i*2+j], i in 0..1, j in 0..1 -> A00=0,A01=1,A10=2,A11=3
        # B indices: B[j*2+k], j in 0..1, k in 0..1 -> B00=0,B01=1,B10=2,B11=3
        # C indices: C[k*2+i], k in 0..1, i in 0..1 -> C00 (k0,i0)=0, C01(k0,i1)=1, C10(k1,i0)=2, C11(k1,i1)=3
        # Term 1: M1=(A11+A22)*(B11+B22)  -> A indices (0,0),(1,1); B (0,0),(1,1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C11 and C22 get +M1
        # Term 2: M2=(A21+A22)*B11 -> A (1,0),(1,1); B (0,0)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # C21 +M2, C22 -M2
        # Term 3: M3=A11*(B12-B22) -> A (0,0); B (0,1),(1,1)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1  # C12 +M3, C22 +M3
        # Term 4: M4=A22*(B21-B11) -> A (1,1); B (1,0),(0,0)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # C11 +M4, C21 +M4
        # Term 5: M5=(A11+A12)*B22 -> A (0,0),(0,1); B (1,1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # C11 -M5, C12 +M5
        # Term 6: M6=(A21-A11)*(B11+B12) -> A (1,0),(0,0); B (0,0),(0,1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C22 +M6
        # Term 7: M7=(A12-A22)*(B21+B22) -> A (0,1),(1,1); B (1,0),(1,1)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C11 +M7
        return U, V, W

    # Build the full decomposition.
    # n=3 -> n2=2 (rows 0,1) + n1=1 (row 2)
    # m=4 -> m2a (cols 0,1) + m2b (cols 2,3)
    # p=8 -> 4 blocks of 2: k in {0,1},{2,3},{4,5},{6,7}
    #
    # For each (j-group in {a,b}, k-block in {0..3}, n-part in {2,1}):
    #   sub-problem: n_sub x 2 x 2
    #   For n_sub=2: use Strassen (rank 7)
    #   For n_sub=1: use schoolbook rank 4
    #
    # Total terms = 2 j-groups * 4 k-blocks * (7 + 4) = 8 * 11 = 88.

    total_rank = 0
    terms = []  # list of (U_contrib, V_contrib, W_contrib) as sparse dicts
    # We'll accumulate into full matrices.
    max_rank = 2 * 4 * (7 + 4)  # 88
    Ufull = np.zeros((n * m, max_rank))
    Vfull = np.zeros((m * p, max_rank))
    Wfull = np.zeros((p * n, max_rank))
    rank = 0

    def add_term(U_sub, V_sub, W_sub, i_off, j_off, k_off):
        """Add a sub-decomposition. U_sub shape (n_sub*m_sub, r_sub), etc.
        i_off: offset in i (rows of A), j_off: offset in j, k_off: offset in k.
        n_sub, m_sub, p_sub inferred from shapes.
        """
        nonlocal rank
        n_sub = U_sub.shape[0] // m_sub
        p_sub = W_sub.shape[0] // n_sub
        r_sub = U_sub.shape[1]
        for c in range(r_sub):
            col = rank
            # U: index (i_off+ii, j_off+jj) -> flat (i_off+ii)*m + (j_off+jj)
            for flat_idx in range(n_sub * m_sub):
                ii = flat_idx // m_sub
                jj = flat_idx % m_sub
                Ufull[(i_off + ii) * m + (j_off + jj), col] = U_sub[flat_idx, c]
            # V: index (j_off+jj, k_off+kk) -> flat (j_off+jj)*p + (k_off+kk)
            for flat_idx in range(m_sub * p_sub):
                jj = flat_idx // p_sub
                kk = flat_idx % p_sub
                Vfull[(j_off + jj) * p + (k_off + kk), col] = V_sub[flat_idx, c]
            # W: index (k_off+kk, i_off+ii) -> flat (k_off+kk)*n + (i_off+ii)
            for flat_idx in range(p_sub * n_sub):
                kk = flat_idx // n_sub
                ii = flat_idx % n_sub
                Wfull[(k_off + kk) * n + (i_off + ii), col] = W_sub[flat_idx, c]
            rank += 1

    # Strassen sub (2x2x2)
    sU, sV, sW = strassen_2x2x2()
    # Schoolbook 1x2x2 sub: Z[i,k] = X[i,0]*Y[0,k] + X[i,1]*Y[1,k], i=0 only.
    def school_1x2x2():
        U = np.zeros((1 * 2, 4))
        V = np.zeros((2 * 2, 4))
        W = np.zeros((2 * 1, 4))
        # terms: (a,k) for a in 0,1; k in 0,1
        col = 0
        for a in range(2):
            for k in range(2):
                U[a, col] = 1
                V[a * 2 + k, col] = 1
                W[k * 1 + 0, col] = 1
                col += 1
        return U, V, W

    schU, schV, schW = school_1x2x2()

    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            # n=2 part (rows 0,1): Strassen 2x2x2
            add_term(sU, sV, sW, i_off=0, j_off=j_off, k_off=k_off)
            # n=1 part (row 2): schoolbook 1x2x2
            add_term(schU, schV, schW, i_off=2, j_off=j_off, k_off=k_off)

    # Trim to actual rank
    Ufull = Ufull[:, :rank]
    Vfull = Vfull[:, :rank]
    Wfull = Wfull[:, :rank]

    # Verify
    if verify(Ufull, Vfull, Wfull, trials=30):
        best_rank = rank
        best_U, best_V, best_W = Ufull.copy(), Vfull.copy(), Wfull.copy()
    else:
        # Fall back to schoolbook
        best_rank = rank_school
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # If we have time, try to push lower with a more aggressive approach.
    # Try n=3 split as 1+2 instead (symmetric), same rank. Skip.
    # Try to use rank-6 for 2x2x2? No, Strassen 7 is minimal for 2x2x2.

    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
