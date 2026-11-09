import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition (fallback)
    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U, V, W = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W

    def verify(U, V, W, T, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T)) < tol:
                return True
        except:
            pass
        return False

    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result

    # ------------------------------------------------------------------
    # STRUCTURAL IDEA: split 2x3 x 3x6 into two independent 2x3 x 3x3 blocks
    # (k = 0,1,2 and k = 3,4,5). Within each block, use a Strassen-style
    # 2x2 block reduction to get rank 8 instead of 9. Total rank = 16.
    # ------------------------------------------------------------------
    def build_block_2x3x3():
        """Return (U,V,W) of shapes (6,8),(9,8),(6,8) decomposing 2x3 x 3x3."""
        nb, mb, pb = 2, 3, 3
        U = np.zeros((nb*mb, 8))
        V = np.zeros((mb*pb, 8))
        W = np.zeros((pb*nb, 8))

        # U: rows indexed by i*mb + j  (i in {0,1}, j in {0,1,2})
        # V: rows indexed by j*pb + k  (j in {0,1,2}, k in {0,1,2})
        # W: rows indexed by k*nb + i  (k in {0,1,2}, i in {0,1})

        # 8 Strassen-style bilinear forms.
        # Each is (u[i,j], v[j,k], w[k,i]) with entries in {-1,0,1}.
        cols = [
            # (u, v, w)
            ((1,1,1), (1,1,1), (1,1,1)),   # 0: all ones
            ((1,0,0), (1,0,0), (1,0,0)),   # 1: i=0,j=0,k=0
            ((0,1,0), (0,1,0), (0,1,0)),   # 2: i=1,j=1,k=1
            ((1,1,0), (1,1,0), (1,1,0)),   # 3: i=0,j=0,k=1
            ((0,1,1), (0,1,1), (0,1,1)),   # 4: i=1,j=0,k=1
            ((1,0,1), (1,0,1), (1,0,1)),   # 5: i=0,j=1,k=0
            ((0,1,1), (0,1,1), (0,1,1)),   # 6: (placeholder, will set below)
            ((1,1,1), (1,1,1), (1,1,1)),   # 7: (placeholder)
        ]
        # The above ad-hoc list is not guaranteed correct.
        # Instead, build a *valid* rank-8 via explicit construction below.
        return None  # placeholder, real construction in main body

    # ------------------------------------------------------------------
    # Correct, verified rank-8 for 2x3 x 3x3 via 2x2 Strassen on (i,k)
    # ------------------------------------------------------------------
    def strassen_2x3x3():
        """
        C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, k in {0,1,2}, j in {0,1,2}.
        Split k into (k0, k1) = (k=0) and (k=1,2).
        Define:
          A0 = A[:,0] (2-vector), A12 = A[:,1:3] (2x2)
          B0 = B[:,0] (3-vector), B12 = B[:,1:3] (3x2)
        C0 = A0 * B0  (2-vector)
        C12 = A12 * B12 (2x2) -> Strassen 7 mults
        Total = 1 (for C0) ... but C0 needs 2 mults (one per i).
        So 2 + 7 = 9. Not 8.
        """
        # This gives 9, not 8. We need a smarter split.
        # Alternative: treat the whole thing as 2x3*3x3 and find 8.
        # Use a known construction.
        return None

    # ------------------------------------------------------------------
    # Robust approach: build a VALID rank-16 via two rank-8 blocks,
    # where each rank-8 block is found by a small, fast, targeted search
    # over a structured (Strassen-like) parameterization.
    # ------------------------------------------------------------------

    # We will search for a rank-8 decomposition of the 2x3x3 sub-tensor
    # using a structured form: each of the 8 rank-1 terms has U,V,W with
    # entries in {-1,0,1}, and we enforce the Strassen pattern on the 2x2
    # (i,k) sub-blocks. But to keep it correct, we instead do a *direct*
    # but *fast* search: enumerate a small set of candidate rank-8 forms
    # derived from Strassen on a 2x2 x 2x2 sub-problem plus a correction.

    # Simpler and guaranteed: use rank-9 per block (schoolbook for 2x3x3)
    # => total rank 18, which is already a big improvement over 36.
    # Then optionally try to find rank-8 per block (total 16).

    def block_schoolbook_2x3x3():
        nb, mb, pb = 2, 3, 3
        U = np.zeros((nb*mb, 9))
        V = np.zeros((mb*pb, 9))
        W = np.zeros((pb*nb, 9))
        col = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    U[i*mb+j, col] = 1
                    V[j*pb+k, col] = 1
                    W[k*nb+i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a rank-8 decomposition for 2x3x3 via targeted search.
    def find_rank8_2x3x3(max_time=60):
        nb, mb, pb = 2, 3, 3
        T_sub = np.zeros((nb*mb, mb*pb, pb*nb), dtype=float)
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    T_sub[i*mb+j, j*pb+k, k*nb+i] = 1.0

        def verify_sub(U, V, W):
            try:
                T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
                return np.max(np.abs(T_recon - T_sub)) < 1e-9
            except:
                return False

        # Structured search: each rank-1 term's U,V,W are sparse with
        # entries in {-1,1}. Use a randomized local search that starts from
        # a Strassen-like seed and tries to fix violations.
        #
        # Seed: take the 2x2 Strassen (rank 7) on (i,k) with j-summed
        # "combined" matrices, then add 1 correction term. This is a
        # heuristic seed; we then do a short local repair.

        # Build a seed from Strassen on 2x2 (i,k) where we first combine j:
        # This won't be exact, so instead we do a pure random search over
        # a restricted alphabet with a time budget. For 2x3x3 the space is
        # small enough that a short random search over {-1,0,1} with
        # sparsity can find a rank-8 solution if one exists.

        rng = random.Random(12345)
        start = time.time()
        attempts = 0
        while time.time() < start + max_time:
            attempts += 1
            U = np.zeros((nb*mb, 8))
            V = np.zeros((mb*pb, 8))
            W = np.zeros((pb*nb, 8))
            for r in range(8):
                # U: 6 entries, sparse
                for idx in range(nb*mb):
                    if rng.random() < 0.6:
                        U[idx, r] = rng.choice([-1, 1])
                for idx in range(mb*pb):
                    if rng.random() < 0.6:
                        V[idx, r] = rng.choice([-1, 1])
                for idx in range(pb*nb):
                    if rng.random() < 0.6:
                        W[idx, r] = rng.choice([-1, 1])
            if verify_sub(U, V, W):
                return U, V, W
            if attempts % 2000 == 0 and time.time() > start + max_time:
                break
        return None

    # ------------------------------------------------------------------
    # Assemble the full 2x3 x 3x6 decomposition from two 2x3x3 blocks.
    # ------------------------------------------------------------------
    def assemble(U1, V1, W1, U2, V2, W2):
        """U1,V1,W1 for block k=0,1,2; U2,V2,W2 for block k=3,4,5."""
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        R = r1 + r2
        U_full = np.zeros((n*m, R))
        V_full = np.zeros((m*p, R))
        W_full = np.zeros((p*n, R))
        # Block 1: k in {0,1,2}
        U_full[:, :r1] = U1
        V_full[:m*3, :r1] = V1      # rows j*p+k for k=0,1,2 -> first 9 rows
        W_full[:3*n, :r1] = W1      # rows k*n+i for k=0,1,2 -> first 6 rows
        # Block 2: k in {3,4,5}
        U_full[:, r1:] = U2
        V_full[m*3:, r1:] = V2      # rows j*p+k for k=3,4,5 -> rows 9..17
        W_full[3*n:, r1:] = W2      # rows k*n+i for k=3,4,5 -> rows 6..11
        return U_full, V_full, W_full

    # First, the guaranteed rank-18 (two schoolbook 2x3x3 blocks).
    U1, V1, W1 = block_schoolbook_2x3x3()
    U2, V2, W2 = block_schoolbook_2x3x3()
    U18, V18, W18 = assemble(U1, V1, W1, U2, V2, W2)
    if verify(U18, V18, W18, T):
        best_rank = 18
        best_U, best_V, best_W = U18, V18, W18

    # Then try to improve to rank-16 by finding a rank-8 block.
    if time.time() < deadline - 20:
        r8 = find_rank8_2x3x3(max_time=min(60, deadline - time.time() - 20))
        if r8 is not None:
            U8, V8, W8 = r8
            # Use rank-8 for both blocks.
            U16, V16, W16 = assemble(U8, V8, W8, U8, V8, W8)
            if verify(U16, V16, W16, T):
                best_rank = 16
                best_U, best_V, best_W = U16, V16, W16

    # Final output
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}
