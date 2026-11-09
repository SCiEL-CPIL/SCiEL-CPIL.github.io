"""Program searching for low-rank exact tensor decomposition of 2x3 by 3x8 matrix multiplication.

Structural approach: Winograd / frequency-domain factorisation over the 8-length
column dimension.  The 2x3x8 multiplication tensor, for fixed (i,j), is a length-8
linear combination in k.  We pack the 8 columns of B into two length-4 blocks and
apply a 4-point integer transform (entries in {0, +/-1, +/-2}) so that the
per-column products are performed in transform space.  This fuses the 8 columns
and can reduce the tensor rank below the schoolbook bound of 48.  Every candidate
is verified exactly against the target tensor; the best verified decomposition is
returned, with the schoolbook decomposition as the guaranteed fallback.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    deadline = time.time() + 140  # safety margin before the 150s wall-clock limit

    n, m, p = DIMENSIONS
    base_rank = n * m * p

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return bool(np.allclose(computed, T, atol=1e-9, rtol=1e-9))

    # ---------- Schoolbook baseline (guaranteed rank 48) ----------
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    best = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist(), "rank": base_rank}

    # ---------- Winograd / frequency-domain factorisation ----------
    #
    # For a fixed output row i and intermediate index j, the contribution to the
    # 8 outputs C[i, :] is  A[i,j] * B[j, :]  (a length-8 vector).  Winograd packs
    # the 8 columns into two length-4 blocks and applies a 4-point integer
    # transform M (with integer entries) to each block, multiplies by the
    # transformed A entry, and inverse-transforms.  Because M is a fixed integer
    # matrix, the transform can be absorbed into the V and W factor matrices,
    # yielding a lower-rank exact decomposition.
    #
    # Concretely, split the 8 columns k into two blocks of 4:
    #   block a: k in {0,1,2,3},  block b: k in {4,5,6,7}
    # Let M be a 4x4 integer transform.  Define transformed B entries:
    #   Bt[j, t, block] = sum_{k in block} M[t, k] * B[j, k]
    # and the A-side transform (applied to the scalar A[i,j] over the 4
    # transform slots is just replication scaled by the inverse transform).
    # The product in transform space is A[i,j] * Bt[j, t, block], and the
    # inverse transform recovers C[i, k].  The rank is n * m * (2 blocks) * 4
    # transform slots = 2*3*2*4 = 48 in the naive case, but the transform can
    # share slots across the two A rows (i) and across j, reducing the rank.
    #
    # We enumerate a small family of 4x4 integer transforms and build the
    # corresponding factor matrices, verifying each exactly.

    def build_winograd(M_a, M_b, inv_a, inv_b):
        """Build U, V, W for a Winograd factorisation.

        M_a, M_b: 4x4 integer transforms applied to the two length-4 blocks of B.
        inv_a, inv_b: 4x4 integer inverse transforms (inv_a @ M_a = 4*I etc., up
        to a known scale that is absorbed into U).

        The transform slots are shared across the two A rows i, so the rank is
        n * m * 2 * 4 but with i-fold sharing we get rank = m * 2 * 4 * (n) in
        the worst case; we let the verification decide.
        """
        # Each transform slot (block, t) corresponds to one "column" c of the
        # decomposition.  There are 2 blocks * 4 slots = 8 such slots.  For each
        # (i, j, slot) we need a column.  To share, we note that the A-side
        # transform for row i is a fixed row of a matrix; if the two rows i use
        # the same transform structure, the slot index is shared.
        #
        # We build the full (i, j, block, t) columns and then let the rank be
        # the number of distinct columns after removing zero columns.
        slots = []  # list of (block, t)
        for block in range(2):
            for t in range(4):
                slots.append((block, t))
        n_slots = len(slots)

        # U rows: (i*m + j), V rows: (j*p + k), W rows: (k*n + i)
        # Number of columns = n * m * n_slots  (before sharing)
        r = n * m * n_slots
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))

        col = 0
        for i in range(n):
            for j in range(m):
                for (block, t) in slots:
                    # V entry: B[j, k] transformed -> V[j*p+k, col] = M[t, k_local]
                    M = M_a if block == 0 else M_b
                    for k_local in range(4):
                        k = block * 4 + k_local
                        V[j * p + k, col] = M[t, k_local]
                    # W entry: inverse transform on the A side -> W[k*n+i, col]
                    invM = inv_a if block == 0 else inv_b
                    for k_local in range(4):
                        k = block * 4 + k_local
                        W[k * n + i, col] = invM[k_local, t]
                    # U entry: A[i, j] scalar -> U[i*m+j, col] = 1
                    U[i * m + j, col] = 1
                    col += 1

        return U, V, W

    # Candidate 4x4 integer transforms.  The identity gives schoolbook-like rank;
    # non-trivial transforms (DFT-like, Hadamard, Winograd F_4) can reduce rank
    # when the inverse transform has a small integer support.
    #
    # Winograd F_4 transform for length-4 convolution:
    #   M = [[1, 0, 1, 0],
    #        [1, 0, -1, 0],
    #        [1, 1, 1, 1],
    #        [1, -1, 1, -1]]
    # with inverse (1/4) * M_inv.  We absorb the 1/4 into U (making U entries
    # quarter-integers) -- but the task allows half-integers only.  So we scale
    # M by 4 and keep U integer, or we keep M and let U carry 1/4.  Since the
    # task allows integer/half-integer entries, we must keep U in {0, +/-1/2, +/-1, ...}.
    # We therefore use transforms whose inverse has entries that are multiples of
    # 1/2, or we scale so that U entries are half-integers.
    #
    # Strategy: enumerate transforms, build U/V/W, and if U has entries that are
    # not half-integers, scale the whole decomposition so U becomes half-integer.

    transforms = []
    # Identity (baseline within the Winograd frame)
    transforms.append(("I", np.eye(4, dtype=int)))
    # Hadamard H_4
    H4 = np.array([[1, 1, 1, 1],
                   [1, -1, 1, -1],
                   [1, 1, -1, -1],
                   [1, -1, -1, 1]], dtype=int)
    transforms.append(("H4", H4))
    # Winograd F_4
    F4 = np.array([[1, 0, 1, 0],
                   [1, 0, -1, 0],
                   [1, 1, 1, 1],
                   [1, -1, 1, -1]], dtype=int)
    transforms.append(("F4", F4))
    # DFT-like 4-point (integer approximation with +/-1)
    D4 = np.array([[1, 1, 1, 1],
                   [1, 0, -1, 0],
                   [1, -1, 1, -1],
                   [1, 0, -1, 0]], dtype=int)
    transforms.append(("D4", D4))

    def is_half_integer(arr, tol=1e-9):
        return bool(np.allclose(arr * 2, np.round(arr * 2), atol=tol))

    def make_half_integer(U, V, W, scale):
        """Scale U by 1/scale, V by scale, W unchanged so that U becomes
        half-integer (if U*scale is integer)."""
        return U / scale, V * scale, W

    for name, M in transforms:
        if time.time() > deadline:
            break
        # Inverse transform: M_inv = pinv(M) scaled to integer/half-integer.
        # For orthogonal-ish integer M, M_inv = (1/det_factor) * adj(M).
        # We compute the exact rational inverse and check half-integer-ness.
        Mf = M.astype(float)
        det = np.linalg.det(Mf)
        if abs(det) < 1e-9:
            continue
        # adjugate / det
        invM = np.linalg.inv(Mf)
        # We need invM entries to be half-integers (or we scale).
        # Build the factor matrices with U carrying the inverse-transform scale.
        # For each (i,j,slot), U entry = 1, V entry = M[t,k], W entry = invM[k,t].
        # If invM has non-half-integer entries, scale: multiply U by det, divide
        # W by det, so W entries become adj(M) entries (integers) and U becomes
        # det (integer).  This keeps the product U*V*W unchanged.
        U, V, W = build_winograd(M, M, invM, invM)
        # Check if W is half-integer; if not, scale.
        if not is_half_integer(W):
            # Scale by det: U *= det, W /= det -> W becomes adj(M)/det * det = adj(M) (integer)
            det_val = det
            U2 = U * det_val
            W2 = W / det_val
            if is_half_integer(U2) and is_half_integer(W2):
                U, V, W = U2, V, W2
        if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
            # Round to half-integers and verify
            Ur = np.round(U * 2) / 2
            Vr = np.round(V * 2) / 2
            Wr = np.round(W * 2) / 2
            if verify(Ur, Vr, Wr):
                # Remove zero columns to minimise rank
                cols = np.where(np.any(Ur != 0, axis=0) | np.any(Vr != 0, axis=0) | np.any(Wr != 0, axis=0))[0]
                Ur, Vr, Wr = Ur[:, cols], Vr[:, cols], Wr[:, cols]
                r = Ur.shape[1]
                if r < best["rank"]:
                    best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist(), "rank": r}

    # ---------- Gradient-descent refinement on the best candidate ----------
    # If a Winograd candidate with rank < 48 was found, try to polish it to
    # exact half-integer values with a few gradient-descent steps, then re-verify.
    if best["rank"] < base_rank:
        U = np.array(best["U"], dtype=float)
        V = np.array(best["V"], dtype=float)
        W = np.array(best["W"], dtype=float)
        r = U.shape[1]
        lr = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
            computed = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = computed - T
            loss = np.sum(diff * diff)
            if loss < 1e-12:
                break
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        # Round to half-integers and verify
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        if verify(Ur, Vr, Wr):
            cols = np.where(np.any(Ur != 0, axis=0) | np.any(Vr != 0, axis=0) | np.any(Wr != 0, axis=0))[0]
            Ur, Vr, Wr = Ur[:, cols], Vr[:, cols], Wr[:, cols]
            if Ur.shape[1] < best["rank"]:
                best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist(), "rank": Ur.shape[1]}

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
