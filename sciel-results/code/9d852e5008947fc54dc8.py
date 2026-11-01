import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    dim_u = n * m  # 4
    dim_v = m * p  # 10
    dim_w = p * n  # 10

    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def verify(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    # ============================================================
    # Rank-14 decomposition: rank-13 for 2x2x4 + 1 schoolbook col
    # ============================================================
    # The 2x2x4 multiplication tensor T_{i,j,k} for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # has rank 13. We use a known construction.
    #
    # For 2x2x2, Strassen gives rank 7.
    # For 2x2x3, rank 11 is known (Strassen + refinements).
    # For 2x2x4, rank 13 is known.
    #
    # Construction approach: Use the fact that T_{2,2,4} can be built from
    # T_{2,2,2} (rank 7) + T_{2,2,2} (rank 7) - overlap (rank 1) = rank 13.
    # More precisely, the 4 k-indices split into two pairs, and we use
    # Strassen for each pair, with a correction term.
    #
    # Actually, let me use a direct construction. The rank-13 for 2x2x4
    # can be derived from the following:
    #
    # For k in {0,1}: Strassen on (A[:,0:2] * B[0:2,0:2])
    # For k in {2,3}: Strassen on (A[:,0:2] * B[0:2,2:4])
    # But that gives rank 14. We need to save 1.
    #
    # Alternative: Use the known rank-13 construction directly.
    # Let me construct it explicitly.
    #
    # The 2x2x4 tensor: T[i,j,k] = 1 iff (i,j,k) is a valid index.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # U indices: i*m+j = i*2+j -> 0,1,2,3
    # V indices: j*p+k = j*4+k -> 0,1,2,3,4,5,6,7,8,9 (but for 2x2x4: j*4+k, j in {0,1}, k in {0,1,2,3})
    # W indices: k*n+i = k*2+i -> 0,1,2,3,4,5,6,7
    #
    # Let me use a systematic approach: build rank-13 for 2x2x4, then extend to 2x2x5.

    # Actually, let me try a different approach. Let me use the known
    # rank-14 construction for 2x2x5 directly, which is simpler to verify.
    #
    # Strategy: Use Strassen (rank 7) for the first 2 k-values,
    # Strassen (rank 7) for the next 2 k-values,
    # and 1 schoolbook for the last k-value. Total: 7+7+1 = 15.
    # That's not 14.
    #
    # Better: Use a rank-13 for 2x2x4 and add 1 schoolbook = 14.
    #
    # Let me construct the rank-13 for 2x2x4 explicitly.
    #
    # The 2x2x4 tensor T has entries T[i,j,k]=1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # This is the tensor for computing C = A * B where A is 2x2, B is 2x4, C is 2x4.
    #
    # Known result: rank(T_{2,2,4}) = 13.
    #
    # Construction via "doubling" with correction:
    # T_{2,2,4} = T_{2,2,2}^{(k=0,1)} + T_{2,2,2}^{(k=2,3)} - correction
    #
    # Let me just try to find it via a constrained optimization or
    # use a known explicit construction.
    #
    # For now, let me use a rank-15 construction (7+7+1) which I can
    # verify easily, and then try to improve to 14 or 13 via local search.

    # Rank-15 construction: Strassen for k=0,1 + Strassen for k=2,3 + schoolbook for k=4
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6

    # For 2x2x4, we have two 2x2x2 blocks:
    # Block 1: B columns 0,1 -> C columns 0,1
    # Block 2: B columns 2,3 -> C columns 2,3

    # Let me build this explicitly.
    # U has 4 rows (i*2+j), V has 10 rows (j*5+k), W has 10 rows (k*2+i)

    # For the Strassen part for k=0,1 (block 1):
    # The 7 Strassen terms involve products of A entries and B entries.
    # A entries: A00, A01, A10, A11 -> U indices 0,1,2,3
    # B entries for k=0,1: B00,B01,B10,B11 -> V indices 0,1,2,3
    # C entries for k=0,1: C00,C01,C10,C11 -> W indices 0,1,2,3

    # For Strassen block 1 (k=0,1):
    # M1: (A00+A11)(B00+B11)
    #   U contribution: A00 and A11 -> but Strassen uses sums, so we need
    #   separate rank-1 terms for each product.
    #
    # Actually, Strassen's 7 terms are already rank-1 in the tensor sense
    # when we write them out. Let me be more careful.
    #
    # T_{2,2,2}[i,j,k] = 1 for all i,j,k in {0,1}
    # Strassen decomposition:
    # C00 = (A00+A11)(B00+B11) - (A00+A01)B11 + A00(B10-B11) - A11(B01-B10) ... no
    #
    # Let me just use the standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    #
    # Each M is a product of a linear form in A and a linear form in B.
    # So each M contributes one rank-1 term to the tensor.
    #
    # M1: (A00+A11)(B00+B11)
    #   U: A00 gets +1, A11 gets +1 -> U[0,r]=1, U[3,r]=1
    #   V: B00 gets +1, B11 gets +1 -> V[0,r]=1, V[3,r]=1
    #   W: C00 gets +1, C11 gets +1 -> W[0,r]=1, W[3,r]=1
    #
    # Wait, that's not right. The W part depends on how C is assembled.
    # Actually, in the tensor decomposition, each rank-1 term is:
    #   u * v * w
    # where u is a vector in U-space, v in V-space, w in W-space.
    #
    # For M1 = (A00+A11)(B00+B11), this contributes to C00 and C11.
    # But in the tensor, each (i,j,k) entry is separate.
    #
    # Let me think differently. The tensor T_{2,2,2} has 8 entries.
    # T[i,j,k] = 1 for all i,j,k.
    #
    # Strassen says we can compute all 8 products with 7 multiplications.
    # In tensor language, rank(T_{2,2,2}) = 7.
    #
    # The 7 rank-1 terms are:
    # r1: (A00+A11) * B00 * C00  ... no, that's not right either.
    #
    # Let me look at this more carefully. The matrix multiplication tensor
    # for 2x2x2 is:
    # T = sum_{i,j,k} e_i^U ⊗ e_j^V ⊗ e_k^W
    # where e_i^U = e_{i*2+j} in U-space, etc.
    #
    # Actually, T[i*2+j, j*2+k, k*2+i] = 1.
    #
    # Strassen's decomposition gives 7 rank-1 terms that sum to T.
    # Let me write them out:
    #
    # Term 1: (A00+A11)(B00+B11) contributes to C00 and C11
    #   In tensor form: (e0+e3) ⊗ (e0+e3) ⊗ (e0+e3) ... no.
    #
    # I think I'm overcomplicating this. Let me just use a numerical
    # approach to find the rank-13 or rank-14 decomposition.

    # ============================================================
    # Numerical approach: Use gradient descent / local search
    # ============================================================

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def try_gradient_descent(R, max_restarts=30, steps_per_restart=2000, lr=0.005):
        """Try to find rank-R decomposition via gradient descent."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None

            # Initialize with small random values
            U = np.random.randn(dim_u, R) * 0.5
            V = np.random.randn(dim_v, R) * 0.5
            W = np.random.randn(dim_w, R) * 0.5

            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return None

                recon = np.einsum('ar,br,cr->abc', U, V, W)
                residual = recon - T

                # Gradients
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V)

                # Adaptive learning rate
                loss = np.sum(residual**2)
                if loss < 1e-12:
                    break

                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

            # Snap and verify
            U_s = snap_to_half_int(U)
            V_s = snap_to_half_int(V)
            W_s = snap_to_half_int(W)
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s

            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify(U_i, V_i, W_i):
                return U_i, V_i, W_i

        return None

    def try_local_search(R, max_attempts=50000):
        """Local search: start from schoolbook rank-20, remove columns, fix."""
        # Start with schoolbook
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))

        # Initialize first R columns from schoolbook (take first R of 20)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= R:
                        break
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
                    if col >= R:
                        break
                if col >= R:
                    break
            if col >= R:
                break

        # Now try to fix the decomposition by adjusting entries
        # Use coordinate descent
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None

            # Try random perturbations
            U_p = U.copy()
            V_p = V.copy()
            W_p = W.copy()
            mats = [U_p, V_p, W_p]

            # Perturb some entries
            for _ in range(3):
                mat = mats[np.random.randint(0, 3)]
                idx = np.random.randint(0, mat.shape[0])
                col_idx = np.random.randint(0, mat.shape[1])
                mat[idx, col_idx] += np.random.choice([-1.0, -0.5, 0.5, 1.0])

            if verify(U_p, V_p, W_p):
                return U_p, V_p, W_p

            # Also try snapping
            U_s = snap_to_half_int(U_p)
            V_s = snap_to_half_int(V_p)
            W_s = snap_to_half_int(W_p)
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s

        return None

    # ============================================================
    # Try ranks from 13 upward
    # ============================================================
    for R in [13, 14, 15, 16, 17, 18, 19]:
        if time.time() > deadline:
            break

        # Try gradient descent first
        result = try_gradient_descent(R, max_restarts=20, steps_per_restart=1500)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Try local search
        result = try_local_search(R, max_attempts=20000)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ============================================================
    # Fallback: schoolbook rank-20
    # ============================================================
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
