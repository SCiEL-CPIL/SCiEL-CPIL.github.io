"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(mat):
        return mat.tolist()

    def verify(result, atol=1e-6):
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i * m + j, j * p + k, k * n + i] = 1.0
        return np.allclose(T, target, atol=atol)

    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def strassen_block_decomp():
        """
        Partition: n=3+1, m=3+2, p=3+3
        Use Strassen-3 (rank 7) for A11*B11 (3x3 * 3x3 -> 3x3)
        Use schoolbook for all other block products.
        Total rank = 7 + schoolbook for remaining blocks.
        """
        # Block structure:
        # A: 4x5, partitioned as (3+1) x (3+2)
        #   A11: 3x3 (i=0..2, j=0..2)
        #   A12: 3x2 (i=0..2, j=3..4)
        #   A21: 1x3 (i=3, j=0..2)
        #   A22: 1x2 (i=3, j=3..4)
        # B: 5x6, partitioned as (3+2) x (3+3)
        #   B11: 3x3 (j=0..2, k=0..2)
        #   B12: 3x3 (j=0..2, k=3..5)
        #   B21: 2x3 (j=3..4, k=0..2)
        #   B22: 2x3 (j=3..4, k=3..5)
        # C: 6x4, partitioned as (3+3) x (3+1)
        #   C11: 3x3 (k=0..2, i=0..2)
        #   C12: 3x1 (k=0..2, i=3)
        #   C21: 3x3 (k=3..5, i=0..2)
        #   C22: 3x1 (k=3..5, i=3)

        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22

        # For each block product, we add rank factors.
        # Key: for a product A_block * B_block -> C_block,
        # the tensor entries T[a,b,c] = 1 where:
        #   a = i*m+j (global A index)
        #   b = j*p+k (global B index)
        #   c = k*n+i (global C index)
        # For the block product to contribute, (i,j) must be in A_block,
        # (j,k) must be in B_block, and (k,i) must be in C_block.

        # Strassen-3 factors for 3x3x3:
        # T3[a,b,c] = 1 if a=i*3+k, b=k*3+j, c=j*3+i for i,j,k in {0,1,2}
        # We need U3 (9x7), V3 (9x7), W3 (9x7) such that
        # sum_r U3[a,r]*V3[b,r]*W3[c,r] = 1 iff a=i*3+k, b=k*3+j, c=j*3+i

        # Let me construct Strassen-3 by solving the linear system.
        # Variables: 7*9 = 63 (for U3), 63 (for V3), 63 (for W3) = 189 vars
        # Equations: 9*9*9 = 729, but only 27 are non-zero (the target entries)
        # This is underdetermined, so we can find a solution.

        # Actually, let me just use a known rank-7 scheme.
        # Strassen's original 3x3 scheme:
        # Let me define it via the 7 bilinear forms.

        # I'll use a different approach: construct the 3x3x3 tensor and find
        # a rank-7 decomposition via random init + ALS.

        def make_3x3x3_strassen():
            """Find rank-7 decomposition of 3x3x3 matrix multiplication tensor."""
            n3, m3, p3 = 3, 3, 3
            N3, M3, P3 = 9, 9, 9
            T3 = np.zeros((N3, M3, P3))
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        a = i * m3 + j
                        b = j * p3 + k
                        c = k * n3 + i
                        T3[a, b, c] = 1.0

            # Try rank 7 with ALS
            for trial in range(20):
                U3 = np.random.randn(N3, 7) * 0.5
                V3 = np.random.randn(M3, 7) * 0.5
                W3 = np.random.randn(P3, 7) * 0.5
                prev_err = np.inf
                for it in range(100):
                    H_U = np.einsum('br,cr->bcr', V3, W3).reshape(M3 * P3, 7)
                    U3 = np.linalg.lstsq(H_U, T3.reshape(N3, M3 * P3).T, rcond=None)[0].T
                    H_V = np.einsum('ar,cr->acr', U3, W3).reshape(N3 * P3, 7)
                    V3 = np.linalg.lstsq(H_V, T3.transpose(1, 0, 2).reshape(M3, N3 * P3).T, rcond=None)[0].T
                    H_W = np.einsum('ar,br->abr', U3, V3).reshape(N3 * M3, 7)
                    W3 = np.linalg.lstsq(H_W, T3.transpose(2, 0, 1).reshape(P3, N3 * M3).T, rcond=None)[0].T
                    T3_recon = np.einsum('ar,br,cr->abc', U3, V3, W3)
                    err = np.linalg.norm(T3_recon - T3)
                    if abs(prev_err - err) < 1e-10:
                        break
                    prev_err = err
                if err < 1e-6:
                    return U3, V3, W3
            return None

        U3, V3, W3 = make_3x3x3_strassen()
        if U3 is None:
            return None  # Fallback to schoolbook

        # Now build the full decomposition.
        # We'll collect U, V, W blocks and concatenate.
        U_blocks = []
        V_blocks = []
        W_blocks = []

        def add_block(U_block, V_block, W_block, a_map, b_map, c_map):
            """
            U_block: (n_a, r), V_block: (n_b, r), W_block: (n_c, r)
            a_map: dict from local a index to global a index
            b_map: dict from local b index to global b index
            c_map: dict from local c index to global c index
            """
            r = U_block.shape[1]
            n_a = len(a_map)
            n_b = len(b_map)
            n_c = len(c_map)
            U_full = np.zeros((N, r))
            V_full = np.zeros((M, r))
            W_full = np.zeros((P, r))
            for la, ga in a_map.items():
                U_full[ga, :] = U_block[la, :]
            for lb, gb in b_map.items():
                V_full[gb, :] = V_block[lb, :]
            for lc, gc in c_map.items():
                W_full[gc, :] = W_block[lc, :]
            U_blocks.append(U_full)
            V_blocks.append(V_full)
            W_blocks.append(W_full)

        # Block 1: A11*B11 -> C11 (Strassen-3, rank 7)
        # A11: i=0..2, j=0..2 -> a = i*5+j
        # B11: j=0..2, k=0..2 -> b = j*6+k
        # C11: k=0..2, i=0..2 -> c = k*4+i
        # Local indices for 3x3x3:
        # a_local = i*3+j, b_local = j*3+k, c_local = k*3+i
        # Global:
        # a_global = i*5+j, b_global = j*6+k, c_global = k*4+i

        a_map_11 = {}
        b_map_11 = {}
        c_map_11 = {}
        for i in range(3):
            for j in range(3):
                a_map_11[i * 3 + j] = i * 5 + j
                b_map_11[j * 3 + i] = j * 6 + i  # Wait, b_local = j*3+k, but here k=i? No.
                # b_local = j*3+k where k is the column of A11 = row of B11
                # In A11*B11, the "k" in a=i*3+k is the column of A11, which is the row of B11.
                # So b_local = k*3+j where k is row of B11, j is col of B11.
                # Let me re-derive:
                # A11 is 3x3, indexed by (i, k) where i=0..2, k=0..2
                # B11 is 3x3, indexed by (k, j) where k=0..2, j=0..2
                # C11 is 3x3, indexed by (i, j) where i=0..2, j=0..2
                # a = i*5+k (global A index, m=5)
                # b = k*6+j (global B index, p=6)
                # c = j*4+i (global C index, n=4)
                # For the 3x3x3 local tensor:
                # a_local = i*3+k, b_local = k*3+j, c_local = j*3+i
                pass

        # Let me redo the mapping carefully.
        # For A11*B11:
        # i in 0..2 (row of A11, row of C11)
        # k in 0..2 (col of A11, row of B11)
        # j in 0..2 (col of B11, col of C11)
        # Global A index: a = i*m + k = i*5 + k
        # Global B index: b = k*p + j = k*6 + j
        # Global C index: c = j*n + i = j*4 + i
        # Local 3x3x3 indices:
        # a_local = i*3 + k
        # b_local = k*3 + j
        # c_local = j*3 + i

        a_map_11 = {}
        b_map_11 = {}
        c_map_11 = {}
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    a_map_11[i * 3 + k] = i * 5 + k
                    b_map_11[k * 3 + j] = k * 6 + j
                    c_map_11[j * 3 + i] = j * 4 + i

        add_block(U3, V3, W3, a_map_11, b_map_11, c_map_11)

        # Block 2: A12*B21 -> C11 (schoolbook, rank 3*2*3 = 18)
        # A12: i=0..2, j=3..4 -> a = i*5+j
        # B21: j=3..4, k=0..2 -> b = j*6+k
        # C11: k=0..2, i=0..2 -> c = k*4+i
        # For each (i, j, k): a = i*5+j, b = j*6+k, c = k*4+i
        r_12_21 = 0
        U_12_21 = np.zeros((N, 3 * 2 * 3))
        V_12_21 = np.zeros((M, 3 * 2 * 3))
        W_12_21 = np.zeros((P, 3 * 2 * 3))
        col = 0
        for i in range(3):
            for j in range(3, 5):
                for k in range(3):
                    U_12_21[i * 5 + j, col] = 1
                    V_12_21[j * 6 + k, col] = 1
                    W_12_21[k * 4 + i, col] = 1
                    col += 1
        U_blocks.append(U_12_21)
        V_blocks.append(V_12_21)
        W_blocks.append(W_12_21)

        # Block 3: A11*B12 -> C12 (schoolbook, rank 3*3*3 = 27)
        # A11: i=0..2, j=0..2 -> a = i*5+j
        # B12: j=0..2, k=3..5 -> b = j*6+k
        # C12: k=3..5, i=0..2 -> c = k*4+i
        U_11_12 = np.zeros((N, 3 * 3 * 3))
        V_11_12 = np.zeros((M, 3 * 3 * 3))
        W_11_12 = np.zeros((P, 3 * 3 * 3))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3, 6):
                    U_11_12[i * 5 + j, col] = 1
                    V_11_12[j * 6 + k, col] = 1
                    W_11_12[k * 4 + i, col] = 1
                    col += 1
        U_blocks.append(U_11_12)
        V_blocks.append(V_11_12)
        W_blocks.append(W_11_12)

        # Block 4: A12*B22 -> C12 (schoolbook, rank 3*2*3 = 18)
        # A12: i=0..2, j=3..4 -> a = i*5+j
        # B22: j=3..4, k=3..5 -> b = j*6+k
        # C12: k=3..5, i=0..2 -> c = k*4+i
        U_12_22 = np.zeros((N, 3 * 2 * 3))
        V_12_22 = np.zeros((M, 3 * 2 * 3))
        W_12_22 = np.zeros((P, 3 * 2 * 3))
        col = 0
        for i in range(3):
            for j in range(3, 5):
                for k in range(3, 6):
                    U_12_22[i * 5 + j, col] = 1
                    V_12_22[j * 6 + k, col] = 1
                    W_12_22[k * 4 + i, col] = 1
                    col += 1
        U_blocks.append(U_12_22)
        V_blocks.append(V_12_22)
        W_blocks.append(W_12_22)

        # Block 5: A21*B11 -> C21 (schoolbook, rank 1*3*3 = 9)
        # A21: i=3, j=0..2 -> a = 3*5+j
        # B11: j=0..2, k=0..2 -> b = j*6+k
        # C21: k=0..2, i=3 -> c = k*4+3
        U_21_11 = np.zeros((N, 1 * 3 * 3))
        V_21_11 = np.zeros((M, 1 * 3 * 3))
        W_21_11 = np.zeros((P, 1 * 3 * 3))
        col = 0
        for j in range(3):
            for k in range(3):
                U_21_11[3 * 5 + j, col] = 1
                V_21_11[j * 6 + k, col] = 1
                W_21_11[k * 4 + 3, col] = 1
                col += 1
        U_blocks.append(U_21_11)
        V_blocks.append(V_21_11)
        W_blocks.append(W_21_11)

        # Block 6: A22*B21 -> C21 (schoolbook, rank 1*2*3 = 6)
        # A22: i=3, j=3..4 -> a = 3*5+j
        # B21: j=3..4, k=0..2 -> b = j*6+k
        # C21: k=0..2, i=3 -> c = k*4+3
        U_22_21 = np.zeros((N, 1 * 2 * 3))
        V_22_21 = np.zeros((M, 1 * 2 * 3))
        W_22_21 = np.zeros((P, 1 * 2 * 3))
        col = 0
        for j in range(3, 5):
            for k in range(3):
                U_22_21[3 * 5 + j, col] = 1
                V_22_21[j * 6 + k, col] = 1
                W_22_21[k * 4 + 3, col] = 1
                col += 1
        U_blocks.append(U_22_21)
        V_blocks.append(V_22_21)
        W_blocks.append(W_22_21)

        # Block 7: A21*B12 -> C22 (schoolbook, rank 1*3*3 = 9)
        # A21: i=3, j=0..2 -> a = 3*5+j
        # B12: j=0..2, k=3..5 -> b = j*6+k
        # C22: k=3..5, i=3 -> c = k*4+3
        U_21_12 = np.zeros((N, 1 * 3 * 3))
        V_21_12 = np.zeros((M, 1 * 3 * 3))
        W_21_12 = np.zeros((P, 1 * 3 * 3))
        col = 0
        for j in range(3):
            for k in range(3, 6):
                U_21_12[3 * 5 + j, col] = 1
                V_21_12[j * 6 + k, col] = 1
                W_21_12[k * 4 + 3, col] = 1
                col += 1
        U_blocks.append(U_21_12)
        V_blocks.append(V_21_12)
        W_blocks.append(W_21_12)

        # Block 8: A22*B22 -> C22 (schoolbook, rank 1*2*3 = 6)
        # A22: i=3, j=3..4 -> a = 3*5+j
        # B22: j=3..4, k=3..5 -> b = j*6+k
        # C22: k=3..5, i=3 -> c = k*4+3
        U_22_22 = np.zeros((N, 1 * 2 * 3))
        V_22_22 = np.zeros((M, 1 * 2 * 3))
        W_22_22 = np.zeros((P, 1 * 2 * 3))
        col = 0
        for j in range(3, 5):
            for k in range(3, 6):
                U_22_22[3 * 5 + j, col] = 1
                V_22_22[j * 6 + k, col] = 1
                W_22_22[k * 4 + 3, col] = 1
                col += 1
        U_blocks.append(U_22_22)
        V_blocks.append(V_22_22)
        W_blocks.append(W_22_22)

        # Concatenate all blocks
        U_full = np.hstack(U_blocks)
        V_full = np.hstack(V_blocks)
        W_full = np.hstack(W_blocks)

        return {"U": U_full, "V": V_full, "W": W_full}

    # Try Strassen-based construction
    strassen_result = strassen_block_decomp()
    if strassen_result is not None and verify(strassen_result):
        best = strassen_result
        best_rank = best["U"].shape[1]
        print(f"Strassen-3 based decomposition with rank {best_rank}")
    else:
        best = schoolbook()
        best_rank = n * m * p

    def als_decompose(target, rank, max_iter=50, tol=1e-10, polish_iters=10):
        U = np.random.randn(N, rank) * 0.5
        V = np.random.randn(M, rank) * 0.5
        W = np.random.randn(P, rank) * 0.5
        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 20:
                break
            H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
            target_flat = target.reshape(N, M * P)
            try:
                U = np.linalg.lstsq(H_U, target_flat.T, rcond=None)[0].T
            except:
                pass
            H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
            target_v = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V = np.linalg.lstsq(H_V, target_v.T, rcond=None)[0].T
            except:
                pass
            H_W = np.einsum('ar,br->abr', U, V).reshape(N * M, rank)
            target_w = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W = np.linalg.lstsq(H_W, target_w.T, rcond=None)[0].T
            except:
                pass
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        if time_remaining() > 25:
            for _ in range(polish_iters):
                if time_remaining() < 20:
                    break
                H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
                try:
                    U = np.linalg.lstsq(H_U, target.reshape(N, M * P).T, rcond=None)[0].T
                except:
                    pass
                H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
                try:
                    V = np.linalg.lstsq(H_V, target.transpose(1, 0, 2).reshape(M, N * P).T, rcond=None)[0].T
                except:
                    pass
                H_W = np.einsum('ar,br->abr', U, V).reshape(N * M, rank)
                try:
                    W = np.linalg.lstsq(H_W, target.transpose(2, 0, 1).reshape(P, N * M).T, rcond=None)[0].T
                except:
                    pass
        return U, V, W

    def try_rank(r, num_trials=5):
        if time_remaining() < 25:
            return None
        for trial in range(num_trials):
            if time_remaining() < 20:
                break
            U, V, W = als_decompose(target, r, max_iter=40)
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    for r in range(best_rank - 1, max(best_rank - 25, 95), -1):
        if time_remaining() < 35:
            break
        result = try_rank(r, num_trials=3)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            continue

    if not verify(best, atol=1e-5):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
