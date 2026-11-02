"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    start = time.time()
    deadline = start + 130.0

    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)

    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i

    def make_schoolbook():
        """Pure schoolbook: rank 40"""
        rb = n * m * p
        U = np.zeros((n*m, rb))
        V = np.zeros((m*p, rb))
        W = np.zeros((p*n, rb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W

    def make_strassen_36():
        """Block Strassen: rank 36"""
        cap = 50
        U = np.zeros((n * m, cap))
        V = np.zeros((m * p, cap))
        W = np.zeros((p * n, cap))
        col = 0

        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices

            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
            col += 1

            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
            col += 1

            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1]), col] = 1
            V[v_idx(z01[0], z01[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
            col += 1

            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z10[0], z10[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
            col += 1

            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1]), col] = 1
            U[u_idx(y01[0], y01[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
            col += 1

            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1]), col] = 1
            U[u_idx(y00[0], y00[1]), col] = -1
            V[v_idx(z00[0], z00[1]), col] = 1
            V[v_idx(z01[0], z01[1]), col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
            col += 1

            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = -1
            V[v_idx(z10[0], z10[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
            col += 1

        # Four 2x2x2 Strassen blocks
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)],
                       [(2,0),(3,0),(2,1),(3,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)],
                       [(2,0),(3,0),(2,1),(3,1)])

        # Block 5: A_L * B_col4 (2x2 x 2x1)
        for (i, j, r, c) in [(0,0,0,0),(0,1,1,0),(1,0,0,1),(1,1,1,1)]:
            U[u_idx(i, j), col] = 1
            V[v_idx(r, 4), col] = 1
            W[w_idx(4, c), col] += 1
            col += 1

        # Block 6: A_R * B_col4
        for (i, j, r, c) in [(0,2,2,0),(0,3,3,0),(1,2,2,1),(1,3,3,1)]:
            U[u_idx(i, j), col] = 1
            V[v_idx(r, 4), col] = 1
            W[w_idx(4, c), col] += 1
            col += 1

        return U[:, :col].copy(), V[:, :col].copy(), W[:, :col].copy()

    def verify(U, V, W):
        """Check that the decomposition exactly reconstructs the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def column_tensors(U, V, W):
        """Compute the tensor for each column (rank-1 term)."""
        r = U.shape[1]
        T = np.zeros((r, U.shape[0] * V.shape[0] * W.shape[0]))
        for c in range(r):
            T[c, :] = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).ravel()
        return T

    def try_drop_columns(U, V, W, cols_to_drop):
        """Try dropping specified columns and verify."""
        keep = [c for c in range(U.shape[1]) if c not in cols_to_drop]
        if len(keep) < 1:
            return None
        U2, V2, W2 = U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy()
        if verify(U2, V2, W2):
            return U2, V2, W2
        return None

    def reduce_single(U, V, W, max_rounds=100):
        """Iteratively try dropping single columns."""
        r = U.shape[1]
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            dropped = False
            for c in range(U.shape[1]):
                if time.time() > deadline - 10:
                    break
                res = try_drop_columns(U, V, W, [c])
                if res is not None:
                    U, V, W = res
                    dropped = True
                    break
            if not dropped:
                break
        return U, V, W

    def reduce_pairs(U, V, W, max_rounds=20):
        """Try dropping pairs of columns."""
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            r = U.shape[1]
            dropped = False
            for i in range(r):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, r):
                    if time.time() > deadline - 10:
                        break
                    res = try_drop_columns(U, V, W, [i, j])
                    if res is not None:
                        U, V, W = res
                        dropped = True
                        break
                if time.time() > deadline - 10:
                    break
                if dropped:
                    break
            if not dropped:
                break
        return U, V, W

    # --- Strategy: Try multiple base constructions, then reduce ---
    candidates = []

    # Base 1: Schoolbook rank 40
    U1, V1, W1 = make_schoolbook()
    candidates.append((U1, V1, W1))

    # Base 2: Strassen rank 36
    U2, V2, W2 = make_strassen_36()
    candidates.append((U2, V2, W2))

    # Base 3: Try a 2x3x3 Strassen for sub-blocks to get different structure
    # For 2x3 matrix * 3x3 matrix, we can use a rank-8 decomposition
    # (standard Strassen gives 7 for 2x2x2, but 2x3x3 needs more)
    # Actually let's try: split 4 into 2+2, and 5 into 3+2
    # Use Strassen for 2x2*2x3 (rank 7 per 2x2x2 block, but 2x2x3 is different)
    
    # Alternative: use a custom rank-10 decomposition for 2x3*3x5
    # Let's try: treat as 2x2*2x3 + 2x1*3x5 etc.
    
    # Actually, let's try a different partition: 2x4*4x5 as 2x3*3x5 + 2x1*1x5
    # 2x3*3x5: use Strassen-like
    # For now, stick with the two bases and do aggressive reduction

    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None

    for idx, (Uc, Vc, Wc) in enumerate(candidates):
        if time.time() > deadline - 15:
            break
        if not verify(Uc, Vc, Wc):
            continue
        
        U_r, V_r, W_r = Uc.copy(), Vc.copy(), Wc.copy()
        
        # Phase 1: single column removal
        U_r, V_r, W_r = reduce_single(U_r, V_r, W_r, max_rounds=80)
        
        # Phase 2: pair removal
        U_r, V_r, W_r = reduce_pairs(U_r, V_r, W_r, max_rounds=15)
        
        # Phase 3: try triple removal (only if time allows)
        if time.time() < deadline - 20:
            for _ in range(5):
                if time.time() > deadline - 15:
                    break
                r = U_r.shape[1]
                dropped = False
                for i in range(r):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i+1, r):
                        if time.time() > deadline - 15:
                            break
                        for k in range(j+1, r):
                            if time.time() > deadline - 15:
                                break
                            res = try_drop_columns(U_r, V_r, W_r, [i, j, k])
                            if res is not None:
                                U_r, V_r, W_r = res
                                dropped = True
                                break
                            if time.time() > deadline - 15:
                                break
                        if time.time() > deadline - 15:
                            break
                        if dropped:
                            break
                    if time.time() > deadline - 15:
                        break
                    if dropped:
                        break
                if not dropped:
                    break
        
        r = U_r.shape[1]
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()

    # If we didn't find anything better than schoolbook, return schoolbook
    if best_U is None:
        U, V, W = make_schoolbook()
    else:
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
