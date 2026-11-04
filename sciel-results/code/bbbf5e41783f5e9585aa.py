"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _start = time.time()
    def _time_left():
        return 145 - (time.time() - _start)

    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_M_fast(U_mat, V_mat, W_mat, r_cur):
        """Build M matrix efficiently using vectorized operations."""
        # M[flat_idx, c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where flat_idx = i*m*p + j*p + k
        # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
        # We need: for each (i,j,k), M[i*m*p+j*p+k, c] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # Pre-index
        U_idx = np.zeros((n, m, p, r_cur), dtype=np.float64)
        V_idx = np.zeros((n, m, p, r_cur), dtype=np.float64)
        W_idx = np.zeros((n, m, p, r_cur), dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                U_idx[i, j, :, :] = U_mat[i * m + j, :]
                V_idx[i, j, :, :] = V_mat[j * p + np.arange(p), :]
                W_idx[i, j, :, :] = W_mat[np.arange(p) * n + i, :]
        
        M = (U_idx * V_idx * W_idx).reshape(n * m * p, r_cur)
        return M

    def build_rank66():
        U66 = np.zeros((n * m, 66))
        V66 = np.zeros((m * p, 66))
        W66 = np.zeros((p * n, 66))
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U66[A00, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1
            W66[C10, col] += 1; W66[C11, col] -= 1
            col += 1

            U66[A00, col] += 1
            V66[B01, col] += 1; V66[B11, col] -= 1
            W66[C01, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A11, col] += 1
            V66[B10, col] += 1; V66[B00, col] -= 1
            W66[C00, col] += 1; W66[C10, col] += 1
            col += 1

            U66[A00, col] += 1; U66[A01, col] += 1
            V66[B11, col] += 1
            W66[C00, col] -= 1; W66[C01, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A00, col] -= 1
            V66[B00, col] += 1; V66[B01, col] += 1
            W66[C11, col] += 1
            col += 1

            U66[A01, col] += 1; U66[A11, col] -= 1
            V66[B10, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U66[A0, col] += 1; V66[B00, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B10, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A0, col] += 1; V66[B01, col] += 1; W66[C1, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B11, col] += 1; W66[C1, col] += 1
            col += 1

        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        return U66, V66, W66

    def build_rank64():
        """Build rank-64 using 3x3x3 rank-23 decomposition."""
        # We'll use a known rank-23 scheme for 3x3x3
        # For now, use a simpler approach: try to find dependencies in rank-66
        
        # Alternative: use a different partitioning
        # Let's try: A cols [0,1,2] and [3], B rows [0,1,2] and [3]
        # A1 (3x3) * B1 (3x6) + A2 (3x1) * B2 (1x6)
        # A1*B1: split B1 into two 3x3 blocks, each rank <= 23
        # A2*B2: rank 18
        # Total: 2*23 + 18 = 64
        
        # Rank-23 decomposition of 3x3x3 (one known construction)
        # This is complex; let's use a simpler rank bound
        # Actually, let's just use the rank-66 and try harder to reduce
        
        return None  # Fallback

    def aggressive_null_space(U, V, W, r):
        """Aggressively try to reduce rank by finding null space vectors."""
        best_U, best_V, best_W, best_r = U.copy(), V.copy(), W.copy(), r
        
        for iteration in range(50):
            if _time_left() < 3:
                break
            M = build_M_fast(best_U, best_V, best_W, best_r)
            try:
                U_sv, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = np.sum(s > tol)
                if rank_M >= best_r:
                    break
                null_vecs = Vt[rank_M:, :]
                found = False
                
                # Strategy 1: Try removing single columns guided by null space
                for nv in null_vecs:
                    if _time_left() < 3:
                        break
                    col_order = np.argsort(np.abs(nv))[::-1]
                    for idx_to_remove in col_order:
                        if abs(nv[idx_to_remove]) < 1e-8:
                            break
                        keep = [c for c in range(best_r) if c != idx_to_remove]
                        U_new = best_U[:, keep]
                        V_new = best_V[:, keep]
                        W_new = best_W[:, keep]
                        if verify(U_new, V_new, W_new, best_r - 1):
                            best_U, best_V, best_W, best_r = U_new, V_new, W_new, best_r - 1
                            found = True
                            break
                    if found:
                        break
                
                # Strategy 2: Try removing pairs of columns
                if not found and best_r >= 4:
                    for i_rem in range(best_r):
                        if _time_left() < 3:
                            break
                        for j_rem in range(i_rem + 1, best_r):
                            keep = [c for c in range(best_r) if c != i_rem and c != j_rem]
                            U_new = best_U[:, keep]
                            V_new = best_V[:, keep]
                            W_new = best_W[:, keep]
                            if verify(U_new, V_new, W_new, best_r - 2):
                                best_U, best_V, best_W, best_r = U_new, V_new, W_new, best_r - 2
                                found = True
                                break
                        if found:
                            break
                
                if not found:
                    break
            except:
                break
        
        return best_U, best_V, best_W, best_r

    def gf_peel(U, V, W, r, primes=[7, 11, 13, 17, 19, 23]):
        """Try to find linear dependencies over finite fields."""
        best_U, best_V, best_W, best_r = U.copy(), V.copy(), W.copy(), r
        
        def _build_M_gf(U_mat, V_mat, W_mat, r_cur, q):
            M = np.zeros((n * m * p, r_cur), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        flat_idx = i * m * p + j * p + k
                        for c in range(r_cur):
                            M[flat_idx, c] = int((U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]) % q)
            return M
        
        def _row_reduce(M, q):
            M_work = M.copy()
            rows, cols = M_work.shape
            rank = 0
            pivot_cols = []
            
            for c in range(cols):
                if rank >= rows:
                    break
                pivot = -1
                for r_idx in range(rank, rows):
                    if M_work[r_idx, c] != 0:
                        pivot = r_idx
                        break
                if pivot == -1:
                    continue
                if pivot != rank:
                    M_work[[rank, pivot]] = M_work[[pivot, rank]]
                inv = pow(int(M_work[rank, c]), q - 2, q)
                M_work[rank, :] = (M_work[rank, :] * inv) % q
                for r_idx in range(rows):
                    if r_idx != rank and M_work[r_idx, c] != 0:
                        factor = M_work[r_idx, c]
                        M_work[r_idx, :] = (M_work[r_idx, :] - factor * M_work[rank, :]) % q
                pivot_cols.append(c)
                rank += 1
            
            return rank, pivot_cols
        
        for q in primes:
            if _time_left() < 5:
                break
            # Iteratively try to remove columns
            for attempt in range(20):
                if _time_left() < 3:
                    break
                M = _build_M_gf(best_U, best_V, best_W, best_r, q)
                rank, pivot_cols = _row_reduce(M, q)
                
                if rank >= best_r:
                    break
                
                free_cols = [c for c in range(best_r) if c not in pivot_cols]
                if not free_cols:
                    break
                
                found = False
                for fc in free_cols:
                    if _time_left() < 3:
                        break
                    keep = [c for c in range(best_r) if c != fc]
                    U_new = best_U[:, keep]
                    V_new = best_V[:, keep]
                    W_new = best_W[:, keep]
                    if verify(U_new, V_new, W_new, best_r - 1):
                        best_U, best_V, best_W, best_r = U_new, V_new, W_new, best_r - 1
                        found = True
                        break
                
                if not found:
                    break
        
        return best_U, best_V, best_W, best_r

    def schoolbook():
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank_school

    # Build initial rank-66 decomposition
    U66, V66, W66 = build_rank66()

    if not verify(U66, V66, W66, 66):
        U, V, W, r = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    U_best, V_best, W_best, r_best = U66.copy(), V66.copy(), W66.copy(), 66

    # Multiple passes of reduction strategies
    for pass_num in range(5):
        if _time_left() < 5:
            break
        
        prev_r = r_best
        
        # Try GF peeling
        if _time_left() > 8:
            U_try, V_try, W_try, r_try = gf_peel(U_best, V_best, W_best, r_best)
            if r_try < r_best:
                U_best, V_best, W_best, r_best = U_try, V_try, W_try, r_try
        
        # Try rational null space
        if _time_left() > 8:
            U_try, V_try, W_try, r_try = aggressive_null_space(U_best, V_best, W_best, r_best)
            if r_try < r_best:
                U_best, V_best, W_best, r_best = U_try, V_try, W_try, r_try
        
        # If no progress, break early
        if r_best == prev_r and pass_num > 1:
            break

    if r_best > 72:
        U, V, W, r = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
