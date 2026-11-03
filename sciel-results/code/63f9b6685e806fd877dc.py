import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin

    def Aidx(i, j): return i * m + j
    def Bidx(j, k): return j * p + k
    def Cidx(i, k): return k * n + i

    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i * m + j][l] * V[j * p + k][l] * W[k * n + i][l]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_block_decomposition():
        """Build a block-structured decomposition using Strassen for 2x2 blocks
        and direct multiplication for mixed blocks.
        
        n=3, m=4, p=6.
        Split: n into 2+1, m into 2+2, p into 2+2+2.
        """
        U = []
        V = []
        W = []
        rank = 0
        
        def add_term(u_entries, v_entries, w_entries):
            """Add one rank-1 term. u_entries: dict (i,j)->val, v_entries: dict (j,k)->val, w_entries: dict (i,k)->val"""
            nonlocal rank
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            for (i, j), val in u_entries.items():
                u_vec[Aidx(i, j)] += val
            for (j, k), val in v_entries.items():
                v_vec[Bidx(j, k)] += val
            for (i, k), val in w_entries.items():
                w_vec[Cidx(i, k)] += val
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            rank += 1
        
        def strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
            """Strassen for A[a_i0:a_i0+2, a_j0:a_j0+2] * B[b_j0:b_j0+2, b_k0:b_k0+2] -> C[c_i0:c_i0+2, c_k0:c_k0+2]"""
            # M1 = (A00+A11)(B00+B11)
            add_term(
                {(a_i0, a_j0): 1, (a_i0+1, a_j0+1): 1},
                {(b_j0, b_k0): 1, (b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): 1, (c_i0+1, c_k0+1): 1}
            )
            # M2 = (A10+A11)B00
            add_term(
                {(a_i0+1, a_j0): 1, (a_i0+1, a_j0+1): 1},
                {(b_j0, b_k0): 1},
                {(c_i0+1, c_k0): 1, (c_i0+1, c_k0+1): -1}
            )
            # M3 = A00(B01-B11)
            add_term(
                {(a_i0, a_j0): 1},
                {(b_j0, b_k0+1): 1, (b_j0+1, b_k0+1): -1},
                {(c_i0, c_k0+1): 1, (c_i0+1, c_k0+1): 1}
            )
            # M4 = A11(B10-B00)
            add_term(
                {(a_i0+1, a_j0+1): 1},
                {(b_j0+1, b_k0): 1, (b_j0, b_k0): -1},
                {(c_i0, c_k0): 1, (c_i0+1, c_k0): 1}
            )
            # M5 = (A00+A01)B11
            add_term(
                {(a_i0, a_j0): 1, (a_i0, a_j0+1): 1},
                {(b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): -1, (c_i0, c_k0+1): 1}
            )
            # M6 = (A10-A00)(B00+B01)
            add_term(
                {(a_i0+1, a_j0): 1, (a_i0, a_j0): -1},
                {(b_j0, b_k0): 1, (b_j0, b_k0+1): 1},
                {(c_i0+1, c_k0+1): 1}
            )
            # M7 = (A11-A01)(B10+B11)
            add_term(
                {(a_i0+1, a_j0+1): 1, (a_i0, a_j0+1): -1},
                {(b_j0+1, b_k0): 1, (b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): 1}
            )
        
        def direct_2x2_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
            """Direct multiplication for 2x2 * 2x2 -> 2x2"""
            for di in range(2):
                for dk in range(2):
                    for dj in range(2):
                        add_term(
                            {(a_i0 + di, a_j0 + dj): 1},
                            {(b_j0 + dj, b_k0 + dk): 1},
                            {(c_i0 + di, c_k0 + dk): 1}
                        )
        
        def direct_1x2_block(a_i, a_j0, b_j0, b_k0, c_i, c_k0):
            """Direct multiplication for 1x2 * 2x2 -> 1x2"""
            for dk in range(2):
                for dj in range(2):
                    add_term(
                        {(a_i, a_j0 + dj): 1},
                        {(b_j0 + dj, b_k0 + dk): 1},
                        {(c_i, c_k0 + dk): 1}
                    )
        
        # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        strassen_block(0, 0, 0, 0, 0, 0)
        direct_2x2_block(0, 2, 2, 0, 0, 0)
        
        # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
        strassen_block(0, 0, 0, 2, 0, 2)
        direct_2x2_block(0, 2, 2, 2, 0, 2)
        
        # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
        strassen_block(0, 0, 0, 4, 0, 4)
        direct_2x2_block(0, 2, 2, 4, 0, 4)
        
        # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
        direct_1x2_block(2, 0, 0, 0, 2, 0)
        direct_1x2_block(2, 2, 2, 0, 2, 0)
        
        # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
        direct_1x2_block(2, 0, 0, 2, 2, 2)
        direct_1x2_block(2, 2, 2, 2, 2, 2)
        
        # C[2:3, 4:6] = A[2:3,0:2]*B[0:2,4:6] + A[2:3,2:4]*B[2:4,4:6]
        direct_1x2_block(2, 0, 0, 4, 2, 4)
        direct_1x2_block(2, 2, 2, 4, 2, 4)
        
        U = np.array(U).T  # shape (n*m, rank)
        V = np.array(V).T  # shape (m*p, rank)
        W = np.array(W).T  # shape (p*n, rank)
        return U, V, W, rank

    # Build the block decomposition
    U, V, W, rank = build_block_decomposition()
    
    # Verify
    if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank using APRIL
    def check_exact_fast(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.all(np.abs(T - 1.0) <= tol)

    def april_step(U, V, W, lr=0.05):
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        dU = np.einsum('jkr,kir,ijk->ijr', V_t, W_t, T - 1.0)
        U_t -= lr * dU
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        dV = np.einsum('ijr,kir,ijk->jkr', U_t, W_t, T - 1.0)
        V_t -= lr * dV
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        dW = np.einsum('ijr,jkr,ijk->kir', U_t, V_t, T - 1.0)
        W_t -= lr * dW
        
        return U_t.reshape(n*m, R), V_t.reshape(m*p, R), W_t.reshape(p*n, R)

    # Run APRIL for a few iterations
    if time.time() < deadline - 10:
        U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            U_cur, V_cur, W_cur = april_step(U_cur, V_cur, W_cur, lr=0.01)
            # Check if we can round to half-integers and still be valid
            Ur = np.round(U_cur * 2) / 2.0
            Vr = np.round(V_cur * 2) / 2.0
            Wr = np.round(W_cur * 2) / 2.0
            if check_exact_fast(Ur, Vr, Wr):
                # Try to reduce rank by removing zero columns
                mask = (np.abs(Ur) > 1e-9) | (np.abs(Vr) > 1e-9) | (np.abs(Wr) > 1e-9)
                mask = mask.any(axis=0)
                Ur = Ur[:, mask]
                Vr = Vr[:, mask]
                Wr = Wr[:, mask]
                if check_exact_fast(Ur, Vr, Wr) and Ur.shape[1] < best_rank:
                    best_rank = Ur.shape[1]
                    best_U, best_V, best_W = Ur, Vr, Wr

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
