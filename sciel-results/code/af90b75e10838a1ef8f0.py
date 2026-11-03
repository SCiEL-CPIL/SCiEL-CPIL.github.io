import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    def build_matrix(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T = np.einsum('ic,kc,lc->cilk', Ua, Va, Wa)
        return T.reshape(rank, -1)

    def get_row_pivot_order(M, tol=1e-9):
        """Use QR with column pivoting on M.T to find which rows are independent."""
        if M.shape[0] == 0:
            return []
        # QR with column pivoting on M.T: M.T = Q * R * P
        # The first r columns of P correspond to independent rows of M
        try:
            # Use scipy if available for more robust pivoting
            from scipy.linalg import qr
            Q, R, P = qr(M.T, pivoting=True)
            # P is the permutation: M.T[:, P] = Q @ R
            # So M[P] = (Q @ R).T
            # The diagonal of R gives the pivots
            diag = np.abs(np.diag(R))
            s_max = diag[0] if len(diag) > 0 and diag[0] > 0 else 1.0
            tol_val = tol * s_max
            # Find how many pivots are significant
            num_pivots = int(np.sum(diag > tol_val))
            # The independent rows are M[P[:num_pivots]]
            # So the redundant rows are M[P[num_pivots:]]
            independent = P[:num_pivots].tolist()
            redundant = P[num_pivots:].tolist()
            return independent, redundant
        except Exception:
            # Fallback to SVD-based approach
            try:
                s = np.linalg.svd(M, compute_uv=False)
                tol_val = tol * max(s[0], 1.0) if len(s) > 0 else 0
                num_rank = int(np.sum(s > tol_val))
                # Just return all rows as independent if we can't determine
                return list(range(M.shape[0])), []
            except:
                return list(range(M.shape[0])), []

    def prune_decomposition(U, V, W, max_iter=100):
        current_U, current_V, current_W = U, V, W
        current_rank = len(U[0])
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
                
            M = build_matrix(current_U, current_V, current_W, current_rank)
            
            # Get pivot order to identify redundant rows
            try:
                independent, redundant = get_row_pivot_order(M)
            except:
                break
            
            if not redundant:
                break
            
            # Try removing redundant rows one by one (starting with the last pivot)
            removed_any = False
            for rem_idx in reversed(redundant):
                if time.time() > deadline - 5.0:
                    break
                
                U_new = [row for i, row in enumerate(current_U) if i != rem_idx]
                V_new = [row for i, row in enumerate(current_V) if i != rem_idx]
                W_new = [row for i, row in enumerate(current_W) if i != rem_idx]
                
                new_rank = current_rank - 1
                
                if new_rank == 0:
                    # Only valid if the tensor is zero (it's not)
                    continue
                
                # Verify the new decomposition still reconstructs the target tensor
                Ua = np.array(U_new, dtype=float)
                Va = np.array(V_new, dtype=float)
                Wa = np.array(W_new, dtype=float)
                R_new = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
                
                # Quick check: compare with original reconstruction
                Ua_orig = np.array(current_U, dtype=float)
                Va_orig = np.array(current_V, dtype=float)
                Wa_orig = np.array(current_W, dtype=float)
                R_orig = np.einsum('ic,kc,lc->ikl', Ua_orig, Va_orig, Wa_orig)
                
                if np.allclose(R_new, R_orig, atol=1e-8):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank = new_rank
                    removed_any = True
                    break  # Restart with updated matrices
                    
            if not removed_any:
                break
                
        return current_U, current_V, current_W, current_rank

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
                    
        if not np.allclose(R, target, atol=1e-9):
            return False
        return True

    def get_schoolbook():
        # Rank n*m*p = 72
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_66():
        # Reconstruct the 66 rank solution from the seed
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

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

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
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

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W

    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Generate candidates
    candidates = []
    
    # 1. The 66-rank solution
    U66, V66, W66 = get_strassen_66()
    candidates.append((U66, V66, W66))
    
    # 2. The 72-rank schoolbook solution
    U72, V72, W72 = get_schoolbook()
    candidates.append((U72, V72, W72))
    
    # Prune each candidate
    for U, V, W in candidates:
        if time.time() > deadline - 10.0:
            break
            
        U_p, V_p, W_p, r_p = prune_decomposition(U, V, W)
        
        # Verify the pruned decomposition
        if verify(U_p, V_p, W_p):
            if r_p < best_rank:
                best_rank = r_p
                best_U, best_V, best_W = U_p, V_p, W_p
        else:
            # If pruning broke it (due to numerical issues), fall back to original if valid
            if verify(U, V, W):
                r_orig = len(U[0])
                if r_orig < best_rank:
                    best_rank = r_orig
                    best_U, best_V, best_W = U, V, W

    # If we didn't find anything better than 66, return 66
    if best_U is None:
        best_U, best_V, best_W = get_strassen_66()
        best_rank = 66

    return {"U": best_U, "V": best_V, "W": best_W}
