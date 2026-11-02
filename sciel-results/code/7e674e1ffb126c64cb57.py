import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin before 150s kill

    n, m, p = 2, 2, 15

    # Known Strassen decomposition for 2x2x2 (rank 7)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Schoolbook for 2x2x1 (rank 4)
    U_sc = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    V_sc = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    W_sc = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    def build_target_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T

    def tensor_from_factors(U, V, W):
        nm, rank = U.shape
        mp = V.shape[0]
        pn = W.shape[0]
        T = np.zeros((nm, mp, pn))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is valid."""
        # Check shapes
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        # Check tensor match
        T_target = build_target_tensor(n, m, p)
        T_found = tensor_from_factors(U, V, W)
        return np.linalg.norm(T_target - T_found) < tol

    def get_rank_53():
        """Fallback: 7 Strassen blocks for first 14 cols + 1 schoolbook for last col."""
        rank = 53
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for b in range(7):
            k0 = 2 * b
            k1 = 2 * b + 1
            for r in range(7):
                c = col + r
                for i in range(2):
                    for j in range(2):
                        U[i*m + j, c] = U_s[i*2 + j, r]
                for j in range(2):
                    V[j*p + k0, c] = V_s[j*2 + 0, r]
                    V[j*p + k1, c] = V_s[j*2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk*n + i, c] = W_s[k*2 + i, r]
            col += 7
            
        k_last = 14
        for r in range(4):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j, c] = U_sc[i*2 + j, r]
            for j in range(2):
                V[j*p + k_last, c] = V_sc[j, r]
            for i in range(2):
                W[k_last*n + i, c] = W_sc[i, r]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def get_rank_52():
        """6 Strassen blocks for first 12 cols + 1 schoolbook for last 3 cols (rank 10 assumed).
        But we don't have a verified rank 10 for 2x2x3. 
        Let's use 5 Strassen (10 cols, rank 35) + 1 block of 5 cols.
        Actually, let's just use the known 53 construction as the safe fallback.
        The previous code attempted 52 but it's risky without a verified 2x2x3 rank 10.
        Let's stick to 53 for safety, or try to find a better one.
        Wait, the task says "rank-minimal". The known lower bound is 49.
        Let's try to find a better decomposition using ALS.
        """
        # For now, return 53. The ALS will try to beat it.
        return get_rank_53()

    def als_decompose(T, target_rank, max_iters=500, tol=1e-8, seed=42, time_budget=10):
        """
        Alternating Least Squares for tensor decomposition.
        T: (nm, mp, pn)
        target_rank: int
        Returns (U, V, W) if successful, None otherwise.
        """
        np.random.seed(seed)
        nm, mp, pn = T.shape
        
        # Initialize
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            # Solve for U: fix V, W
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For each a: T[a, :, :] = sum_r U[a, r] (V[:, r] outer W[:, r])
            # Reshape: T_flat[a, (b,c)] = sum_r U[a, r] * (V[:, r] * W[:, r])_flat
            # A[r, (b,c)] = V[:, r] * W[:, r] (flattened)
            # T_flat[a, :] = U[a, :] @ A.T  =>  U[a, :] = T_flat[a, :] @ pinv(A.T)
            
            # Build A (target_rank, mp*pn)
            A = np.zeros((target_rank, mp * pn))
            for r in range(target_rank):
                A[r, :] = (V[:, r] * W[:, r]).flatten()
            
            # T_flat: (nm, mp*pn)
            T_flat = T.reshape(nm, mp * pn)
            
            # Solve U
            # U @ A.T = T_flat
            # U = T_flat @ pinv(A.T)
            try:
                U = T_flat @ np.linalg.pinv(A.T)
            except:
                pass
            
            # Solve for V: fix U, W
            # T[b, a, c] = sum_r V[b, r] U[a, r] W[c, r]
            # B[r, (a,c)] = U[:, r] * W[:, r]
            # T_transposed[b, (a,c)] = V[b, :] @ B.T
            B = np.zeros((target_rank, nm * pn))
            for r in range(target_rank):
                B[r, :] = (U[:, r] * W[:, r]).flatten()
            
            # T is (nm, mp, pn). We need T[b, a, c].
            # T.transpose(1, 0, 2) -> (mp, nm, pn)
            T_trans_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            
            try:
                V = T_trans_b @ np.linalg.pinv(B.T)
            except:
                pass
            
            # Solve for W: fix U, V
            # T[c, a, b] = sum_r W[c, r] U[a, r] V[b, r]
            # C[r, (a,b)] = U[:, r] * V[:, r]
            # T_transposed[c, (a,b)] = W[c, :] @ C.T
            C = np.zeros((target_rank, nm * mp))
            for r in range(target_rank):
                C[r, :] = (U[:, r] * V[:, r]).flatten()
            
            # T is (nm, mp, pn). We need T[c, a, b].
            # T.transpose(2, 0, 1) -> (pn, nm, mp)
            T_trans_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            
            try:
                W = T_trans_c @ np.linalg.pinv(C.T)
            except:
                pass
            
            # Check error
            T_curr = tensor_from_factors(U, V, W)
            err = np.linalg.norm(T - T_curr)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < tol:
                return best_U, best_V, best_W
        
        if best_err < tol:
            return best_U, best_V, best_W
        return None

    def try_find_rank(n, m, p, target_rank, time_budget=15):
        """Try to find a decomposition of rank target_rank for M_{n,m,p}."""
        T = build_target_tensor(n, m, p)
        for seed in [42, 123, 999, 777, 555]:
            if time.time() > time_budget:
                break
            result = als_decompose(T, target_rank, max_iters=100, tol=1e-8, seed=seed, time_budget=time_budget)
            if result is not None:
                U, V, W = result
                # Verify
                if verify_decomposition(U, V, W, n, m, p, tol=1e-7):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def try_sliding_window():
        """
        Try to decompose 2x2x15 using smaller blocks.
        15 = 5 * 3. If we can find rank 10 for 2x2x3, total rank = 5 * 10 = 50.
        15 = 3 * 5. If we can find rank 16 for 2x2x5 (schoolbook is 20, Strassen-based might be lower?), total = 48?
        Let's try 2x2x3 rank 10.
        """
        # Try 2x2x3 rank 10
        result_223 = try_find_rank(2, 2, 3, 10, time_budget=10)
        if result_223 is not None:
            # Assemble 5 blocks of 2x2x3
            U_223 = np.array(result_223["U"])
            V_223 = np.array(result_223["V"])
            W_223 = np.array(result_223["W"])
            rank_223 = U_223.shape[1]
            total_rank = 5 * rank_223
            
            U = np.zeros((4, total_rank))
            V = np.zeros((30, total_rank))
            W = np.zeros((30, total_rank))
            
            col_offset = 0
            for w in range(5):
                k_start = 3 * w
                for r in range(rank_223):
                    c = col_offset + r
                    for i in range(4):
                        U[i, c] = U_223[i, r]
                    for j in range(2):
                        for k in range(3):
                            V[j*15 + k_start + k, c] = V_223[j*3+k, r]
                    for k in range(3):
                        for i in range(2):
                            W[(k_start + k)*2 + i, c] = W_223[(k)*2+i, r]
                col_offset += rank_223
            
            # Verify full assembly
            if verify_decomposition(U, V, W, 2, 2, 15, tol=1e-7):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try 2x2x4 rank 13? (Schoolbook 16, Strassen 14)
        # If rank 13, then 3 blocks of 4 (12 cols) + 1 block of 3 (rank 10) = 3*13 + 10 = 49
        result_224 = try_find_rank(2, 2, 4, 13, time_budget=10)
        if result_224 is not None:
            U_224 = np.array(result_224["U"])
            V_224 = np.array(result_224["V"])
            W_224 = np.array(result_224["W"])
            rank_224 = U_224.shape[1]
            
            result_223 = try_find_rank(2, 2, 3, 10, time_budget=5)
            if result_223 is not None:
                U_223 = np.array(result_223["U"])
                V_223 = np.array(result_223["V"])
                W_223 = np.array(result_223["W"])
                rank_223 = U_223.shape[1]
                
                total_rank = 3 * rank_224 + rank_223
                U = np.zeros((4, total_rank))
                V = np.zeros((30, total_rank))
                W = np.zeros((30, total_rank))
                
                col_offset = 0
                # 3 blocks of 4
                for w in range(3):
                    k_start = 4 * w
                    for r in range(rank_224):
                        c = col_offset + r
                        for i in range(4):
                            U[i, c] = U_224[i, r]
                        for j in range(2):
                            for k in range(4):
                                V[j*15 + k_start + k, c] = V_224[j*4+k, r]
                        for k in range(4):
                            for i in range(2):
                                W[(k_start + k)*2 + i, c] = W_224[(k)*2+i, r]
                    col_offset += rank_224
                
                # 1 block of 3
                k_start = 12
                for r in range(rank_223):
                    c = col_offset + r
                    for i in range(4):
                        U[i, c] = U_223[i, r]
                    for j in range(2):
                        for k in range(3):
                            V[j*15 + k_start + k, c] = V_223[j*3+k, r]
                    for k in range(3):
                        for i in range(2):
                            W[(k_start + k)*2 + i, c] = W_223[(k)*2+i, r]
                    col_offset += rank_223
                
                if verify_decomposition(U, V, W, 2, 2, 15, tol=1e-7):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    def try_lower_rank_full():
        """Try to find a lower rank decomposition for the full 2x2x15 problem directly."""
        # Try ranks 49, 50, 51
        for rank in [49, 50, 51]:
            if time.time() > deadline - 10:
                break
            result = try_find_rank(2, 2, 15, rank, time_budget=20)
            if result is not None:
                return result
        return None

    # Try to find a lower rank solution
    result = try_sliding_window()
    if result is not None:
        return result
    
    result = try_lower_rank_full()
    if result is not None:
        return result
    
    return get_rank_53()
