import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = 2, 2, 15
    
    # Fallback: Rank 53 construction
    def get_rank_53():
        rank = 53
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for 2x2x2
        U_s = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_s = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W_s = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        # Schoolbook for 2x2x1
        U_sc = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_sc = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W_sc = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        col = 0
        for b in range(7):
            k0 = 2 * b
            k1 = 2 * b + 1
            for r in range(7):
                c = col + r
                for i in range(2):
                    for j in range(2):
                        U[i*m + j][c] = U_s[i*2 + j][r]
                for j in range(2):
                    V[j*p + k0][c] = V_s[j*2 + 0][r]
                    V[j*p + k1][c] = V_s[j*2 + 1][r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk*n + i][c] = W_s[k*2 + i][r]
            col += 7
            
        k_last = 14
        for r in range(4):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_sc[i*2 + j][r]
            for j in range(2):
                V[j*p + k_last][c] = V_sc[j][r]
            for i in range(2):
                W[k_last*n + i][c] = W_sc[i][r]
                
        return {"U": U, "V": V, "W": W}

    def build_target_tensor(n, m, p):
        """Build the matrix multiplication tensor for M_{n,m,p}."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T

    def tensor_from_factors(U, V, W):
        """Compute tensor from factor matrices."""
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        nm, rank = U.shape
        mp = V.shape[0]
        pn = W.shape[0]
        T = np.zeros((nm, mp, pn))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T

    def solve_tensor_decomposition(T, target_rank, max_iters=500, lr=0.001, seed=42, time_budget=60):
        """
        Try to find a rank decomposition using iterative method.
        Returns (U, V, W) if successful, None otherwise.
        """
        np.random.seed(seed)
        nm, mp, pn = T.shape
        
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            # Compute current tensor
            T_curr = tensor_from_factors(U, V, W)
            err = T_curr - T
            loss = np.sum(err**2)
            
            if loss < 1e-12:
                return U, V, W
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Compute gradients
            # dT/du_r = v_r ⊗ w_r, so dL/du_r = 2 * <err, v_r ⊗ w_r>
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(target_rank):
                # err contracted with V[:,r] and W[:,r]
                # grad_U[:,r] = sum_{b,c} err[:,b,c] * V[b,r] * W[c,r]
                temp = err @ V[:, r]  # shape (nm, pn)
                grad_U[:, r] = temp @ W[:, r]
                
                temp = err @ W[:, r]  # shape (nm, mp)
                grad_V[:, r] = temp @ U[:, r]
                
                temp = err @ U[:, r]  # shape (mp, pn)
                grad_W[:, r] = temp @ V[:, r]
            
            # Update with learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if we found a good solution
        T_curr = tensor_from_factors(best_U, best_V, best_W)
        if np.linalg.norm(T_curr - T) < 1e-9:
            return best_U, best_V, best_W
        return None

    def try_sliding_window():
        """
        Try sliding window approach: decompose 15 columns into overlapping 3-column windows.
        Each 2x2x3 block can potentially be rank 10 (vs 2*7=14 for two independent 2x2x2 blocks).
        With windows at positions [0,3], [3,6], [6,9], [9,12], [12,14], we get 5 windows.
        If each is rank 10, total = 50. If we can share, maybe less.
        """
        # First, try to find a rank-10 decomposition for 2x2x3
        def build_tensor_223():
            T = np.zeros((4, 6, 6))
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        T[i*2+j, j*3+k, k*2+i] = 1.0
            return T
        
        def try_rank_10_223():
            T = build_tensor_223()
            for rank in [10, 11, 12]:
                result = solve_tensor_decomposition(T, rank, max_iters=300, lr=0.001, seed=42, time_budget=15)
                if result is not None:
                    return rank, result
            return None, None
        
        # Try to find 2x2x3 decomposition
        rank_223, factors_223 = try_rank_10_223()
        
        if factors_223 is None:
            return None
        
        r_223, (U_223, V_223, W_223) = rank_223, factors_223
        
        # Now try to combine windows with sharing
        # Windows: [0,3), [3,6), [6,9), [9,12), [12,15)
        # Each window covers 3 columns, uses rank_223 terms
        # Total without sharing: 5 * rank_223
        # With sharing between adjacent windows, we might reduce
        
        # For now, just concatenate (no sharing) to see if we beat 53
        total_rank = 5 * rank_223
        if total_rank >= 53:
            return None
        
        U = [[0] * total_rank for _ in range(4)]
        V = [[0] * total_rank for _ in range(30)]
        W = [[0] * total_rank for _ in range(30)]
        
        col_offset = 0
        for w in range(5):
            k_start = 3 * w
            for r in range(rank_223):
                c = col_offset + r
                for i in range(4):
                    U[i][c] = U_223[i, r]
                for j in range(2):
                    for k in range(3):
                        V[j*15 + k_start + k][c] = V_223[j*3+k, r]
                for k in range(3):
                    for i in range(2):
                        W[(k_start + k)*2 + i][c] = W_223[(k)*2+i, r]
            col_offset += rank_223
        
        return {"U": U, "V": V, "W": W}

    def try_lower_rank():
        """Try to find lower rank decompositions for the full 2x2x15 problem."""
        # First try sliding window approach
        result = try_sliding_window()
        if result is not None:
            return result
        
        T = build_target_tensor(n, m, p)
        nm, mp, pn = T.shape
        
        # Try various ranks starting from low
        for rank in [40, 45, 50]:
            if time.time() > deadline - 30:
                break
            
            result = solve_tensor_decomposition(T, rank, max_iters=200, lr=0.0005, seed=42, time_budget=25)
            if result is not None:
                U, V, W = result
                # Convert to list format
                U_list = [[U[i, r] for r in range(rank)] for i in range(nm)]
                V_list = [[V[i, r] for r in range(rank)] for i in range(mp)]
                W_list = [[W[i, r] for r in range(rank)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try to find a lower rank solution
    result = try_lower_rank()
    if result is not None:
        return result
    
    return get_rank_53()
