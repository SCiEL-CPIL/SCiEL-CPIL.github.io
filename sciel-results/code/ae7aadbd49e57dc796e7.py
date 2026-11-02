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
        # T: (nm, mp, pn)
        return np.einsum('ar,br,cr->abc', U, V, W)

    def compute_gradients(err, U, V, W):
        """Compute gradients for U, V, W given error tensor."""
        # err: (nm, mp, pn)
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # grad_U: (nm, rank)
        grad_U = np.einsum('abc,br,cr->ar', err, V, W)
        # grad_V: (mp, rank)
        grad_V = np.einsum('abc,ar,cr->br', err, U, W)
        # grad_W: (pn, rank)
        grad_W = np.einsum('abc,ar,br->cr', err, U, V)
        return grad_U, grad_V, grad_W

    def find_rank_10_m223():
        """
        Attempt to find a rank 10 decomposition for 2x2x3 matrix multiplication.
        Returns (U, V, W) as numpy arrays if successful, None otherwise.
        """
        n, m, p = 2, 2, 3
        T_target = build_target_tensor(n, m, p)
        nm, mp, pn = T_target.shape
        target_rank = 10
        
        # We will use a simple iterative approach or random restarts with gradient descent
        # Given the small size, we can try multiple random initializations.
        
        best_U, best_V, best_W = None, None, None
        best_loss = float('inf')
        
        for seed in range(50):
            if time.time() > deadline - 20:
                break
            np.random.seed(seed)
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.001
            for it in range(1000):
                T_curr = tensor_from_factors(U, V, W)
                err = T_curr - T_target
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    return U, V, W
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Gradients
                grad_U, grad_V, grad_W = compute_gradients(err, U, V, W)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if best_loss < 1e-8:
                # Check if it's close to integers
                if np.all(np.abs(best_U - np.round(best_U)) < 1e-6) and \
                   np.all(np.abs(best_V - np.round(best_V)) < 1e-6) and \
                   np.all(np.abs(best_W - np.round(best_W)) < 1e-6):
                    U_int = np.round(best_U).astype(int)
                    V_int = np.round(best_V).astype(int)
                    W_int = np.round(best_W).astype(int)
                    # Verify exactness
                    T_check = tensor_from_factors(U_int, V_int, W_int)
                    if np.allclose(T_check, T_target, atol=1e-10):
                        return U_int, V_int, W_int
        
        return None

    def try_rank_50():
        """
        Try to construct a rank 50 solution using 5 blocks of 2x2x3 rank 10.
        """
        m223_factors = find_rank_10_m223()
        if m223_factors is None:
            return None
            
        U_m223, V_m223, W_m223 = m223_factors
        rank_block = 10
        total_rank = 5 * rank_block
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for block in range(5):
            k_offset = block * 3
            r_offset = block * rank_block
            
            for r in range(rank_block):
                c = r_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i*m + j][c] = U_m223[i*2 + j, r]
                for j in range(2):
                    for k in range(3):
                        V[j*p + (k_offset + k)][c] = V_m223[j*3 + k, r]
                for k in range(3):
                    for i in range(2):
                        W[(k_offset + k)*n + i][c] = W_m223[(k_offset + k)*2 + i, r]
                        
        # Verify the full tensor
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        T_full = tensor_from_factors(U_np, V_np, W_np)
        T_target_full = build_target_tensor(n, m, p)
        
        if np.allclose(T_full, T_target_full, atol=1e-10):
            return {"U": U, "V": V, "W": W}
        else:
            return None

    # Try to find a rank 50 solution
    result = try_rank_50()
    if result is not None:
        return result
    
    # Fallback to rank 53
    return get_rank_53()
