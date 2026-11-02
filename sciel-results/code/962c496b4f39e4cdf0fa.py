"""
Rank-minimal tensor decomposition for 2x2x9 matrix multiplication.
Strategy: Use 4 Strassen blocks (rank 7 each) for 8 columns + schoolbook for 1 column = rank 32.
Then attempt to find a rank-31 decomposition by checking if any column is redundant
or can be absorbed via local optimization.
"""
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank32():
    """
    Build the known rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 schoolbook (col 8).
    Returns (U, V, W) as lists of lists of floats.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b_{0,k0}+b_{1,k1})
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2: (a10+a11) * b_{0,k0}
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3: a00 * (b_{0,k1} - b_{1,k1})
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4: a11 * (b_{1,k0} - b_{0,k0})
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5: (a00+a01) * b_{1,k1}
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6: (a10-a00) * (b_{0,k0}+b_{0,k1})
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7: (a01-a11) * (b_{1,k0}+b_{1,k1})
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    return U, V, W

def _try_rank31_local_search(U0, V0, W0, deadline):
    """
    Try to find a rank-31 decomposition by dropping one column and adjusting
    the remaining columns to absorb the residual.
    """
    n, m, p = 2, 2, 9
    U_arr = np.array(U0, dtype=float)
    V_arr = np.array(V0, dtype=float)
    W_arr = np.array(W0, dtype=float)
    
    for drop_col in range(32):
        if time.time() > deadline:
            break
        
        U_t = np.delete(U_arr, drop_col, axis=1)
        V_t = np.delete(V_arr, drop_col, axis=1)
        W_t = np.delete(W_arr, drop_col, axis=1)
        
        # Compute the residual: what's missing from the tensor
        # Residual[i,j,k] = 1 - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        residual = np.ones((n, m, p))
        for r in range(31):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i,j,k] -= U_t[i*m+j, r] * V_t[j*p+k, r] * W_t[k*n+i, r]
        
        max_abs = np.max(np.abs(residual))
        if max_abs < 1e-9:
            # The dropped column was redundant!
            return U_t.tolist(), V_t.tolist(), W_t.tolist()
        
        # Try to repair: adjust one of the remaining columns to absorb the residual.
        # This is a nonlinear problem, so we use a simple gradient descent approach.
        # We'll try adjusting each of the 31 columns in turn.
        for adjust_col in range(31):
            if time.time() > deadline:
                break
            
            U_adj = U_t.copy()
            V_adj = V_t.copy()
            W_adj = W_t.copy()
            
            # Initialize with a small perturbation
            U_adj[:, adjust_col] += 0.01 * np.random.randn(4)
            V_adj[:, adjust_col] += 0.01 * np.random.randn(18)
            W_adj[:, adjust_col] += 0.01 * np.random.randn(18)
            
            # Gradient descent on the squared error
            lr = 0.001
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute current error
                err = np.zeros((n, m, p))
                for r in range(31):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                err[i,j,k] += U_adj[i*m+j, r] * V_adj[j*p+k, r] * W_adj[k*n+i, r]
                err -= 1.0
                loss = np.sum(err**2)
                
                if loss < 1e-12:
                    return U_adj.tolist(), V_adj.tolist(), W_adj.tolist()
                
                # Compute gradients (approximate)
                eps = 1e-7
                grad_U = np.zeros_like(U_adj)
                grad_V = np.zeros_like(V_adj)
                grad_W = np.zeros_like(W_adj)
                
                # Only compute gradients for the adjusted column to save time
                for idx in range(4):
                    U_adj[idx, adjust_col] += eps
                    err_plus = np.zeros((n, m, p))
                    for r in range(31):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    err_plus[i,j,k] += U_adj[i*m+j, r] * V_adj[j*p+k, r] * W_adj[k*n+i, r]
                    err_plus -= 1.0
                    loss_plus = np.sum(err_plus**2)
                    grad_U[idx, adjust_col] = (loss_plus - loss) / eps
                    U_adj[idx, adjust_col] -= eps
                
                for idx in range(18):
                    V_adj[idx, adjust_col] += eps
                    err_plus = np.zeros((n, m, p))
                    for r in range(31):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    err_plus[i,j,k] += U_adj[i*m+j, r] * V_adj[j*p+k, r] * W_adj[k*n+i, r]
                    err_plus -= 1.0
                    loss_plus = np.sum(err_plus**2)
                    grad_V[idx, adjust_col] = (loss_plus - loss) / eps
                    V_adj[idx, adjust_col] -= eps
                
                for idx in range(18):
                    W_adj[idx, adjust_col] += eps
                    err_plus = np.zeros((n, m, p))
                    for r in range(31):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    err_plus[i,j,k] += U_adj[i*m+j, r] * V_adj[j*p+k, r] * W_adj[k*n+i, r]
                    err_plus -= 1.0
                    loss_plus = np.sum(err_plus**2)
                    grad_W[idx, adjust_col] = (loss_plus - loss) / eps
                    W_adj[idx, adjust_col] -= eps
                
                # Update
                U_adj[:, adjust_col] -= lr * grad_U[:, adjust_col]
                V_adj[:, adjust_col] -= lr * grad_V[:, adjust_col]
                W_adj[:, adjust_col] -= lr * grad_W[:, adjust_col]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with the known rank-32 solution
    U, V, W = _build_rank32()
    
    if not _verify(U, V, W, 2, 2, 9):
        # Fallback to full schoolbook rank 36
        rank_fb = 36
        U_fb = [[0.0] * rank_fb for _ in range(4)]
        V_fb = [[0.0] * rank_fb for _ in range(18)]
        W_fb = [[0.0] * rank_fb for _ in range(18)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][c] = 1.0
                    V_fb[j*9+k][c] = 1.0
                    W_fb[k*2+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 32
    best_U, best_V, best_W = U, V, W
    
    # Try to find rank-31 decomposition via local search
    if time.time() < deadline - 20:
        result = _try_rank31_local_search(U, V, W, deadline - 20)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 31
    
    return {"U": best_U, "V": best_V, "W": best_W}
