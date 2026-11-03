import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-8)

def _build_strassen_2x3x8(n, m, p):
    """
    Build a rank-44 decomposition using Strassen for the 2x2x2 blocks.
    
    Structure:
    - j=0: 16 schoolbook terms (2 i-values * 8 k-values)
    - j=1,2: 4 blocks of 2x2x2, each using Strassen (7 terms) = 28 terms
    Total: 16 + 28 = 44
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: j=0, schoolbook for 2x8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 in 4 blocks of 2x2x2 using Strassen
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen for 2x2x2:
        # m1 = (a00+a11)(b00+b11)
        U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1; V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1; V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return U, V, W, rank

def _try_rank_reduction(U_init, V_init, W_init, n, m, p, target_rank, max_iters=8000):
    """
    Try to reduce the rank by finding a rank-target_rank decomposition.
    Uses gradient descent with better initialization from the known solution.
    """
    if time.time() - _START_TIME > _DEADLINE - 5:
        return None
    
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Build target tensor
    T_true = np.zeros((a_dim, b_dim, c_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    # Initialize by projecting the known rank-44 solution onto rank-target_rank
    np.random.seed(42)
    r = target_rank
    r_init = len(U_init[0])
    
    U_arr = np.array(U_init, dtype=float)
    V_arr = np.array(V_init, dtype=float)
    W_arr = np.array(W_init, dtype=float)
    
    # Create random mixing matrices to project from r_init to r
    P_U = np.random.randn(r_init, r)
    P_V = np.random.randn(r_init, r)
    P_W = np.random.randn(r_init, r)
    
    U = U_arr @ P_U
    V = V_arr @ P_V
    W = W_arr @ P_W
    
    # Normalize
    for t in range(r):
        nu = np.linalg.norm(U[:, t])
        nv = np.linalg.norm(V[:, t])
        nw = np.linalg.norm(W[:, t])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, t] /= nu
            V[:, t] /= nv
            W[:, t] /= nw
    
    lr = 0.005
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Compute T_decomp = sum_t U[:,t] * V[:,t] * W[:,t]
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Loss: ||T_decomp - T_true||^2
        loss = np.sum((T_decomp - T_true) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Gradients
        diff = T_decomp - T_true
        grad_U = 2 * np.einsum('abc,bt,ct->at', diff, V, W)
        grad_V = 2 * np.einsum('abc,at,ct->bt', diff, U, W)
        grad_W = 2 * np.einsum('abc,at,bt->ct', diff, U, V)
        
        # Update with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        if it % 1000 == 999:
            lr *= 0.9
        
        # Normalize columns to prevent blowup
        for t in range(r):
            nu = np.linalg.norm(U[:, t])
            nv = np.linalg.norm(V[:, t])
            nw = np.linalg.norm(W[:, t])
            if nu > 1e-10:
                U[:, t] /= nu
            if nv > 1e-10:
                V[:, t] /= nv
            if nw > 1e-10:
                W[:, t] /= nw
    
    # Check if we found a valid solution
    if best_loss < 1e-6:
        U_list = [[round(float(x), 6) for x in row] for row in best_U]
        V_list = [[round(float(x), 6) for x in row] for row in best_V]
        W_list = [[round(float(x), 6) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list, r
    
    return None

def construct():
    n, m, p = 2, 3, 8
    
    # Start with the rank-44 Strassen-based construction
    U, V, W, rank = _build_strassen_2x3x8(n, m, p)
    
    # Try to reduce rank if time permits
    if time.time() - _START_TIME < _DEADLINE - 10:
        for target_rank in [43, 42, 41]:
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
            result = _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iters=5000)
            if result is not None:
                U, V, W, rank = result
                break
    
    return {"U": U, "V": V, "W": W}
