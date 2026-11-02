"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Build the full tensor for verification
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T = build_tensor(n, m, p)
    nm, m_p, p_n = n*m, m*p, p*n
    
    def verify_decomp(U, V, W, n, m, p, T, tol=1e-9):
        """Check if decomposition is exact."""
        nm, m_p, p_n = n*m, m*p, p*n
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Compute predicted tensor
        pred = np.zeros((nm, m_p, p_n))
        for r_idx in range(r):
            pred += np.einsum('a,b,c->abc', U_mat[:, r_idx], V_mat[:, r_idx], W_mat[:, r_idx])
        
        return np.allclose(pred, T, atol=tol)
    
    # Strategy: Try to find a lower rank decomposition using optimization
    # We focus on rank 59 first, as 60 is the schoolbook rank
    np.random.seed(42)
    
    # Try rank 59 with multiple random initializations and gradient descent
    r = rank_schoolbook - 1
    max_attempts = 5
    max_iter = 200
    lr = 0.05
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U_mat = np.random.randn(nm, r) * 0.1
        V_mat = np.random.randn(m_p, r) * 0.1
        W_mat = np.random.randn(p_n, r) * 0.1
        
        # Gradient descent
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute predicted tensor
            pred = np.zeros((nm, m_p, p_n))
            for r_idx in range(r):
                pred += np.einsum('a,b,c->abc', U_mat[:, r_idx], V_mat[:, r_idx], W_mat[:, r_idx])
            
            error = pred - T
            loss = np.sum(error**2)
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U_mat)
            dV = np.zeros_like(V_mat)
            dW = np.zeros_like(W_mat)
            
            for r_idx in range(r):
                dU[:, r_idx] = np.einsum('abc,b,c->a', error, V_mat[:, r_idx], W_mat[:, r_idx])
                dV[:, r_idx] = np.einsum('abc,a,c->b', error, U_mat[:, r_idx], W_mat[:, r_idx])
                dW[:, r_idx] = np.einsum('abc,a,b->c', error, U_mat[:, r_idx], V_mat[:, r_idx])
            
            # Update
            U_mat -= lr * dU
            V_mat -= lr * dV
            W_mat -= lr * dW
            
            # Periodic verification
            if it % 20 == 0 and verify_decomp(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, T):
                best_rank = r
                best_U = U_mat.tolist()
                best_V = V_mat.tolist()
                best_W = W_mat.tolist()
                return {"U": best_U, "V": best_V, "W": best_W}
        
        # Final check for this attempt
        if verify_decomp(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, T):
            best_rank = r
            best_U = U_mat.tolist()
            best_V = V_mat.tolist()
            best_W = W_mat.tolist()
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # If rank 59 didn't work, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
