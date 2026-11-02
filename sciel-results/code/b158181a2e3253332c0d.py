"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill
    
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
    
    # Try to find lower rank decomposition
    np.random.seed(42)
    
    # Strategy: Try ranks from 60 down to some minimum
    # Use gradient descent with smart initialization
    
    for r in range(rank_schoolbook - 1, 50, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_mat = np.random.randn(n*m, r) * 0.1
            V_mat = np.random.randn(m*p, r) * 0.1
            W_mat = np.random.randn(p*n, r) * 0.1
            
            # Gradient descent to minimize error
            lr = 0.001
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute predictions for all entries
                pred = np.zeros((n*m, m*p, p*n))
                for r_idx in range(r):
                    pred += np.einsum('a,b,c->abc', U_mat[:, r_idx], V_mat[:, r_idx], W_mat[:, r_idx])
                
                # Compute error
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
            
            # Verify the solution
            if verify_decomp(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, T):
                best_rank = r
                best_U = U_mat.tolist()
                best_V = V_mat.tolist()
                best_W = W_mat.tolist()
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Final verification of best solution
    if best_rank < rank_schoolbook:
        if verify_decomp(best_U, best_V, best_W, n, m, p, T):
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return schoolbook as fallback
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
