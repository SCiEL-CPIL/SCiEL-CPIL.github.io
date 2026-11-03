import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to be safe (150s limit)
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_half_int(U_int, V_int, W_int):
        """Verify that U_int, V_int, W_int (scaled by 2) give exact decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U_int[0])
        
        # Check all entries
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                for c in range(p*n):
                    k1 = c // n
                    i2 = c % n
                    expected = 1 if (i1 == i2 and j1 == j2 and k1 == k2) else 0
                    s = 0
                    for r in range(rank):
                        s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                    if s != 4 * expected:  # 2*2*2 = 8 for 1, 0 for 0; but we use half-integers so 2*(1/2)=1, so product of three half-integers scaled by 2 gives 8*expected
                        # Actually if U,V,W are scaled by 2, then actual values are U/2, V/2, W/2
                        # So product is (U/2)*(V/2)*(W/2) = U*V*W/8
                        # We want this to equal expected, so U*V*W = 8*expected
                        if s != 8 * expected:
                            return False
        return True
    
    def rationalize_to_half_int(M, tol=1e-4):
        """Convert numpy array to list of lists of integers (scaled by 2) if close to half-integers."""
        rows, cols = M.shape
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                k_float = val * 2
                k_int = round(k_float)
                if abs(k_float - k_int) < tol:
                    result[i][j] = k_int
                else:
                    return None
        return result
    
    def gradient_descent(T, R, max_iters=200, tol=1e-10, lr_schedule=None):
        """Gradient descent on U, V, W to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2"""
        np.random.seed(42)
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute gradient with respect to U, V, W
            # Loss = ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            # dLoss/dU[a,r] = -2 * sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] + 2 * sum_{b,c} (sum_s U[a,s]*V[b,s]*W[c,s]) * V[b,r] * W[c,r]
            
            # Compute T_calc
            T_calc = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.linalg.norm(T_calc - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                return U, V, W, err
            
            # Learning rate schedule
            if lr_schedule:
                lr = lr_schedule(it)
            else:
                lr = 0.01 / (1 + it * 0.01)
            
            # Gradient for U
            dU = np.zeros_like(U)
            for a in range(n*m):
                for r in range(R):
                    # d/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - T_calc[a,b,c]) * V[b,r] * W[c,r]
                    grad = 0.0
                    for b in range(m*p):
                        for c in range(p*n):
                            grad += (T[a, b, c] - T_calc[a, b, c]) * V[b, r] * W[c, r]
                    dU[a, r] = -2 * grad
            
            # Gradient for V
            dV = np.zeros_like(V)
            for b in range(m*p):
                for r in range(R):
                    grad = 0.0
                    for a in range(n*m):
                        for c in range(p*n):
                            grad += (T[a, b, c] - T_calc[a, b, c]) * U[a, r] * W[c, r]
                    dV[b, r] = -2 * grad
            
            # Gradient for W
            dW = np.zeros_like(W)
            for c in range(p*n):
                for r in range(R):
                    grad = 0.0
                    for a in range(n*m):
                        for b in range(m*p):
                            grad += (T[a, b, c] - T_calc[a, b, c]) * U[a, r] * V[b, r]
                    dW[c, r] = -2 * grad
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return best_U, best_V, best_W, best_err
    
    # Search for minimal R
    best_result = schoolbook()
    
    # Try ranks from 35 down to 20
    for R in range(35, 19, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 7919 + R * 104729)
            
            # Initialize with small random values
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Use gradient descent with adaptive learning rate
            max_iters = 500 if R <= 30 else 300
            U, V, W, err = gradient_descent(T, R, max_iters=max_iters, tol=1e-8)
            
            if err < 1e-3:
                # Try to rationalize to half-integers
                U_int = rationalize_to_half_int(U, tol=1e-3)
                V_int = rationalize_to_half_int(V, tol=1e-3)
                W_int = rationalize_to_half_int(W, tol=1e-3)
                
                if U_int is not None and V_int is not None and W_int is not None:
                    if verify_half_int(U_int, V_int, W_int):
                        # Success - convert back to actual values
                        U_out = [[x / 2.0 for x in row] for row in U_int]
                        V_out = [[x / 2.0 for x in row] for row in V_int]
                        W_out = [[x / 2.0 for x in row] for row in W_int]
                        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    return best_result
