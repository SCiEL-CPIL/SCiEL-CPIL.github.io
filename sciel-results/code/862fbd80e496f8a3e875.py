import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    
    # Build the target Multiplication Tensor T
    # Shape: (n*p, n*m, m*p)
    # T[i*p + k, i*m + j, j*p + k] = 1
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * p + k, i * m + j, j * p + k] = 1.0

    start_time = time.time()
    timeout = 140.0

    # Schoolbook solution as fallback
    U0 = np.zeros((n * m, rank_seed))
    V0 = np.zeros((m * p, rank_seed))
    W0 = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    def verify(U, V, W):
        D = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.sum((D - T) ** 2)
        return err < 1e-6

    def try_rank(R, seed=42):
        """Try to find a rank-R decomposition using structured initialization."""
        np.random.seed(seed)
        
        # Use Kronecker product structure: 3x4x8 = (3x1)x(2x2)x(4x2)
        # We'll initialize using the schoolbook for each factor and combine
        # Factor 1: 3x2x4 (n1=3, m1=2, p1=4)
        # Factor 2: 1x2x2 (n2=1, m2=2, p2=2)
        
        n1, m1, p1 = 3, 2, 4
        n2, m2, p2 = 1, 2, 2
        
        # Schoolbook for factor 1
        U1 = np.zeros((n1*m1, n1*m1*p1))
        V1 = np.zeros((m1*p1, n1*m1*p1))
        W1 = np.zeros((p1*n1, n1*m1*p1))
        c = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i*m1+j, c] = 1
                    V1[j*p1+k, c] = 1
                    W1[k*n1+i, c] = 1
                    c += 1
        
        # Schoolbook for factor 2
        U2 = np.zeros((n2*m2, n2*m2*p2))
        V2 = np.zeros((m2*p2, n2*m2*p2))
        W2 = np.zeros((p2*n2, n2*m2*p2))
        c = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i*m2+j, c] = 1
                    V2[j*p2+k, c] = 1
                    W2[k*n2+i, c] = 1
                    c += 1
        
        # Kronecker product gives rank = 24*4 = 96
        # We want rank R < 96, so we need to find a compressed representation
        
        # For now, just return the schoolbook if R >= 96
        if R >= rank_seed:
            return U0[:R], V0[:R], W0[:R]
        
        # Try random initialization with optimization
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Simple gradient descent
        lr = 0.01
        for it in range(200):
            if time.time() - start_time > timeout - 5:
                break
            D = np.einsum('ar,br,cr->abc', W, U, V)
            diff = D - T
            err = np.sum(diff * diff)
            
            if err < 1e-6:
                return U, V, W
            
            grad_W = 2 * np.einsum('abc,br,cr->ar', diff, U, V)
            grad_U = 2 * np.einsum('abc,ar,cr->br', diff, W, V)
            grad_V = 2 * np.einsum('abc,ar,br->cr', diff, W, U)
            
            W -= lr * grad_W
            U -= lr * grad_U
            V -= lr * grad_V
        
        return U, V, W

    # Try ranks from 49 upward
    best_result = None
    for R in [49, 50, 55, 60, 64, 70, 80, 90, 95, 96]:
        if time.time() - start_time > timeout - 10:
            break
        
        for seed in [42, 123, 999]:
            if time.time() - start_time > timeout - 5:
                break
            
            U, V, W = try_rank(R, seed)
            if verify(U, V, W):
                # Round to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
