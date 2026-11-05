"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def make_output(U, V, W):
        r = U.shape[1]
        return {
            "U": [[U[i,c] for c in range(r)] for i in range(nm)],
            "V": [[V[i,c] for c in range(r)] for i in range(mp)],
            "W": [[W[i,c] for c in range(r)] for i in range(pn)]
        }
    
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(n):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+l, :])
                        if abs(actual - expected) > tol:
                            return False
        return True
    
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Try a few ranks with limited iterations
    for rank in [45, 48, 50]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(3000 + rank * 10 + attempt)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            lr = 0.003
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for l in range(n):
                                expected = 1.0 if (i == l and j == k) else 0.0
                                actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+l, :])
                                err = actual - expected
                                dU[i*m+j, :] += 2 * err * (V[j*p+k, :] * W[k*n+l, :])
                                dV[j*p+k, :] += 2 * err * (U[i*m+j, :] * W[k*n+l, :])
                                dW[k*n+l, :] += 2 * err * (U[i*m+j, :] * V[j*p+k, :])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for l in range(n):
                            expected = 1.0 if (i == l and j == k) else 0.0
                            actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+l, :])
                            max_err = max(max_err, abs(actual - expected))
            
            if max_err < 1e-4:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return make_output(U_r, V_r, W_r)
    
    return make_output(U_sb, V_sb, W_sb)
# EVOLVE-BLOCK-END
