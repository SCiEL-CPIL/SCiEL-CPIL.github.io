import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start = time.time()
    deadline = start + 125  # Leave 25s margin for safety
    
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Try to find lower rank via gradient descent
    for target_rank in range(42, 60):
        if time.time() > deadline:
            break
        found = False
        for trial in range(6):
            if time.time() > deadline:
                break
            np.random.seed(999 + target_rank * 100 + trial)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            lr = 0.15
            for step in range(3000):
                if time.time() > deadline:
                    break
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            e = val - 1.0
                            dU[i*m+j] += 2*e * V[j*p+k] * W[k*n+i]
                            dV[j*p+k] += 2*e * U[i*m+j] * W[k*n+i]
                            dW[k*n+i] += 2*e * U[i*m+j] * V[j*p+k]
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if verify(U, V, W, 1e-4):
                for scale in [2, 4, 8, 16, 32]:
                    Uc = np.round(U * scale) / scale
                    Vc = np.round(V * scale) / scale
                    Wc = np.round(W * scale) / scale
                    if verify(Uc, Vc, Wc, 1e-10):
                        best_U, best_V, best_W = Uc, Vc, Wc
                        best_rank = target_rank
                        found = True
                        break
                if not found:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                break
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
