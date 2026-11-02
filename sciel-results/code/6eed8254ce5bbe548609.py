import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    def validate(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    if j*p <= b < (j+1)*p:
                        k = b - j*p
                        if c == k*n + i:
                            continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    # Try to find rank 29 decomposition via gradient descent
    if time.time() < deadline:
        for seed in range(200):
            if time.time() > deadline:
                break
            np.random.seed(seed)
            r = 29
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            lr = 0.05
            for step in range(500):
                if time.time() > deadline:
                    break
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a, b, c = i*m+j, j*p+k, k*n+i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err = val - 1.0
                            dU[a, :] += 2 * err * V[b, :] * W[c, :]
                            dV[b, :] += 2 * err * U[a, :] * W[c, :]
                            dW[c, :] += 2 * err * U[a, :] * V[b, :]
                
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            i, j = divmod(a, m)
                            if j*p <= b < (j+1)*p:
                                k = b - j*p
                                if c == k*n + i:
                                    continue
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            dU[a, :] += 2 * val * V[b, :] * W[c, :]
                            dV[b, :] += 2 * val * U[a, :] * W[c, :]
                            dW[c, :] += 2 * val * U[a, :] * V[b, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if step % 50 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a, b, c = i*m+j, j*p+k, k*n+i
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-8:
                        break
            
            if validate(U, V, W, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
