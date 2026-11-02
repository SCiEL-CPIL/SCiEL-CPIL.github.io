"""Matrix multiplication 3,3,8 - seeking rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W form a valid tensor decomposition."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with the multiplication tensor T[i,j,k] = 1 iff j == j (always 1 for valid indices)
    # Actually the tensor is T[i,j,k] = delta_{j,j} which is just 1 for all valid i,j,k
    # Wait, the multiplication tensor for A(3x3) * B(3x8) = C(3x8)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i, j, k
    # Because every term A[i,j]*B[j,k] contributes to C[i,k]
    T_from_decomp = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        # Uc has shape (n*m,), Vc has shape (m*p,), Wc has shape (p*n,)
        # Reshape to 3D
        Uc_3d = Uc.reshape(n, m)
        Vc_3d = Vc.reshape(m, p)
        Wc_3d = Wc.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_from_decomp[i, j, k] += Uc_3d[i, j] * Vc_3d[j, k] * Wc_3d[k, i]
    T_target = np.ones((n, m, p))
    return np.allclose(T_from_decomp, T_target, atol=tol)

def construct():
    start = time.time()
    deadline = start + 130  # 130s deadline, 20s safety margin
    
    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = 9 * 3 * 8  # 216? No, n*m*p = 3*3*8 = 72
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Let's try ranks from 24 up to 71
    
    for rank in range(24, 72):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(n * m, rank) if 'n' in dir() else np.random.randn(9, rank)
            V = np.random.randn(m * p, rank) if 'm' in dir() else np.random.randn(24, rank)
            W = np.random.randn(p * n, rank) if 'p' in dir() else np.random.randn(24, rank)
            
            n, m, p = DIMENSIONS
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Simple local search / gradient descent
            lr = 0.01
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute error
                T_decomp = np.zeros((n, m, p))
                for c in range(rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_decomp[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
                
                err = T_decomp - np.ones((n, m, p))
                if np.max(np.abs(err)) < 1e-6:
                    # Found a solution!
                    best = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = rank
                    break
                
                # Compute gradients (approximate)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for c in range(rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                dU[i * m + j, c] += err[i, j, k] * Vc[j, k] * Wc[k, i]
                                dV[j * p + k, c] += err[i, j, k] * Uc[i, j] * Wc[k, i]
                                dW[k * n + i, c] += err[i, j, k] * Uc[i, j] * Vc[j, k]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if best_rank < 72 and time.time() > deadline:
                break
        if best_rank < 72:
            break
    
    if best_rank < 72:
        return best
    return _build_schoolbook()
