"""Matrix multiplication 3x3x6 rank-minimal decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_exceeded():
    return _elapsed() > _TIME_LIMIT

def construct():
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    # We'll try to find a rank-45 decomposition using a structured approach.
    # The key insight: for 3x3x6, we can view this as two 3x3x3 multiplications
    # but with shared structure. Let's try a direct algebraic construction.
    
    # Strategy: Use a rank-r decomposition where r is minimized.
    # Lower bound: max(9, 18, 18) = 18. Known best is around 45-46.
    
    # Let's try a numerical approach with better initialization.
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in [0,n)x[0,m)x[0,p).
    
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Use gradient descent with momentum and adaptive learning rate.
    
    def _verify(U, V, W, tol=1e-7):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - 1.0) > tol:
                        return False
        return True
    
    def _try_decomposition(r, seed=42, max_iters=50000, lr_init=0.1, momentum=0.9):
        """Try to find rank-r decomposition using gradient descent with momentum."""
        if time.time() - t0 > _TIME_LIMIT - 5:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() - t0 > _TIME_LIMIT - 5:
                return None
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r2 in range(r):
                            val += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                        err = val - 1.0
                        for r2 in range(r):
                            dU[i*m+j, r2] += err * V[j*p+k, r2] * W[k*n+i, r2]
                            dV[j*p+k, r2] += err * U[i*m+j, r2] * W[k*n+i, r2]
                            dW[k*n+i, r2] += err * U[i*m+j, r2] * V[j*p+k, r2]
            
            # Update with momentum
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            U += vU
            V += vV
            W += vW
            
            # Adaptive learning rate
            if it % 1000 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r2 in range(r):
                                val += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-6:
                    # Try rounding to half-integers
                    Ur = np.round(U * 2) / 2
                    Vr = np.round(V * 2) / 2
                    Wr = np.round(W * 2) / 2
                    if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), tol=1e-8):
                        return (Ur.tolist(), Vr.tolist(), Wr.tolist())
                # Decay learning rate slowly
                lr *= 0.95
        
        return None
    
    # Start with schoolbook rank as fallback
    rank_schoolbook = n * m * p  # 54
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Start from rank 27 (theoretical lower bound) and go up
    for r in range(27, rank_schoolbook):
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
        
        for seed in range(10):
            if time.time() - t0 > _TIME_LIMIT - 10:
                break
            
            result = _try_decomposition(r, seed=seed + r * 1000, max_iters=20000)
            if result is not None:
                Ur, Vr, Wr = result
                if _verify(Ur, Vr, Wr, tol=1e-8):
                    best_result = {"U": Ur, "V": Vr, "W": Wr}
                    best_rank = r
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
