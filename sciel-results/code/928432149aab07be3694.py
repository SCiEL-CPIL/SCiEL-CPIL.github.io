"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# Deadline guard: kill before 150s, return best found with margin
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 3.0

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U,V,W exactly decompose matmul tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Expected: T[i,j,k] = 1 if consistent with matmul (i.e., the matmul tensor has
    # T[i,j,k]=1 for all i,j,k in the matmul sense). Actually the matmul tensor is
    # T[i,j,k] = delta (the standard tensor) where (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    # Wait: the matmul tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because (AB)[i,k] = sum_j A[i,j] B[j,k], so the structure tensor has T[i,j,k]=1.
    # Let's verify: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] should equal 1 for all i,j,k.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using numpy broadcasting."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0))

def _try_rank(r, n, m, p, max_iter=5000, lr=0.01, seed=42):
    """Try to find a rank-r decomposition using multiplicative Weiszfeld / ALS style."""
    if _time_left() < 10:
        return None
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U = np_rng.randn(n*m, r) * 0.1
    V = np_rng.randn(m*p, r) * 0.1
    W = np_rng.randn(p*n, r) * 0.1
    
    # Use ALS: fix two, solve for third
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _time_left() < 5:
            break
        # Update U: for each (i,j), U[i*m+j,:] = argmin sum_k (T[i,j,k] - sum_c V[j*p+k,c]W[k*n+i,c]U[i*m+j,c])
        # Actually this is a linear system. Let's do gradient descent instead for simplicity.
        
        # Compute gradient via finite differences is too slow. Use analytic gradient.
        # L = sum_{i,j,k} (1 - sum_c U[i,j,c]V[j,k,c]W[k,i,c])^2
        # dL/dU[i,j,c] = -2 * sum_k (1 - ...) * V[j,k,c] * W[k,i,c]
        
        Ua = U.reshape(n, m, r)
        Va = V.reshape(m, p, r)
        Wa = W.reshape(p, n, r)
        
        # Compute residuals
        T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
        R = 1.0 - T  # residual
        
        # Gradient for U
        # dU[i,j,c] = -2 * sum_k R[i,j,k] * Va[j,k,c] * Wa[k,i,c]
        dU = -2.0 * np.einsum('ijk,jkc,kic->ijc', R, Va, Wa)
        # Gradient for V
        dV = -2.0 * np.einsum('ijk,ijc,kic->jkc', R, Ua, Wa)
        # Gradient for W
        dW = -2.0 * np.einsum('ijk,ijc,jkc->kic', R, Ua, Va)
        
        # Adaptive learning rate
        err = np.max(np.abs(R))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        
        if err < 1e-8:
            break
        
        # Scale learning rate
        scale = 1.0 / (1.0 + 10*err)
        U -= lr * scale * dU.reshape(n*m, r)
        V -= lr * scale * dV.reshape(m*p, r)
        W -= lr * scale * dW.reshape(p*n, r)
    
    if best_err < tol:
        # Convert back
        U_list = best_U.reshape(n*m, r).tolist()
        V_list = best_V.reshape(m*p, r).tolist()
        W_list = best_W.reshape(p*n, r).tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _try_rank_structured(r, n, m, p, max_iter=3000, seed=42):
    """Try structured initialization: start from schoolbook and zero out some columns."""
    if _time_left() < 10:
        return None
    if r > n*m*p:
        return None
    rng = random.Random(seed)
    
    # Start from schoolbook, then try to merge/remove columns
    U0 = np.zeros((n*m, n*m*p))
    V0 = np.zeros((m*p, n*m*p))
    W0 = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    # Randomly select r columns to keep, then optimize
    cols = rng.sample(range(n*m*p), r)
    U = U0[:, cols].copy()
    V = V0[:, cols].copy()
    W = W0[:, cols].copy()
    
    # Optimize with gradient descent
    lr = 0.01
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _time_left() < 5:
            break
        Ua = U.reshape(n, m, r)
        Va = V.reshape(m, p, r)
        Wa = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
        R = 1.0 - T
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        
        if err < 1e-8:
            break
        
        scale = 1.0 / (1.0 + 10*err)
        dU = -2.0 * np.einsum('ijk,jkc,kic->ijc', R, Va, Wa)
        dV = -2.0 * np.einsum('ijk,ijc,kic->jkc', R, Ua, Wa)
        dW = -2.0 * np.einsum('ijk,ijc,jkc->kic', R, Ua, Va)
        
        U -= lr * scale * dU.reshape(n*m, r)
        V -= lr * scale * dV.reshape(m*p, r)
        W -= lr * scale * dW.reshape(p*n, r)
    
    if best_err < 1e-6:
        return {"U": best_U.reshape(n*m, r).tolist(),
                "V": best_V.reshape(m*p, r).tolist(),
                "W": best_W.reshape(p*n, r).tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _construct_schoolbook()
    best_rank = n * m * p  # 120
    
    # Try decreasing ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Also rank >= ceil(n*m*p / max(n,m,p)) = ceil(120/6) = 20
    # Realistic target: try from 60 down to 30
    
    if _time_left() < 20:
        return best
    
    # Phase 1: Try random initializations for moderate ranks
    for r in range(80, 29, -1):
        if _time_left() < 15:
            break
        found = False
        for seed in range(5):
            if _time_left() < 5:
                break
            result = _try_rank(r, n, m, p, max_iter=2000, lr=0.05, seed=seed)
            if result is not None:
                best = result
                best_rank = r
                found = True
                break
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find rank r, try r+1, r+2... up to best_rank
            for r2 in range(r+1, best_rank):
                if _time_left() < 5:
                    break
                for seed in range(3):
                    if _time_left() < 3:
                        break
                    result = _try_rank(r2, n, m, p, max_iter=1000, lr=0.05, seed=seed)
                    if result is not None:
                        best = result
                        best_rank = r2
                        break
                if best_rank == r2:
                    break
    
    # Phase 2: Try structured approach for lower ranks
    if _time_left() > 30:
        for r in range(40, 29, -1):
            if _time_left() < 10:
                break
            for seed in range(3):
                if _time_left() < 5:
                    break
                result = _try_rank_structured(r, n, m, p, max_iter=2000, seed=seed)
                if result is not None and r < best_rank:
                    best = result
                    best_rank = r
                    break
    
    return best

# Ensure the schoolbook solution is always available as fallback
if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
