import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W decompose the matmul tensor exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-target entries are 0
    # Build set of target indices
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                    if abs(val) > tol:
                        return False
    return True

def _construct_rank_18(n, m, p):
    """
    Constructive rank-18 decomposition for 2x3x6 matmul.
    
    Structure: For each (j,k) pair, we create terms that handle both i=0 and i=1.
    
    Key insight: Use the fact that for fixed (j,k), the contributions to C[0,k] and C[1,k]
    both involve B[j,k]. We can structure the rank-1 terms to exploit this.
    
    Let's use 18 terms indexed by (j,k) for j in 0..2, k in 0..5.
    For each (j,k), we need:
      - Contribution to T[a=0*3+j, b=j*6+k, c=k*2+0] = T[j, j*6+k, 2k] = 1
      - Contribution to T[a=1*3+j, b=j*6+k, c=k*2+1] = T[3+j, j*6+k, 2k+1] = 1
    
    A single rank-1 term with U vector having entries at positions j and 3+j,
    V vector having entry at position j*6+k, and W vector having entries at positions 2k and 2k+1.
    
    Let U[j,r]=u0, U[3+j,r]=u1, V[j*6+k,r]=v, W[2k,r]=w0, W[2k+1,r]=w1
    Then:
      T[j, j*6+k, 2k] += u0 * v * w0 = 1
      T[3+j, j*6+k, 2k+1] += u1 * v * w1 = 1
    
    But we also need to ensure no cross-terms! For this to work with a single term per (j,k),
    we need u0*v*w0=1, u1*v*w1=1, AND all other combinations to be zero.
    
    With just one term per (j,k), we can set u0=1, u1=1, v=1, w0=1, w1=1.
    But then T[j, j*6+k, 2k+1] = u0*v*w1 = 1 ≠ 0! That's a problem.
    
    So we need more terms per (j,k) to cancel out the cross terms.
    """
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping: term index t corresponds to (j,k)
    t = 0
    for j in range(m):
        for k in range(p):
            a0 = 0 * m + j  # = j
            a1 = 1 * m + j  # = 3+j
            b = j * p + k
            c0 = k * n + 0  # = 2k
            c1 = k * n + 1  # = 2k+1
            
            # Single term: set all to 1
            U[a0, t] = 1
            U[a1, t] = 1
            V[b, t] = 1
            W[c0, t] = 1
            W[c1, t] = 1
            t += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_36(n, m, p):
    """Schoolbook rank-36 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_with_optimization(n, m, p, r, max_iters=500, lr=0.01):
    """Try to find rank-r decomposition using optimization."""
    if _time_left() < 10:
        return None
    
    # Build target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    triples = list(target.keys())
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, c) in triples:
            current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            error = current - 1.0
            loss += error * error
            
            for rr in range(r):
                dU = error * V[b, rr] * W[c, rr]
                dV = error * U[a, rr] * W[c, rr]
                dW = error * U[a, rr] * V[b, rr]
                grad_U[a, rr] += dU
                grad_V[b, rr] += dV
                grad_W[c, rr] += dW
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 1e-10:
            step = lr / (1.0 + it * 0.01)
            U -= step * grad_U
            V -= step * grad_V
            W -= step * grad_W
        
        if loss < 1e-12:
            break
    
    # Snap to half-integers
    def snap(arr):
        result = np.copy(arr)
        for idx in np.ndindex(result.shape):
            val = result[idx]
            nearest_half = round(val * 2) / 2.0
            if abs(val - nearest_half) < 0.01:
                result[idx] = nearest_half
        return result
    
    U_s = snap(U)
    V_s = snap(V)
    W_s = snap(W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # First, try the constructive rank-18 approach
    # But we showed it doesn't work directly due to cross terms.
    # Let's verify:
    result_18 = _construct_rank_18(n, m, p)
    if _verify(result_18["U"], result_18["V"], result_18["W"], n, m, p, tol=1e-6):
        return result_18
    
    # Try optimization for rank 18
    for attempt in range(5):
        if _time_left() < 30:
            break
        result = _try_rank_with_optimization(n, m, p, 18, max_iters=300, lr=0.02)
        if result is not None:
            return result
    
    # Fallback to schoolbook rank 36
    return _construct_rank_36(n, m, p)
