"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import random
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != r:
            return False
    for row in V:
        if len(row) != r:
            return False
    for row in W:
        if len(row) != r:
            return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _schoolbook():
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

def _smooth_optimize(target_rank, n, m, p, max_iter=2000, lr=0.01, alpha=50.0, timeout=30):
    """
    Smooth optimization for finding rank-r decomposition of all-ones tensor.
    Uses log-sum-exp based soft penalty for |s - 1|.
    """
    start_time = time.time()
    
    # Initialize with random small values
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Precompute index arrays for speed
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i * m + j, j * p + k, k * n + i))
    
    for it in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        # Compute gradients via finite differences or analytical
        # s[i,j,k] = dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        # dL/dU = sum over (j,k) of (s-1) * V[j*p+k] * W[k*n+i]
        # dL/dV = sum over (i,k) of (s-1) * U[i*m+j] * W[k*n+i]
        # dL/dW = sum over (i,j) of (s-1) * U[i*m+j] * V[j*p+k]
        
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    s = np.dot(ui, vj * wk)
                    err = s - 1.0
                    # Soft gradient: use sigmoid-like weighting
                    # For exactness, use linear gradient
                    gU[i * m + j] += err * (vj * wk)
                    gV[j * p + k] += err * (ui * wk)
                    gW[k * n + i] += err * (ui * vj)
        
        # Update
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Check convergence
        if it % 100 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        max_err = max(max_err, abs(s - 1.0))
            if max_err < 1e-10:
                # Snap to nearest integers/half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    return None

def _try_structured_rank(target_rank, n, m, p, timeout=20):
    """
    Try structured initializations for given rank.
    """
    start_time = time.time()
    
    # Pattern 1: Cyclic structure
    for trial in range(50):
        if time.time() - start_time > timeout:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(target_rank):
            # Assign indices based on c
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            # Random signs for exploration
            sign_u = random.choice([-1, 1]) if random.random() < 0.3 else 1
            sign_v = random.choice([-1, 1]) if random.random() < 0.3 else 1
            sign_w = random.choice([-1, 1]) if random.random() < 0.3 else 1
            
            U[i * m + j, c] = sign_u
            V[j * p + k, c] = sign_v
            W[k * n + i, c] = sign_w
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    deadline = time.time() + 140  # 10s safety margin for 150s limit
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook baseline (guaranteed)
    best = _schoolbook()
    best_rank = n * m * p  # 75
    
    # 2. Try smooth optimization for various ranks
    for target_rank in range(25, 75):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 10
        timeout = min(remaining / 3, 15)
        
        res = _smooth_optimize(target_rank, n, m, p, timeout=timeout)
        if res:
            return res
    
    # 3. Try structured search
    if time.time() < deadline - 5:
        for target_rank in range(25, 75, 3):
            if time.time() > deadline - 5:
                break
            res = _try_structured_rank(target_rank, n, m, p, timeout=5)
            if res:
                return res
    
    return best
# EVOLVE-BLOCK-END
