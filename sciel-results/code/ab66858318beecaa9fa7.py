import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 9
P = p * n  # 6
SCHOOLBOOK = n * m * p  # 18

def build_tensor():
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose T."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(val - T[i, j, k]) > tol:
                    return False
    return True

def try_rank(r, T, max_iters=5000, seed=42, time_limit=10.0):
    """Try to find a rank-r decomposition using vectorized iterative refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    for attempt in range(5):
        U = rng.randn(N, r)
        V = rng.randn(M, r)
        W = rng.randn(P, r)
        
        best_obj = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Compute error vectorized
            # T_calc[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # We can compute this efficiently
            # But for simplicity and speed in Python loops for small dims, 
            # the previous loop was slow due to Python overhead.
            # Let's use numpy einsum or similar if possible, but the structure is complex.
            # Actually, for n=2, m=3, p=3, the loops are small.
            # The bottleneck was lstsq. Let's vectorize the lstsq calls.
            
            # Instead of per-element lstsq, we can group them.
            # For U update:
            # For each (i,j), we solve for U[i*m+j, :]
            # A[c,k] = V[j*p+k, c] * W[k*n+i, c]
            # b[k] = T[i,j,k]
            # We can stack these into a batch.
            
            # However, the shape of A depends on i,j.
            # Let's just do it in a loop but use numpy for the solve.
            
            obj = 0.0
            for i in range(n):
                for j in range(m):
                    # Compute A and b for this (i,j)
                    # A is r x p
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    # Solve min ||A^T u - b||^2
                    # u = A (A^T A)^{-1} A^T b
                    # Use lstsq
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    b = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    b = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Calculate objective
            obj = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        obj += (val - T[i,j,k])**2
            
            if obj < 1e-12:
                return U, V, W
            
            if obj < best_obj:
                best_obj = obj
    
    return None

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    T = build_tensor()
    
    # Try ranks from low to high
    for r in range(12, SCHOOLBOOK + 1):
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        # Lower ranks are harder, give them more time?
        # Or just a fixed time per rank.
        # Let's give 10s for rank 12, 5s for 13, etc.
        time_alloc = 10.0 if r == 12 else 5.0
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time()
            if time_alloc < 1.0:
                break
        
        result = try_rank(r, T, max_iters=10000, seed=42, time_limit=time_alloc)
        if result is not None:
            U, V, W = result
            # Verify
            if verify_decomposition(U, V, W, T):
                return {"U": [[float(x) for x in U[i]] for i in range(N)],
                        "V": [[float(x) for x in V[i]] for i in range(M)],
                        "W": [[float(x) for x in W[i]] for i in range(P)]}
    
    # Fallback: schoolbook decomposition
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
