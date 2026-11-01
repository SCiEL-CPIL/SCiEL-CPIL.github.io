import time
import numpy as np
from itertools import product

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

def tensor_residual(U, V, W, T):
    """Compute residual efficiently."""
    rank = U.shape[1]
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                total += (s - T[i,j,k]) ** 2
    return total

def is_half_integer(arr, tol=1e-8):
    """Check if all entries are integer or half-integer."""
    flat = arr.flatten()
    v = flat * 2
    return np.all(np.abs(v - np.round(v)) < tol)

def round_to_half(arr):
    return np.round(arr * 2) / 2

def verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that U, V, W exactly decompose T with half-integer entries."""
    if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
        return False
    res = tensor_residual(U, V, W, T)
    return res < tol

def schoolbook_decomposition():
    rank = SCHOOLBOOK
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def try_rank_structured(r, T, max_iters=2000, seed=42, time_limit=8.0):
    """Try to find a rank-r decomposition using structured initialization and refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    for attempt in range(12):
        if time.time() - start_time > time_limit:
            break
            
        # Structured initialization
        if attempt == 0:
            # Cyclic assignment
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        c = c % r
                        U[i*m+j, c] += 1.0
                        V[j*p+k, c] += 1.0
                        W[k*n+i, c] += 1.0
        elif attempt == 1:
            # Sparse initialization
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        c = c % r
                        U[i*m+j, c] = 1.0
                        V[j*p+k, c] = 1.0
                        W[k*n+i, c] = 1.0
        elif attempt == 2:
            # Block structure: try to group by i
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for i in range(n):
                base = i * (r // n)
                for j in range(m):
                    for k in range(p):
                        c = base + (j * p + k) % (r // n)
                        U[i*m+j, c] = 1.0
                        V[j*p+k, c] = 1.0
                        W[k*n+i, c] = 1.0
        elif attempt == 3:
            # Random half-integer initialization
            np.random.seed(seed + attempt * 100)
            U = np.random.randint(-2, 3, size=(N, r)) / 2.0
            V = np.random.randint(-2, 3, size=(M, r)) / 2.0
            W = np.random.randint(-2, 3, size=(P, r)) / 2.0
        else:
            # Random initialization with scaling
            U = rng.randn(N, r) * (1.0 / np.sqrt(r))
            V = rng.randn(M, r) * (1.0 / np.sqrt(r))
            W = rng.randn(P, r) * (1.0 / np.sqrt(r))
        
        best_obj = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Alternating least squares
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
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
            
            # Update W
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
            obj = tensor_residual(U, V, W, T)
            
            if obj < 1e-14:
                # Check if half-integer
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return U, V, W
                else:
                    # Try rounding
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    if tensor_residual(U_r, V_r, W_r, T) < 1e-8:
                        return U_r, V_r, W_r
            
            if obj < best_obj:
                best_obj = obj
    
    return None

def try_rank_milp(r, T, time_limit=5.0):
    """Try to find a rank-r decomposition using MILP-like approach with scipy.optimize.milp."""
    start_time = time.time()
    
    # This is a simplified approach - we'll use a grid search over small integer/half-integer values
    # For very small ranks, we can enumerate support patterns
    
    if r > 15:
        return None  # Too large for MILP-style enumeration
    
    # Try a few structured patterns with small integer values
    values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # For rank <= 10, try some sparse patterns
    for pattern_idx in range(20):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(12345 + pattern_idx * 7 + r)
        
        U = np.zeros((N, r))
        V = np.zeros((M, r))
        W = np.zeros((P, r))
        
        # Random sparse support
        for col in range(r):
            # Choose which entries are non-zero
            u_mask = np.random.random(N) < 0.3
            v_mask = np.random.random(M) < 0.3
            w_mask = np.random.random(P) < 0.3
            
            for idx in range(N):
                if u_mask[idx]:
                    U[idx, col] = np.random.choice(values)
            for idx in range(M):
                if v_mask[idx]:
                    V[idx, col] = np.random.choice(values)
            for idx in range(P):
                if w_mask[idx]:
                    W[idx, col] = np.random.choice(values)
        
        # Refine with alternating least squares
        for it in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    b = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
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
            
            # Update W
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
            
            obj = tensor_residual(U, V, W, T)
            if obj < 1e-14:
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return U, V, W
                else:
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    if tensor_residual(U_r, V_r, W_r, T) < 1e-8:
                        return U_r, V_r, W_r
    
    return None

def construct():
    start_time = time.time()
    deadline = 130  # Leave 20s safety margin
    
    T = build_tensor()
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = SCHOOLBOOK
    
    # Try ranks from low to high
    for r in range(10, SCHOOLBOOK):
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        time_alloc = 6.0 if r <= 13 else 3.0
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time()
            if time_alloc < 1.0:
                break
        
        # Try structured ALS
        result = try_rank_structured(r, T, max_iters=3000, seed=42, time_limit=time_alloc * 0.6)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, T):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found a valid lower rank
        
        # Try MILP-style search if time permits
        remaining = deadline - time.time()
        if remaining > 2.0:
            result = try_rank_milp(r, T, time_limit=min(remaining * 0.4, 4.0))
            if result is not None:
                U, V, W = result
                if verify_decomposition(U, V, W, T):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
    
    return best
