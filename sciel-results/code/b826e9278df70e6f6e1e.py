"""Optimized rank search for 2x3x9 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def _build_target_tensor():
    """Build the target matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 if the (i,j) element of A
    and (j,k) element of B produce the (i,k) element of C.
    The tensor is T[i,j,k] = 1 for all valid i,j,k.
    """
    n, m, p = DIMENSIONS
    # The tensor T has shape (n, m, p) with T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
    rank = U.shape[1]
    # Reshape for efficient computation
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # We need: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Vectorized approach
    # For each pair (i,k), we need sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's compute this using einsum or matrix operations
    # U_reshaped: (n, m, rank)
    # V_reshaped: (m, p, rank)  
    # W_reshaped: (p, n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i, j, k): sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a 3-tensor contraction
    
    # Compute M[j, k, i] = sum_c V_r[j,k,c] * W_r[k,i,c] first? No, that's not right.
    # We need: result[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Let's do it step by step:
    # First compute A[j, k, i] = sum_c V_r[j,k,c] * W_r[k,i,c] -- this is a 3D array (m, p, n)
    # Then result[i,j,k] = sum_c U_r[i,j,c] * (A[j,k,i])... wait, that's not quite right either.
    
    # Actually: result[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = sum_c U_r[i,j,c] * (V_r[j,k,c] * W_r[k,i,c])
    
    # Let B[j,k,c] = V_r[j,k,c] * W_r[k,i,c] -- but this depends on i, so can't precompute simply.
    
    # Alternative: use einsum
    # result[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    # Check if all entries are close to 1
    return np.allclose(result, np.ones((n, m, p)), atol=tol, rtol=0)

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster verification using random sampling."""
    rank = U.shape[1]
    
    # Test a subset of (i,j,k) triples
    n_tests = min(100, n * m * p)
    rng = np.random.RandomState(42)
    
    for _ in range(n_tests):
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        
        u_val = U[i * m + j]
        v_val = V[j * p + k]
        w_val = W[k * n + i]
        
        dot = np.dot(u_val, v_val * w_val)
        if abs(dot - 1.0) > tol:
            return False
    return True

def _try_rank(rank, n, m, p, max_iterations=5000, time_limit=12.0):
    """Try to find a decomposition of given rank using iterative optimization."""
    start_time = time.time()
    
    # Initialize with random values
    rng = np.random.RandomState(rank * 7 + 42)
    
    # Start with a structured initialization based on the schoolbook approach
    # but with some perturbation
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Initialize each column
    for c in range(rank):
        # Assign to random (i,j,k) triples
        i = c % n
        j = (c // n) % m
        k = (c // (n * m)) % p
        
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    # Add small random perturbations
    U += rng.randn(n * m, rank) * 0.1
    V += rng.randn(m * p, rank) * 0.1
    W += rng.randn(p * n, rank) * 0.1
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradients using finite differences or analytic
        # For speed, use a simplified gradient step
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Compute current result
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        error = np.sum((result - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if error < 1e-12:
                break
        
        if error < 1e-10:
            break
        
        # Compute gradients analytically
        # dL/dU[i,j,c] = 2 * (result[i,j,k] - 1) * V[j,k,c] * W[k,i,c] summed over k
        # dL/dV[j,k,c] = 2 * (result[i,j,k] - 1) * U[i,j,c] * W[k,i,c] summed over i
        # dL/dW[k,i,c] = 2 * (result[i,j,k] - 1) * U[i,j,c] * V[j,k,c] summed over j
        
        grad_U = np.einsum('ijk,jkc,kic->ijc', result - 1.0, V_r, W_r, optimize=True) * 2
        grad_V = np.einsum('ijk,ijc,kic->jkc', result - 1.0, U_r, W_r, optimize=True) * 2
        grad_W = np.einsum('ijk,ijc,jkc->kic', result - 1.0, U_r, V_r, optimize=True) * 2
        
        U -= lr * grad_U.reshape(n * m, rank)
        V -= lr * grad_V.reshape(m * p, rank)
        W -= lr * grad_W.reshape(p * n, rank)
        
        # Adaptive learning rate
        if it % 100 == 0 and error > best_error * 1.5:
            lr *= 0.5
        elif error < best_error * 0.9:
            lr = min(lr * 1.1, 0.1)
    
    # Final check
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-8):
        return best_U, best_V, best_W, True
    
    return None, None, None, False

def _try_rank_structured(rank, n, m, p, max_iterations=2000, time_limit=8.0):
    """Try structured decomposition with specific patterns."""
    start_time = time.time()
    rng = np.random.RandomState(rank * 13 + 7)
    
    # Try different initialization strategies
    for strategy in range(3):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        if strategy == 0:
            # Cyclic pattern
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
        elif strategy == 1:
            # Random assignment
            for c in range(rank):
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = rng.randint(0, p)
                U[i * m + j, c] = rng.choice([-1, 1])
                V[j * p + k, c] = rng.choice([-1, 1])
                W[k * n + i, c] = rng.choice([-1, 1])
        else:
            # Mixed
            for c in range(rank):
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = rng.randint(0, p)
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                # Add noise to some
                if rng.random() < 0.3:
                    U[i * m + j, c] += rng.randn() * 0.5
                    V[j * p + k, c] += rng.randn() * 0.5
                    W[k * n + i, c] += rng.randn() * 0.5
        
        # Optimize
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        lr = 0.02
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
            error = np.sum((result - 1.0) ** 2)
            
            if error < 1e-12:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    return U, V, W, True
                break
            
            grad_U = np.einsum('ijk,jkc,kic->ijc', result - 1.0, V_r, W_r, optimize=True) * 2
            grad_V = np.einsum('ijk,ijc,kic->jkc', result - 1.0, U_r, W_r, optimize=True) * 2
            grad_W = np.einsum('ijk,ijc,jkc->kic', result - 1.0, U_r, V_r, optimize=True) * 2
            
            U -= lr * grad_U.reshape(n * m, rank)
            V -= lr * grad_V.reshape(m * p, rank)
            W -= lr * grad_W.reshape(p * n, rank)
            
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            if it % 200 == 0:
                lr *= 0.95
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W, True
    
    return None, None, None, False

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    deadline = time.time() + 135  # Leave 15s margin
    
    # Try ranks from 27 up to schoolbook_rank
    # Lower bound for matrix multiplication is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    for rank in range(27, best_rank):
        if time.time() > deadline:
            break
        
        # Try structured approach first (faster)
        U, V, W, success = _try_rank_structured(rank, n, m, p, max_iterations=1000, time_limit=6.0)
        if success:
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            continue
        
        # Try general optimization
        U, V, W, success = _try_rank(rank, n, m, p, max_iterations=3000, time_limit=10.0)
        if success:
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            continue
        
        if time.time() > deadline:
            break
    
    # Convert to lists for output
    if isinstance(best_U, np.ndarray):
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out, V_out, W_out = best_U, best_V, best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
