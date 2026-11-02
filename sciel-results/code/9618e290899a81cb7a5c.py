"""Matrix multiplication 3x4x6 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 120.0
START_TIME = time.time()

def check_deadline():
    return (time.time() - START_TIME) > (DEADLINE - 5.0)

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if this is the "j-th term" for output (i,k), else 0
    # Actually T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} ... no
    # The tensor for matmul: T[i,j,k] = 1 for all (i,j,k) in the standard decomposition
    # Wait, let me reconsider.
    
    # Standard: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    rank = U.shape[1]
    # Reshape and check
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build the tensor from decomposition
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] contributes
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Let's check a few random indices for speed
    indices = list(product(range(n), range(m), range(p)))
    random.shuffle(indices)
    check_count = min(50, len(indices))
    
    for i, j, k in indices[:check_count]:
        val = 0.0
        for r in range(rank):
            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if abs(val - 1.0) > tol:
            return False
    return True

def verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomp(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def random_decomp(n, m, p, rank, seed=None):
    """Generate a random decomposition candidate."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Initialize with small random values
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    return U, V, W

def gradient_free_search(n, m, p, max_rank, max_iters=1000, seed=None):
    """Try to find a low-rank decomposition using random search."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    best_rank = max_rank
    best_U, best_V, best_W = None, None, None
    
    # Start with schoolbook
    U, V, W = schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to reduce rank
    for target_rank in range(best_rank - 1, max(1, best_rank - 50)):
        if check_deadline():
            break
        
        found = False
        for attempt in range(50):
            if check_deadline():
                break
            
            # Random perturbation of a subset of columns
            U_cand = best_U.copy()
            V_cand = best_V.copy()
            W_cand = best_W.copy()
            
            # Try to merge some columns
            # Pick two random columns and try to replace them with one
            if target_rank >= 2:
                c1, c2 = rng.choice(best_rank, 2, replace=False)
                # Try linear combination
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    if check_deadline():
                        break
                    
                    U_new = np.zeros((n*m, target_rank))
                    V_new = np.zeros((m*p, target_rank))
                    W_new = np.zeros((p*n, target_rank))
                    
                    # Copy all columns except c1, c2
                    cols_to_keep = [c for c in range(best_rank) if c != c1 and c != c2]
                    for new_idx, old_idx in enumerate(cols_to_keep):
                        U_new[:, new_idx] = best_U[:, old_idx]
                        V_new[:, new_idx] = best_V[:, old_idx]
                        W_new[:, new_idx] = best_W[:, old_idx]
                    
                    # New column is combination
                    U_new[:, len(cols_to_keep)] = (1-alpha)*best_U[:, c1] + alpha*best_U[:, c2]
                    V_new[:, len(cols_to_keep)] = (1-alpha)*best_V[:, c1] + alpha*best_V[:, c2]
                    W_new[:, len(cols_to_keep)] = (1-alpha)*best_W[:, c1] + alpha*best_W[:, c2]
                    
                    if verify_full(U_new, V_new, W_new, n, m, p, tol=1e-6):
                        found = True
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = target_rank
                        break
            
            if found:
                break
        
        if not found and check_deadline():
            break
    
    return best_rank, best_U, best_V, best_W

def structured_search(n, m, p):
    """Try structured decompositions that might achieve lower rank."""
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook_decomp(n, m, p)
    
    # Try various structured approaches
    # The key constraint: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # One approach: use the fact that for each fixed j, we need
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,k
    
    # Try rank = m * max(n,p) or similar structured bounds
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Also rank >= n*p + m*p + n*m - n - m - p + 1 (for some cases)
    
    # Try random search with different rank targets
    for target_rank in range(24, best_rank + 1):
        if check_deadline():
            break
        
        found = False
        for seed in range(100):
            if check_deadline():
                break
            
            U = np.random.RandomState(seed).randn(n*m, target_rank) * 0.5
            V = np.random.RandomState(seed+1000).randn(m*p, target_rank) * 0.5
            W = np.random.RandomState(seed+2000).randn(p*n, target_rank) * 0.5
            
            # Quick check
            if verify_fast(U, V, W, n, m, p, tol=1e-4):
                # Full check
                if verify_full(U, V, W, n, m, p, tol=1e-9):
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if found:
            break
    
    return best_rank, best_U, best_V, best_W

def local_refinement(U, V, W, n, m, p, max_iters=100):
    """Try to refine a decomposition to achieve exactness."""
    rank = U.shape[1]
    U_c = U.copy().astype(float)
    V_c = V.copy().astype(float)
    W_c = W.copy().astype(float)
    
    # Simple iterative refinement: adjust entries to reduce error
    for it in range(max_iters):
        if check_deadline():
            break
        
        # Compute current error at each (i,j,k)
        max_err = 0
        worst_idx = None
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                        worst_idx = (i, j, k)
        
        if max_err < 1e-9:
            break
        
        # Try to adjust one entry to reduce error
        i, j, k = worst_idx
        # Gradient with respect to U[i*m+j, r]: V[j*p+k, r] * W[k*n+i, r]
        # Gradient with respect to V[j*p+k, r]: U[i*m+j, r] * W[k*n+i, r]
        # Gradient with respect to W[k*n+i, r]: U[i*m+j, r] * V[j*p+k, r]
        
        best_improvement = 0
        best_update = None
        
        for r in range(rank):
            # Try adjusting U
            gU = V_c[j*p+k, r] * W_c[k*n+i, r]
            if abs(gU) > 1e-12:
                delta = (1.0 - (np.sum(U_c[i*m+j, :] * V_c[j*p+k, :] * W_c[k*n+i, :]))) / gU
                new_val = U_c[i*m+j, r] + delta * 0.1
                if abs(new_val) < 10:
                    improvement = abs(abs(U_c[i*m+j, r] + delta*0.1 - (U_c[i*m+j, r])) * gU)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_update = ('U', i*m+j, r, new_val)
            
            # Try adjusting V
            gV = U_c[i*m+j, r] * W_c[k*n+i, r]
            if abs(gV) > 1e-12:
                delta = (1.0 - (np.sum(U_c[i*m+j, :] * V_c[j*p+k, :] * W_c[k*n+i, :]))) / gV
                new_val = V_c[j*p+k, r] + delta * 0.1
                if abs(new_val) < 10:
                    improvement = abs(abs(V_c[j*p+k, r] + delta*0.1 - (V_c[j*p+k, r])) * gV)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_update = ('V', j*p+k, r, new_val)
            
            # Try adjusting W
            gW = U_c[i*m+j, r] * V_c[j*p+k, r]
            if abs(gW) > 1e-12:
                delta = (1.0 - (np.sum(U_c[i*m+j, :] * V_c[j*p+k, :] * W_c[k*n+i, :]))) / gW
                new_val = W_c[k*n+i, r] + delta * 0.1
                if abs(new_val) < 10:
                    improvement = abs(abs(W_c[k*n+i, r] + delta*0.1 - (W_c[k*n+i, r])) * gW)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_update = ('W', k*n+i, r, new_val)
        
        if best_update is None:
            break
        
        mat_name, row, col, new_val = best_update
        if mat_name == 'U':
            U_c[row, col] = new_val
        elif mat_name == 'V':
            V_c[row, col] = new_val
        else:
            W_c[row, col] = new_val
    
    return U_c, V_c, W_c

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = schoolbook_decomp(n, m, p)
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try gradient-free search to reduce rank
    rank, U2, V2, W2 = gradient_free_search(n, m, p, best_rank)
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U2, V2, W2
    
    if check_deadline():
        # Return best found
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try structured search
    rank2, U3, V3, W3 = structured_search(n, m, p)
    if rank2 < best_rank:
        best_rank = rank2
        best_U, best_V, best_W = U3, V3, W3
    
    if check_deadline():
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try local refinement on the best solution
    if best_rank < schoolbook_rank:
        U4, V4, W4 = local_refinement(best_U, best_V, best_W, n, m, p)
        if verify_full(U4, V4, W4, n, m, p, tol=1e-9):
            best_U, best_V, best_W = U4, V4, W4
    
    # Final verification
    if not verify_full(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fall back to schoolbook
        best_U, best_V, best_W = schoolbook_decomp(n, m, p)
        best_rank = schoolbook_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
