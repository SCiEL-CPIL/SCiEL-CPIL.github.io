"""Rewritten program: structured search for low-rank exact tensor decomposition of 3x3 @ 3x7."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_decomp(n=3, m=3, p=7):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=7, tol=1e-8):
    """
    Verify that the decomposition exactly reproduces the multiplication tensor.
    T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c] must equal 1 iff (u,v,w) is a valid (i*m+j, j*p+k, k*n+i).
    """
    rank = U.shape[1]
    # Build the full tensor efficiently
    T = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    
    # Check all valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > tol:
                    return False
    
    # Check that invalid positions are zero
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i * m + j, j * p + k, k * n + i))
    
    # Sample invalid positions (and check all if feasible)
    total = n * m * m * p * p * n
    if total <= 50000:
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) not in valid_set:
                        if abs(T[u, v, w]) > tol:
                            return False
    else:
        # Sample 2000 invalid positions
        rng = np.random.RandomState(0)
        for _ in range(2000):
            u = rng.randint(0, n * m)
            v = rng.randint(0, m * p)
            w = rng.randint(0, p * n)
            if (u, v, w) not in valid_set:
                if abs(T[u, v, w]) > tol:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, num_trials=50, seed=42):
    """Try random initializations and verify."""
    rng = np.random.RandomState(seed)
    for t in range(num_trials):
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return {"U": U, "V": V, "W": W}
    return None

def _try_structured_block_decomposition(n, m, p, deadline):
    """
    Strategy: Exploit block structure.
    Split B into B1 (3x3) and B2 (3x4).
    C1 = A @ B1 (3x3), C2 = A @ B2 (3x4).
    Use a known low-rank decomposition for 3x3x3 and add rank-1 terms for the 3x4 part.
    
    For 3x3x3, the best known rank is 23 (Lott). We'll use the schoolbook rank-27 for 3x3x3
    and see if we can do better, or use a structured approach.
    
    Actually, let's try a more direct approach:
    The tensor for 3x3x7 is a sub-tensor of 3x3x7 (which is what we have).
    We can try to find a decomposition by treating it as:
    For each column j of B (0..6), we compute A @ B[:,j] which is a 3x3 matrix times a 3-vector.
    This is a 3x3x1 multiplication, which has rank 9 (schoolbook) but can potentially be lower.
    
    Actually, 3x3x1 is just 3x3 matrix-vector multiplication. The bilinear complexity is 9
    (you can't do better than 9 for 3x3 times a vector in general, since it's a linear map
    from R^3 to R^3 applied to 3 independent vectors... wait, no.
    
    C[:,j] = A @ B[:,j]. This is a linear map from R^3 to R^3. The bilinear complexity
    of computing A @ v for a 3x3 A and 3-vector v is 9 (each output is a dot product of 3 terms).
    But wait, we can use Strassen-like tricks? No, for 3x3x1, the rank is exactly 9 because
    it's just 9 independent dot products.
    
    So for 3x3x7, we have 7 independent 3x3x1 problems. The total schoolbook rank is 7*9 = 63.
    Can we share computations across the 7 columns?
    
    Yes! If we group columns, we can use 3x3xk decompositions. For example:
    - 3x3x3 has rank 23 (Lott) or 27 (schoolbook)
    - 3x3x4 has rank ? 
    - 3x3x7 has rank ?
    
    Let's try to construct a decomposition using the 3x3x3 Lott algorithm (rank 23) for the
    first 3 columns, and then handle the remaining 4 columns.
    
    Actually, let me try a different approach: use the fact that the tensor is a "slice" of a
    larger tensor and try to find a low-rank approximation.
    
    For now, let's try a gradient-based optimization approach.
    """
    # This is a placeholder for a more sophisticated approach.
    # We'll try to use a structured initialization based on block decomposition.
    
    # Split into 3x3x3 (cols 0-2) and 3x3x4 (cols 3-6)
    # For 3x3x3, use schoolbook (rank 27) as a starting point
    # For 3x3x4, use schoolbook (rank 12)
    # Total: 27 + 12 = 39
    
    # But we can do better by sharing structure.
    # Let's try rank 30 and see if we can find a valid decomposition.
    
    for target_rank in [30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]:
        if time.time() > deadline - 15:
            break
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=10, seed=target_rank)
        if result is not None:
            return result
    return None

def _try_gradient_descent(n, m, p, rank, deadline):
    """
    Try to minimize the error of the tensor decomposition using gradient descent.
    """
    rng = np.random.RandomState(123)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    lr = 0.01
    for step in range(500):
        if time.time() > deadline - 10:
            break
        
        # Compute current tensor
        T_current = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            T_current += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        error = np.sum((T_current - T_target) ** 2)
        if error < 1e-10:
            if _verify(U, V, W, n, m, p, tol=1e-8):
                return {"U": U, "V": V, "W": W}
            break
        
        # Compute gradients (approximate)
        # dU[u,c] = sum_{v,w} (T_current[u,v,w] - T_target[u,v,w]) * V[v,c] * W[w,c]
        # This is expensive, so let's use a simpler approach: just try random perturbations
        
        # Random perturbation
        if step % 10 == 0:
            U += rng.randn(n * m, rank) * 0.01
            V += rng.randn(m * p, rank) * 0.01
            W += rng.randn(p * n, rank) * 0.01
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best_result = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p  # 63
    
    # Try structured approaches
    for target_rank in range(24, 63):
        if time.time() > deadline - 20:
            break
        
        # Try random search
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=5, seed=target_rank * 7 + 1)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    # If we found something better, verify and return
    if best_rank < 63:
        return {
            "U": best_result["U"].tolist(),
            "V": best_result["V"].tolist(),
            "W": best_result["W"].tolist()
        }
    
    # Return schoolbook as fallback
    return {
        "U": best_result["U"].tolist(),
        "V": best_result["V"].tolist(),
        "W": best_result["W"].tolist()
    }
# EVOLVE-BLOCK-END
