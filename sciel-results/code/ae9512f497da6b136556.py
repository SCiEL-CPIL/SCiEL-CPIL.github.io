<<<<<<< SEARCH
def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True
=======
def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _local_search(U, V, W, n, m, p, max_iter=5000, tol=1e-9):
    """
    Local search: perturb entries slightly and see if we can reduce rank.
    Actually, for rank reduction, we need a different approach.
    This function tries to find a valid decomposition at the given rank
    by starting from a random initialization and doing gradient-like steps.
    """
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Precompute all non-zero and zero indices
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    zero_indices = []
    for a in range(n * m):
        for b in range(m * p):
            j = a % m
            i = a // m
            k = b % p
            c_expected = k * n + i
            for c in range(p * n):
                if c != c_expected:
                    zero_indices.append((a, b, c))
    
    # Sample a subset for speed
    random.seed(42)
    nz_sample = random.sample(nonzero_indices, min(200, len(nonzero_indices)))
    z_sample = random.sample(zero_indices, min(500, len(zero_indices)))
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        # Compute errors
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Non-zero: want sum_r U[a,r]*V[b,r]*W[c,r] = 1
        for (a, b, c) in nz_sample:
            err = 0.0
            for r in range(rank):
                err += U[a, r] * V[b, r] * W[c, r]
            err -= 1.0
            for r in range(rank):
                grad_U[a, r] += err * V[b, r] * W[c, r]
                grad_V[b, r] += U[a, r] * err * W[c, r]
                grad_W[c, r] += U[a, r] * V[b, r] * err
        
        # Zero: want sum_r U[a,r]*V[b,r]*W[c,r] = 0
        for (a, b, c) in z_sample:
            err = 0.0
            for r in range(rank):
                err += U[a, r] * V[b, r] * W[c, r]
            for r in range(rank):
                grad_U[a, r] += err * V[b, r] * W[c, r]
                grad_V[b, r] += U[a, r] * err * W[c, r]
                grad_W[c, r] += U[a, r] * V[b, r] * err
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check if converged
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, trials=50):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None
>>>>>>> REPLACE

