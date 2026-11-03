import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # 1. Construct the trivial decomposition (Rank 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    # 2. Precompute target tensor and indices
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
                valid_indices.append((a, b, c))
                
    # 3. Helper functions
    def is_valid_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is exact (within tolerance)."""
        # Check all on-diagonal terms equal 1
        for a, b, c in valid_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check all off-diagonal terms equal 0
        # We only need to check a subset if we want to speed up, but for exactness we should check all.
        # Optimization: If the tensor is sparse, we can check only non-zero targets?
        # No, off-diagonals must be zero.
        # We can speed this up by checking the full tensor contraction if R is small enough.
        # But R is up to 63. 9*21*21 = 3969 elements.
        # Let's just iterate.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integer."""
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    def als_step(U, V, W, T, R, max_iter=200, tol=1e-12):
        """
        Perform efficient ALS iterations.
        """
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for r in range(R):
                temp = np.einsum('abc,b->ac', T, V[:, r])
                U[:, r] = np.einsum('ac,c->a', temp, W[:, r])
                
            # Update V
            for r in range(R):
                temp = np.einsum('abc,a->bc', T, U[:, r])
                V[:, r] = np.einsum('bc,c->b', temp, W[:, r])
                
            # Update W
            for r in range(R):
                temp = np.einsum('abc,a->bc', T, U[:, r])
                W[:, r] = np.einsum('bc,b->c', temp, V[:, r])
                
            # Check convergence every 10 steps
            if it % 10 == 0:
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < tol:
                    break
                    
        return U, V, W

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # 4. Search for lower ranks
    # We start from 45 down to 20.
    # If we find a valid rank, we update best and continue to lower ranks.
    # If time runs out, we return the best found.
    
    for R in range(45, 19, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 1000 + R)
            
            # Initialize with small random values
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            # Run ALS
            U, V, W = als_step(U, V, W, target_tensor, R, max_iter=200)
            
            # Calculate error
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_approx - target_tensor)
            
            # If error is very small, try rounding
            if error < 1e-5:
                # Try rounding to half-integers
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-8):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break # Found valid for this rank, try lower
                
    # 5. Return the best result
    if best_rank < trivial_rank:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U_trivial),
            "V": to_list(V_trivial),
            "W": to_list(W_trivial)
        }
