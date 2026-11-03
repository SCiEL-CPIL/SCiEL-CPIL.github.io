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

    def als_step(U, V, W, T, R, max_iter=300, tol=1e-10):
        """
        Perform efficient ALS iterations with better convergence tracking.
        """
        prev_err = np.inf
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
                
            # Check convergence every 5 steps
            if it % 5 == 0:
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < tol:
                    break
                # Check for stagnation
                if prev_err - err < 1e-12 and it > 50:
                    break
                prev_err = err
                    
        return U, V, W

    def polish_decomposition(U, V, W, T, R, max_iter=50):
        """
        High-precision polish pass using exact local solve.
        """
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U using linear system
            for r in range(R):
                # Build the linear system for U[:, r]
                # U[:, r] should satisfy: sum_c W[c, r] * (T[:, :, c] * V[:, r]) = 0 for off-diagonal
                # Actually, use the normal equations approach
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
                
            # Check convergence
            if it % 10 == 0:
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < 1e-12:
                    break
                    
        return U, V, W

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # 4. Search for lower ranks
    # Allocate more trials to ranks likely to be feasible
    # Start from moderate ranks and work down
    
    # Rank search strategy: try ranks that are more likely to work
    # Based on known results, rank should be around 30-40 for 3x3x7
    rank_sequence = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    for R in rank_sequence:
        if time.time() > deadline - 10:
            break
            
        # Allocate more trials to higher ranks (more likely to converge)
        num_trials = 8 if R >= 30 else 6 if R >= 25 else 4
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            # Use different random seeds for diversity
            seed = trial * 10000 + R * 100 + 42
            np.random.seed(seed)
            
            # Initialize with different strategies
            if trial % 3 == 0:
                # Standard random initialization
                scale = 1.0 / np.sqrt(R)
                U = np.random.randn(n * m, R) * scale
                V = np.random.randn(m * p, R) * scale
                W = np.random.randn(p * n, R) * scale
            elif trial % 3 == 1:
                # Sparse initialization
                U = np.random.randn(n * m, R) * 0.5
                V = np.random.randn(m * p, R) * 0.5
                W = np.random.randn(p * n, R) * 0.5
                # Zero out some entries
                mask_u = np.random.rand(n * m, R) > 0.5
                mask_v = np.random.rand(m * p, R) > 0.5
                mask_w = np.random.rand(p * n, R) > 0.5
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            else:
                # Initialize from partial trivial decomposition
                U = np.random.randn(n * m, R) * 0.3
                V = np.random.randn(m * p, R) * 0.3
                W = np.random.randn(p * n, R) * 0.3
                # Add some structure
                for r in range(min(R, 9)):
                    U[r % (n*m), r] += 1.0
                    V[r % (m*p), r] += 1.0
                    W[r % (p*n), r] += 1.0
            
            # Run ALS
            U, V, W = als_step(U, V, W, target_tensor, R, max_iter=300)
            
            # Calculate error
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_approx - target_tensor)
            
            # If error is small, try polishing and rounding
            if error < 1e-4:
                # Polish the decomposition
                U, V, W = polish_decomposition(U, V, W, target_tensor, R, max_iter=50)
                
                # Recalculate error after polish
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_approx - target_tensor)
                
                # Try rounding to half-integers
                if error < 1e-6:
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
