"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  The multiplication tensor T_{i,j,k} for multiplying (n x m) * (m x p) has
  trivial rank n*m*p = 63. We attempt to find a lower-rank decomposition.
  
  Key insight: The system of equations for the decomposition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)
    sum_r U[a, r] * V[b, r] * W[c, r] = 0  for all other (a,b,c)
  
  This is a system of polynomial equations. We can solve it using:
  1. Random initialization + alternating least squares (ALS)
  2. For each ALS step, we fix two matrices and solve linear systems for the third
  3. After convergence, we round to nearest half-integers and verify exactness
  
  The critical improvement over naive ALS: we must ensure that the "off-diagonal"
  terms (where indices don't match) also vanish. This requires careful handling.
  
  Approach:
  - Start with schoolbook rank 63
  - Try to reduce rank by finding linear dependencies
  - Use ALS with both on-diagonal (=1) and off-diagonal (=0) constraints
  - Round to half-integers and verify exactness
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Trivial decomposition (rank = n*m*p = 63)
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
    
    # Precompute target tensor indices
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
                
    # Precompute index mappings for faster verification
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i * m + j, j * p + k, k * n + i))
                
    def is_valid_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is exact (within tolerance)."""
        # Check all on-diagonal terms equal 1
        for a, b, c in valid_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check all off-diagonal terms equal 0
        # Iterate over all a, b, c
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
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return U_rounded, V_rounded, W_rounded

    def compute_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def als_step(U, V, W, T, R, max_iter=500, tol=1e-12):
        """
        Perform efficient ALS iterations to minimize ||T - sum(U_r V_r W_r)||^2.
        Uses direct contraction of T with current V and W estimates.
        """
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for r in range(R):
                # U[:, r] = vec( T x_2 V[:, r] x_3 W[:, r] )
                # Contract T with V[:, r] along axis 1 (b)
                # T: (A, B, C), V[:, r]: (B,) -> (A, C)
                temp = np.einsum('abc,b->ac', T, V[:, r])
                # Contract result with W[:, r] along axis 1 (c)
                # temp: (A, C), W[:, r]: (C,) -> (A,)
                U[:, r] = np.einsum('ac,c->a', temp, W[:, r])
                
            # Update V
            for r in range(R):
                # V[:, r] = vec( T x_1 U[:, r] x_3 W[:, r] )
                # Contract T with U[:, r] along axis 0 (a)
                # T: (A, B, C), U[:, r]: (A,) -> (B, C)
                temp = np.einsum('abc,a->bc', T, U[:, r])
                # Contract result with W[:, r] along axis 1 (c)
                # temp: (B, C), W[:, r]: (C,) -> (B,)
                V[:, r] = np.einsum('bc,c->b', temp, W[:, r])
                
            # Update W
            for r in range(R):
                # W[:, r] = vec( T x_1 U[:, r] x_2 V[:, r] )
                # Contract T with U[:, r] along axis 0 (a)
                # T: (A, B, C), U[:, r]: (A,) -> (B, C)
                temp = np.einsum('abc,a->bc', T, U[:, r])
                # Contract result with V[:, r] along axis 0 (b)
                # temp: (B, C), V[:, r]: (B,) -> (C,)
                W[:, r] = np.einsum('bc,b->c', temp, V[:, r])
                
            # Check convergence (optional, but good for stopping early)
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
    
    # Try ranks from high to low
    # With efficient ALS, we can try more ranks and more trials.
    # Start from 45 down to 15.
    
    for R in range(45, 14, -1):
        if time.time() > deadline - 15:
            break
            
        # Try more random initializations due to speedup
        for trial in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 1000 + R)
            
            # Initialize with small random values
            # Scaling matters. 1/sqrt(R) is often better for stability.
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            # Run ALS
            U, V, W = als_step(U, V, W, target_tensor, R, max_iter=500)
            
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
                
                # If half-integer rounding failed, maybe try integer rounding?
                # Or check if the unrounded solution is effectively exact
                # and just return it if the problem allowed floats? 
                # No, problem requires integer/half-integer.
                # Let's try a local search around the rounded values?
                # For now, just break to try next rank if this trial failed.
            
    # If we found a better rank, return it. Otherwise return trivial.
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
