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

    def als_step(U, V, W, T, R, max_iter=100, tol=1e-10):
        """
        Perform ALS iterations to minimize ||T - sum(U_r V_r W_r)||^2.
        """
        # Precompute contractions for speed
        # For updating U[:, r]: U[:, r] = T x_2 V[:, r] x_3 W[:, r]
        # T x_2 v is a matrix of size (n*m, p*n)
        # Then x_3 w is a vector of size (n*m)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            old_U = U.copy()
            old_V = V.copy()
            old_W = W.copy()
            
            # Update U
            for r in range(R):
                # Compute residual T - sum_{r' != r} U_{r'} V_{r'} W_{r'}
                # This is expensive if done naively.
                # Instead, compute the full tensor T_approx = einsum(U,V,W)
                # And subtract the r-th term.
                # But we can compute the update directly:
                # U[:, r] = (T - T_approx + U[:, r] V[:, r] W[:, r]) x_2 V[:, r] x_3 W[:, r]
                
                # Let's compute T_approx once per outer iteration? No, U changes.
                # Efficient update:
                # Let Z = T - sum_{r' != r} U_{r'} \otimes V_{r'} \otimes W_{r'}
                # We want u such that u \otimes v_r \otimes w_r is best fit to Z.
                # u = Z x_2 v_r x_3 w_r
                
                # Calculate sum_{r' != r}
                # This requires O(R^2) or maintaining a running sum.
                # Given small R (<= 50), O(R^2) is fine.
                
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                # Subtract r-th term
                T_approx -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Contract with V[:, r] and W[:, r]
                # First contract with W[:, r] over c
                # Result shape: (n*m, m*p)
                temp = np.einsum('abc,c->ab', T_approx, W[:, r])
                # Then contract with V[:, r] over b
                # Result shape: (n*m)
                U[:, r] = np.einsum('ab,b->a', temp, V[:, r])
                
            # Update V
            for r in range(R):
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                T_approx -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Contract with U[:, r] over a
                temp = np.einsum('abc,a->bc', T_approx, U[:, r])
                # Then contract with W[:, r] over c
                V[:, r] = np.einsum('bc,c->b', temp, W[:, r])
                
            # Update W
            for r in range(R):
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                T_approx -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Contract with U[:, r] over a
                temp = np.einsum('abc,a->bc', T_approx, U[:, r])
                # Then contract with V[:, r] over b
                W[:, r] = np.einsum('bc,b->c', temp, V[:, r])
                
            # Check convergence
            diff_U = np.linalg.norm(U - old_U)
            diff_V = np.linalg.norm(V - old_V)
            diff_W = np.linalg.norm(W - old_W)
            if diff_U + diff_V + diff_W < tol:
                break
                
        return U, V, W

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # Try ranks from high to low
    # We know rank 63 works.
    # Let's try to find a rank < 63.
    # Based on literature, rank of 3x3x7 might be around 40-50.
    # We'll try from 50 down to 30.
    
    for R in range(50, 29, -1):
        if time.time() > deadline - 15:
            break
            
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 1000 + R)
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Run ALS
            U, V, W = als_step(U, V, W, target_tensor, R, max_iter=200)
            
            # Calculate error
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_approx - target_tensor)
            
            # If error is very small, try rounding
            if error < 1e-6:
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-8):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break # Found valid for this rank, try lower
            
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
