"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Strategy:
    # Decompose the 4x5x5 tensor into two 2x5x5 tensors.
    # If we can find a rank-49 decomposition for 2x5x5, the total rank is 98.
    # Otherwise, fall back to schoolbook rank 100.

    def verify_decomposition(U, V, W, n, m, p, tol=1e-5):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        # Spot check zero entries to ensure no leakage
        # The tensor is zero when indices don't match.
        # e.g., U[i*m+j], V[j'*p+k], W[k'*n+i'] should be 0 if i!=i' or j!=j' or k!=k'
        # We check a few random invalid triples
        import random
        random.seed(42)
        for _ in range(50):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            # Pick a different i' or j' or k'
            i2 = (i + random.randint(1, n-1)) % n
            j2 = (j + random.randint(1, m-1)) % m
            k2 = (k + random.randint(1, p-1)) % p
            
            # We need to ensure the triple (i, j, k) vs (i2, j2, k2) is invalid
            # The tensor T[a,b,c] is non-zero only if a=(i,j), b=(j,k), c=(k,i)
            # So if we pick a random a, b, c that doesn't form a valid cycle, it should be 0.
            
            a = i*m + j
            b = j2*p + k
            c = k2*n + i2
            
            # Calculate sum
            val = np.dot(U[a], V[b] * W[c])
            if abs(val) > tol:
                return False
        return True

    def schoolbook(n, m, p):
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

    # Try to find rank-49 decomposition for 2x5x5
    def find_2x5x5_rank49():
        n_sub, m_sub, p_sub = 2, 5, 5
        target_rank = 49
        
        # Initialize with schoolbook (rank 50) and drop one column, then optimize
        U_sb, V_sb, W_sb = schoolbook(n_sub, m_sub, p_sub)
        
        # Random initialization is often better for local minima avoidance
        np.random.seed(42)
        U = np.random.randn(n_sub*m_sub, target_rank)
        V = np.random.randn(m_sub*p_sub, target_rank)
        W = np.random.randn(p_sub*n_sub, target_rank)
        
        # Alternating Minimization
        # We want to minimize sum_{a,b,c} (sum_k U[a,k]V[b,k]W[c,k] - T[a,b,c])^2
        # T is 1 if (a,b,c) form a valid (i,j,k) cycle, else 0.
        
        # Precompute the target tensor T as a sparse list of (a,b,c) -> 1
        targets = []
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    targets.append((i*m_sub+j, j*p_sub+k, k*n_sub+i))
        
        max_iter = 50
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # For each a, we want U[a, :] to minimize sum_{(a,b,c) in targets} (sum_k U[a,k] V[b,k] W[c,k] - 1)^2
            # This is a linear least squares problem for each a?
            # Let X be a matrix where rows are (b,c) pairs.
            # Y is the target vector (1 for valid, 0 for invalid? No, we only constrain valid ones to 1).
            # Actually, we must also ensure invalid ones are 0.
            # Standard tensor decomposition minimizes the full error.
            
            # Efficient update for U:
            # Fix V, W.
            # For a fixed a, let M be a matrix of size (num_targets_for_a) x target_rank
            # M[row, k] = V[b_row, k] * W[c_row, k]
            # We want U[a, :] such that M @ U[a, :] ~= 1 (for valid targets)
            # But we also have constraints for invalid targets where we want M_inv @ U[a, :] ~= 0.
            
            # Simpler: Use gradient descent on the full loss.
            # Loss = sum_{a,b,c} (sum_k U[a,k]V[b,k]W[c,k] - T[a,b,c])^2
            
            # To speed up, we can just do a few steps of gradient descent with a small learning rate.
            # Or, we can use the fact that the problem is convex in U if V,W are fixed?
            # Yes, it's a linear least squares problem if we include all (b,c) pairs.
            
            # Let's construct the full system for U.
            # Rows correspond to all (b, c) pairs.
            # Columns correspond to rank k.
            # Entry (b, c, k) = V[b, k] * W[c, k]
            # Target y(b, c) = 1 if (a, b, c) is valid, else 0.
            
            # This is O(n_sub * (m_sub*p_sub) * (p_sub*n_sub) * target_rank)
            # 2 * 25 * 10 * 49 ~ 24500 operations per a. Very fast.
            
            for a in range(n_sub * m_sub):
                # Determine i, j from a
                i = a // m_sub
                j = a % m_sub
                
                # Construct M (25 x 10 pairs? No, b is 0..24, c is 0..9)
                # Total pairs = 25 * 10 = 250
                M = np.zeros((m_sub * p_sub * p_sub * n_sub, target_rank))
                y = np.zeros(m_sub * p_sub * p_sub * n_sub)
                
                idx = 0
                for b in range(m_sub * p_sub):
                    for c in range(p_sub * n_sub):
                        # Check if (a, b, c) is a valid target
                        # a = i*m + j
                        # b = j'*p + k
                        # c = k'*n + i'
                        # Valid if j == j', k == k', i == i'
                        j_prime = b // p_sub
                        k = b % p_sub
                        k_prime = c // n_sub
                        i_prime = c % n_sub
                        
                        if j_prime == j and k_prime == k and i_prime == i:
                            y[idx] = 1.0
                        else:
                            y[idx] = 0.0
                        
                        # M row is V[b, :] * W[c, :]
                        M[idx, :] = V[b, :] * W[c, :]
                        idx += 1
                
                # Solve M @ U[a, :] = y
                # Use least squares
                sol, residuals, rank_m, s = np.linalg.lstsq(M, y, rcond=None)
                U[a, :] = sol
            
            # Similarly update V and W
            for b in range(m_sub * p_sub):
                j_prime = b // p_sub
                k = b % p_sub
                
                M = np.zeros((n_sub * m_sub * p_sub * n_sub, target_rank))
                y = np.zeros(n_sub * m_sub * p_sub * n_sub)
                
                idx = 0
                for a in range(n_sub * m_sub):
                    for c in range(p_sub * n_sub):
                        i = a // m_sub
                        j = a % m_sub
                        k_prime = c // n_sub
                        i_prime = c % n_sub
                        
                        if i == i_prime and j == j_prime and k == k_prime:
                            y[idx] = 1.0
                        else:
                            y[idx] = 0.0
                        
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                
                sol, residuals, rank_m, s = np.linalg.lstsq(M, y, rcond=None)
                V[b, :] = sol

            for c in range(p_sub * n_sub):
                k_prime = c // n_sub
                i_prime = c % n_sub
                
                M = np.zeros((n_sub * m_sub * m_sub * p_sub, target_rank))
                y = np.zeros(n_sub * m_sub * m_sub * p_sub)
                
                idx = 0
                for a in range(n_sub * m_sub):
                    for b in range(m_sub * p_sub):
                        i = a // m_sub
                        j = a % m_sub
                        j_prime = b // p_sub
                        k = b % p_sub
                        
                        if i == i_prime and j == j_prime and k == k_prime:
                            y[idx] = 1.0
                        else:
                            y[idx] = 0.0
                        
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                
                sol, residuals, rank_m, s = np.linalg.lstsq(M, y, rcond=None)
                W[c, :] = sol

        # Verify
        if verify_decomposition(U, V, W, n_sub, m_sub, p_sub, tol=1e-4):
            return U, V, W
        return None

    # Attempt to find the rank-98 solution
    U_sub, V_sub, W_sub = find_2x5x5_rank49()
    
    if U_sub is not None:
        # Construct full 4x5x5 decomposition
        # Top half (rows 0,1 of A) uses U_sub, V_sub, W_sub
        # Bottom half (rows 2,3 of A) uses U_sub, V_sub, W_sub (shifted indices)
        
        n, m, p = DIMENSIONS
        rank_total = 98
        
        U_full = np.zeros((n*m, rank_total))
        V_full = np.zeros((m*p, rank_total))
        W_full = np.zeros((p*n, rank_total))
        
        # Fill first 49 columns
        U_full[:10, :49] = U_sub
        V_full[:, :49] = V_sub
        W_full[:10, :49] = W_sub # W indices for top 2 rows of C (0..1) are 0..9
        
        # Fill next 49 columns
        # U for bottom 2 rows (10..19)
        U_full[10:20, 49:] = U_sub
        V_full[:, 49:] = V_sub
        # W for bottom 2 rows of C (10..19)
        # C index is k*n + i.
        # For top rows, i in {0,1}, so c in 0..9.
        # For bottom rows, i in {2,3}, so c in 10..19.
        W_full[10:20, 49:] = W_sub
        
        # Verify full decomposition
        if verify_decomposition(U_full, V_full, W_full, n, m, p, tol=1e-4):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    # Fallback: Schoolbook
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
