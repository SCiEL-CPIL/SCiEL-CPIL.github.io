import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 140.0
    
    # Start with the known rank-42 block-Strassen decomposition
    def make_rank42():
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [0, 1], [2, 3]),
            ([0, 1], [0, 1], [0, 1], [4, 5]),
            ([0, 1], [2, 3], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3], [2, 3]),
            ([0, 1], [2, 3], [2, 3], [4, 5]),
        ]
        
        strassen = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
            ([(0,1)],        [(1,1),(3,-1)],[(1,1),(3,1)]),
            ([(3,1)],        [(2,1),(0,-1)],[(0,1),(2,1)]),
            ([(0,1),(1,1)],  [(3,1)],       [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        col = 0
        for a_rows, a_cols, b_rows, b_cols in blocks:
            a_map = {0: a_rows[0]*m+a_cols[0], 1: a_rows[0]*m+a_cols[1],
                     2: a_rows[1]*m+a_cols[0], 3: a_rows[1]*m+a_cols[1]}
            b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1],
                     2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
            w_map = {0: b_cols[0]*n+a_rows[0], 1: b_cols[1]*n+a_rows[0],
                     2: b_cols[0]*n+a_rows[1], 3: b_cols[1]*n+a_rows[1]}
            
            for u_c, v_c, w_c in strassen:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        return U, V, W
    
    U, V, W = make_rank42()
    
    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)
    
    # Try to reduce rank by finding linear dependencies among columns
    # A column r is redundant if it can be expressed as a linear combination of others
    # such that the tensor contribution remains the same.
    
    def try_reduce(U, V, W, max_rank):
        """Attempt to find a lower-rank equivalent decomposition."""
        rank = min(U.shape[1], max_rank)
        
        # Try eliminating columns one by one
        # Column r can be eliminated if there exist scalars alpha_s (s != r)
        # such that U[:,r] = sum_s alpha_s * U[:,s], V[:,r] = sum_s alpha_s * V[:,s],
        # and W[:,r] is absorbed into the w factors.
        # This is a constrained problem. Simpler approach:
        # Check if any column is linearly dependent in U, V, W simultaneously.
        
        cols = list(range(rank))
        
        for _ in range(20):  # Try to eliminate at most 20 columns
            eliminated = False
            for r in range(rank-1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Try to express column r as combination of columns 0..r-1
                if r == 0:
                    continue
                
                # Build design matrix: [U[:,0:r], V[:,0:r]] as a single matrix
                # We need: U[:,r] = U[:,0:r] @ x_u  AND  V[:,r] = V[:,0:r] @ x_v
                # with the SAME x for both (the combination coefficients)
                # Then W[:,r] must equal sum_s x_s * W[:,s]
                
                # Actually, the condition is:
                # For all i,j,k: sum_r U[i,r]*V[j,r]*W[k,r] = sum_{r != r0} ... 
                # If we set column r0 to zero and adjust other columns, we need:
                # U[:,r0] * V[:,r0] * W[:,r0] = sum_{s != r0} delta_U[s] * V[:,s] * W[:,s]
                #                                  + sum_{s != r0} U[:,s] * delta_V[s] * W[:,s]
                #                                  + sum_{s != r0} U[:,s] * V[:,s] * delta_W[s]
                
                # This is complex. Let's use a simpler heuristic:
                # Check if column r is exactly zero (trivially removable)
                if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    eliminated = True
                    break
                
                # Check if column r is a scalar multiple of another column in all three matrices
                for s in range(r):
                    try:
                        # Check U
                        if np.allclose(U[:, r], U[:, s]):
                            # Check V and W
                            if np.allclose(V[:, r], V[:, s]) and np.allclose(W[:, r], W[:, s]):
                                # Column r is identical to s; remove r
                                U = np.delete(U, r, axis=1)
                                V = np.delete(V, r, axis=1)
                                W = np.delete(W, r, axis=1)
                                eliminated = True
                                break
                        elif np.allclose(V[:, r], V[:, s]) and np.allclose(W[:, r], W[:, s]):
                            # V and W match, check if U is proportional
                            # Need U[:,r] = c * U[:,s] and adjust W or V
                            # If U[:,r] = c*U[:,s], V[:,r]=V[:,s], W[:,r]=W[:,s]
                            # Then term r = c * term s. Can absorb into term s's coefficients.
                            # But that changes term s, which might break other contributions.
                            # Only safe if we can scale term s and compensate.
                            # Skip for safety.
                            pass
                    except:
                        pass
                if eliminated:
                    break
            
            if not eliminated:
                break
        
        return U, V, W
    
    # Apply reduction
    U, V, W = try_reduce(U, V, W, 42)
    
    # Verify
    if not check_decomposition(U, V, W):
        # Fall back to original
        U, V, W = make_rank42()
    
    return to_lists(U, V, W)
