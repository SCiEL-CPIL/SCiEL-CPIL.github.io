"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 115  # 115s budget, 5s safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact using vectorized operations."""
        # For each (i, k), verify that the sum equals delta(j1,j2) for all j1,j2
        # T[i*m+j1, j2*p+k, k*n+i] should be 1 if j1==j2 else 0
        # Sum over c: W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
        
        rank = U.shape[1]
        # Build the full tensor check
        # For each (i, k):
        for i in range(n):
            for k in range(p):
                # Get the relevant slices
                U_slice = U[i*m:(i+1)*m, :]  # shape (m, rank)
                V_slice = V[:, k]  # shape (m,) -- wait, V is (m*p, rank), so V[j*p+k, :] for j in range(m)
                # Actually, V[j*p + k, c] for j in range(m)
                V_cols = np.array([V[j*p + k, :] for j in range(m)])  # shape (m, rank)
                W_val = W[k*n + i, :]  # shape (rank,)
                
                # The product for each (j1, j2):
                # sum_c W_val[c] * U_slice[j1, c] * V_cols[j2, c]
                # = W_val @ (U_slice.T * V_cols.T) -- not quite
                # Actually: for each c, the contribution is W_val[c] * U_slice[:, c] * V_cols[:, c]
                # So the (j1, j2) entry is sum_c W_val[c] * U_slice[j1,c] * V_cols[j2,c]
                # = sum_c (W_val * U_slice.T)[c] * V_cols[j2,c] -- no
                # Let's just do it directly:
                # result[j1, j2] = sum_c W_val[c] * U_slice[j1, c] * V_cols[j2, c]
                
                # Vectorized: 
                # temp[j1, c] = U_slice[j1, c] * W_val[c]
                temp = U_slice * W_val  # shape (m, rank)
                # result[j1, j2] = sum_c temp[j1, c] * V_cols[j2, c]
                result = temp @ V_cols.T  # shape (m, m)
                
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True
    
    def try_rank(r, num_restarts=30):
        """Try to find a valid decomposition with rank r."""
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Pre-compute A for this restart
            A = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
            
            # For each w (output index), solve for W[w, :]
            # The equation: for all (u, v), sum_c U[u,c] * V[v,c] * W[w,c] = T[u,v,w]
            # Where T[u,v,w] = 1 iff (u%m == v//p) and (u//m == w%n) and (v%p == w//n)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                i_out = w % n  # output column
                k_out = w // n  # output row
                
                # Build the linear system
                # Number of equations: n*m * m*p = 16 * 36 = 576
                # But we can be smarter: only need equations where T could be nonzero
                # T[u,v,w] = 1 iff u%m == v//p and u//m == i_out and v%p == k_out
                # So u must have u//m == i_out, i.e., u in [i_out*m, (i_out+1)*m)
                # And v must have v%p == k_out, i.e., v in {k_out, k_out+p, k_out+2p, ...}
                # And u%m == v//p
                
                # Let's build the full system but only for relevant (u,v) pairs
                # Actually, we need ALL (u,v) pairs to ensure zeros are satisfied too
                
                # For efficiency, let's build A and b
                # A has shape (num_eq, r), b has shape (num_eq,)
                
                # Collect all (u, v) pairs where T[u,v,w] might be nonzero
                # T[u,v,w] = 1 iff u%m == v//p and u//m == i_out and v%p == k_out
                # So u must be in [i_out*m, (i_out+1)*m)
                # And for each such u, v must satisfy v//p == u%m and v%p == k_out
                # So v = (u%m)*p + k_out (if u%m < m and k_out < p)
                
                # Nonzero entries: for each j in range(m), u = i_out*m + j, v = j*p + k_out
                # That's m = 4 nonzero entries per w
                
                # But we also need to ensure that for all OTHER (u,v), the sum is 0
                # So we need the full system of 576 equations
                
                # For speed, let's use the full system but vectorize
                # A[(u,v), c] = U[u,c] * V[v,c]
                # b[(u,v)] = 1 if (u%m == v//p and u//m == i_out and v%p == k_out) else 0
                
                # Pre-compute indices for b
                # We can pre-compute the (u,v) pairs for each w
                # But A is the same for all w.
                # Let's move A construction out of the w loop.
                # However, the code structure has A inside the loop.
                # We will refactor the loop to build A once.
                
                # Since we are inside the `for w` loop, we cannot easily move A out without changing the structure.
                # Let's just optimize the b construction.
                # b is non-zero only at specific (u,v) pairs.
                # For a given w (i_out, k_out):
                # Non-zero if i_u == i_out and k_v == k_out and j_u == j_v.
                # u = i_out * m + j
                # v = j * p + k_out
                # So for j in range(m):
                # u = i_out * m + j
                # v = j * p + k_out
                # b[u * m * p + v] = 1.0
                # All other entries are 0.
                
                b = np.zeros(n * m * m * p)
                for j in range(m):
                    u = i_out * m + j
                    v = j * p + k_out
                    b[u * (m * p) + v] = 1.0
                
                # Solve A @ W[w,:] = b
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None
    
    # Get schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try lower ranks
    # Lower bound: max(n*m, m*p, p*n) = 36
    # Try from 36 upward
    for r in range(36, 144):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_restarts=15)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
