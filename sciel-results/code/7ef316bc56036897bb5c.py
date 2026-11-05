import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the multiplication tensor T
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # Actually, the tensor is defined by the multiplication map.
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r equals 1 if (i,j,k) is a valid term, 0 otherwise.
    # The "valid terms" are all (i,j,k) because every element of A and B multiplies every element of B?
    # No. C[i,k] = sum_j A[i,j] B[j,k].
    # So for each i, k, the sum over j of A[i,j] B[j,k] must equal C[i,k].
    # The tensor T is such that T[i, j, k] = 1 for all i, j, k?
    # Let's check the schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum over col (which corresponds to i,j,k) of U*V*W.
    # For a specific i,j,k, only the term where col matches i,j,k is non-zero.
    # So the sum is 1.
    # Thus T[i,j,k] = 1 for all i,j,k.
    
    T = np.ones((n, m, p), dtype=float)

    # Try to find a lower rank decomposition
    # Constraint: entries must be in {0, 0.5, 1, -0.5, -1} ? 
    # "integer/half-integer"
    
    def is_valid_entry(x):
        # Check if x is close to an integer or half-integer
        # x should be k/2 for some integer k
        y = x * 2
        return np.isclose(y, np.round(y), atol=1e-6)

    def check_decomposition(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a tensor contraction.
        # We can do this by iterating over r.
        # Or using einsum.
        # U_reshaped: (n, m, R)
        # V_reshaped: (m, p, R)
        # W_reshaped: (p, n, R)
        
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Sum over R
        # C[i, k] is not the target, the tensor T[i, j, k] is.
        # The product is U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: V is indexed by j, k. W is indexed by k, i.
        # So for fixed i, j, k, we sum over r.
        
        # This is not a standard matmul.
        # We can compute it as:
        # For each r, form the tensor U_r (n,m) * V_r (m,p) * W_r (p,n)?
        # No, it's element-wise products summed.
        
        # Efficient way:
        # Create a tensor P of size (n, m, p)
        # P[i, j, k] = sum_r U_r[i, j] V_r[j, k] W_r[k, i]
        
        # We can do this by summing outer products?
        # Let's just loop over r if R is small.
        P = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U_r[:, :, r]
            v = V_r[:, :, r]
            w = W_r[:, :, r]
            # u is (n, m), v is (m, p), w is (p, n)
            # We need u[i, j] * v[j, k] * w[k, i]
            # This is the trace of u[i, j] * v[j, k] * w[k, i]?
            # For fixed i, j: sum_k v[j, k] w[k, i] is (V_r @ W_r)[j, i]
            # So P[i, j, k] is not just a matrix product.
            
            # Let's expand:
            # Term for i, j, k is u[i, j] * v[j, k] * w[k, i]
            # We can compute this by:
            # 1. Compute M = V_r @ W_r. M is (m, n). M[j, i] = sum_k v[j, k] w[k, i].
            # This sums over k, but we need k fixed.
            
            # Alternative:
            # For each r, compute the tensor T_r = u (n,m) * v (m,p) * w (p,n) ?
            # No, it's element-wise multiplication of the three matrices at specific indices.
            
            # Let's use a loop for simplicity, R is likely small.
            for i in range(n):
                for j in range(m):
                    u_val = u[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        P[i, j, k] += u_val * v[j, k] * w[k, i]
                        
        return np.allclose(P, T, atol=1e-6)

    # Search
    deadline = time.time() + 10.0 # Allow 10s for search
    best_rank = rank_schoolbook
    best_decomp = None
    
    # Initialize schoolbook
    U_s = np.zeros((n * m, rank_schoolbook))
    V_s = np.zeros((m * p, rank_schoolbook))
    W_s = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    best_decomp = {"U": U_s, "V": V_s, "W": W_s}

    # Try ranks below schoolbook
    for R in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline:
            break
            
        # Try a few random inits
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Solve for W
            # System for each k, i:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k] = 1
            # Let A_{j, r} = U[i*m+j, r] * V[j*p+k, r]
            # Then sum_r A_{j, r} W_{k*n+i, r} = 1 for all j.
            # This is A @ w_vec = ones(m)
            
            valid = True
            W = np.zeros((p * n, R))
            
            for k in range(p):
                for i in range(n):
                    # Construct A (m x R)
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    
                    b = np.ones(m)
                    
                    # Solve A w = b
                    # Use least squares
                    try:
                        w_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if solution is exact
                        if not np.allclose(A @ w_vec, b, atol=1e-6):
                            valid = False
                            break
                            
                        # Check if w_vec entries are half-integers
                        if not all(is_valid_entry(x) for x in w_vec):
                            valid = False
                            break
                            
                        W[k * n + i, :] = w_vec
                    except:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            if check_decomposition(U, V, W):
                best_rank = R
                best_decomp = {"U": U, "V": V, "W": W}
                # Break out of loops? 
                # We found a lower rank. Can we find even lower?
                # The loop continues to lower R.
                pass

    # Return the best found
    # Convert to lists of lists to match interface?
    # The seed returns lists of lists.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed uses `[[0] * rank ...]`.
    
    U_out = best_decomp["U"].tolist()
    V_out = best_decomp["V"].tolist()
    W_out = best_decomp["W"].tolist()
    
    # Ensure they are exact numbers if possible
    # If they are floats like 1.0, convert to int?
    # The seed uses ints.
    # If my solution has 0.5, it should stay 0.5.
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
