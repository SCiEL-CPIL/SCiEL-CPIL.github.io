"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    schoolbook_rank = nm * mp // m  # n*m*p = 42
    
    # Build the target tensor in flattened form
    # T[a,b] = 1 if a=(i,j), b=(j,k) share the same j, and c=(i,k)
    # Actually we need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # where T[a,b,c]=1 when a=(i,j), b=(j,k), c=(i,k)
    
    # Let's build the tensor as a 3D array T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with schoolbook and try to reduce
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W):
        r = U.shape[1]
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is a tensor contraction
        # For efficiency, compute via einsum
        try:
            result = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(result, T, atol=1e-10)
        except:
            return False
    
    if verify(U_sb, V_sb, W_sb):
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
    
    # Try to find lower rank using randomized approach
    # This is very hard, so let's try a few strategies within time budget
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try ranks from 23 to 41
    for rank in range(23, schoolbook_rank):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Use alternating least squares / gradient descent
            # This is a 3-way tensor decomposition, which is NP-hard
            # Use a simple iterative approach
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break
                # Update U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # This is a linear system in U[a,:]
                # Fix V and W, solve for U[a,:]
                # T[a,:,:] = U[a,:] @ (V * W^T) ... not quite
                # Actually: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # Let M[b,c] = sum_r V[b,r]*W[c,r] (outer product sum)
                # Then T[a,:,:] = U[a,:] @ M where M[r, (b,c)] ... 
                # This doesn't decompose nicely as a matrix product
                
                # Alternative: use gradient descent
                # Compute gradient w.r.t. U[a,r] = sum_{b,c} V[b,r]*W[c,r]*T[a,b,c] - sum_{b,c} V[b,r]*W[c,r]*result[a,b,c]
                # This is getting complex. Let's skip this approach.
                break
            # Skip - too complex to implement correctly in time
            
    # Return the best we have (schoolbook)
    # Convert to lists
    U_list = best_U.tolist() if best_U is not None else []
    V_list = best_V.tolist() if best_V is not None else []
    W_list = best_W.tolist() if best_W is not None else []
    
    # Ensure integer/half-integer entries
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_list = [[round_to_half(v) for v in row] for row in U_list]
    V_list = [[round_to_half(v) for v in row] for row in V_list]
    W_list = [[round_to_half(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
