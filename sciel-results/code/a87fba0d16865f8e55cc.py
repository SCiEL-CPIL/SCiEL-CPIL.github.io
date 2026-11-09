"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k,l] = 1 if j==k else 0.
    Here i in [0,n), j in [0,m), k in [0,m), l in [0,p).
    The tensor has shape (n*m, m*p, p*n) when flattened appropriately.
    Actually the tensor is T[A_idx, B_idx, C_idx] where:
      A_idx = i*m + j  (i in [0,n), j in [0,m))
      B_idx = j'*p + k (j' in [0,m), k in [0,p))
      C_idx = k'*n + i' (k' in [0,p), i' in [0,n))
    T[A_idx, B_idx, C_idx] = 1 if j==j' and k==k' and i==i' ... 
    Wait, let me think more carefully.
    
    The matrix multiplication tensor: (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    So the tensor T[i, j, k, j', k'] where we multiply A[i,j] and B[j',k] and get C[i,k].
    T[i,j,k,j',k'] = 1 if j==j' else 0, with i in [0,n), j in [0,m), k in [0,p), j' in [0,m), k' in [0,p).
    
    As a 3-tensor: T[A_idx, B_idx, C_idx] where
      A_idx = i*m + j
      B_idx = j'*p + k  
      C_idx = k*n + i
    T[A_idx, B_idx, C_idx] = 1 if j == j' else 0
    (i, k are determined by the indices)
    
    Actually: T[i*m+j, j'*p+k, k*n+i] = 1 if j==j' else 0
    """
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A_idx = i * m + j
                B_idx = j * p + k
                C_idx = k * n + i
                T[A_idx, B_idx, C_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the tensor."""
    T = _build_tensor(n, m, p)
    # Reconstruct: T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _try_rank_random(rank, n, m, p, num_trials=50, time_budget=30.0):
    """Try to find a rank-r decomposition using random initialization and iterative refinement.
    Uses a smooth least-squares approach on the tensor flattening."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    nA, nB, nC = n*m, m*p, p*n
    
    best_err = float('inf')
    best_decomp = None
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(nA, rank)
        V = np.random.randn(nB, rank)
        W = np.random.randn(nC, rank)
        
        # Iterative refinement: minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
        # Using alternating least squares
        for iteration in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each a, minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # This is a linear least squares problem for each a
            # T[a] = sum_r U[a,r] * (V[r] ⊗ W[r])
            # So U[a,:] = T[a].reshape(-1) @ (V⊗W matrix)^+
            
            # Build the matrix M where M[:, r] = V[:,r] ⊗ W[:,r]
            # M has shape (nB*nC, rank)
            M_VW = np.zeros((nB * nC, rank))
            for r in range(rank):
                M_VW[:, r] = np.outer(V[:, r], W[:, r])
            
            # For each a: T[a].flatten() = M_VW @ U[a,:]
            # Solve: U = T_flat @ pinv(M_VW)
            T_flat = T.reshape(nA, nB * nC)
            U_new = T_flat @ np.linalg.pinv(M_VW)
            
            # Update V: for each b, T[:,b,:] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
            M_UW = np.zeros((nA * nC, rank))
            for r in range(rank):
                M_UW[:, r] = np.outer(U[:, r], W[:, r])
            T_flat_b = T.transpose(1, 0, 2).reshape(nB, nA * nC)
            V_new = T_flat_b @ np.linalg.pinv(M_UW)
            
            # Update W: for each c, T[:,:,c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
            M_UV = np.zeros((nA * nB, rank))
            for r in range(rank):
                M_UV[:, r] = np.outer(U[:, r], V[:, r])
            T_flat_c = T.transpose(2, 0, 1).reshape(nC, nA * nB)
            W_new = T_flat_c @ np.linalg.pinv(M_UV)
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            T_recon = np.zeros((nA, nB, nC))
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                best_decomp = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                return U, V, W, err
    
    if best_decomp is not None and best_err < 1e-10:
        return best_decomp[0], best_decomp[1], best_decomp[2], best_err
    return None

def _try_rank_greedy(rank, n, m, p, time_budget=30.0):
    """Try greedy decomposition: iteratively find the best rank-1 term."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    nA, nB, nC = n*m, m*p, p*n
    T_remaining = T.copy()
    
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(rank):
        if time.time() - start_time > time_budget:
            break
        
        # Find best rank-1 approximation of T_remaining
        # Flatten and use SVD-like approach
        # T_remaining[a,b,c] ≈ u[a] * v[b] * w[c]
        # This is equivalent to finding the best rank-1 in a 3-way tensor
        # We can use alternating projections
        
        # Initialize
        u = np.random.randn(nA)
        v = np.random.randn(nB)
        w = np.random.randn(nC)
        
        best_rank1_err = float('inf')
        best_u, best_v, best_w = u, v, w
        
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            
            # Update u: for each a, T_remaining[a,:,:] ≈ u[a] * outer(v, w)
            vw = np.outer(v, w)
            if np.linalg.norm(vw) > 1e-15:
                u_new = np.array([np.dot(T_remaining[a], vw.flatten()) / (np.dot(vw.flatten(), vw.flatten())) for a in range(nA)])
            else:
                u_new = u
            
            # Update v: for each b, T_remaining[:,b,:] ≈ outer(u, w)
            uw = np.outer(u, w)
            if np.linalg.norm(uw) > 1e-15:
                v_new = np.array([np.dot(T_remaining[:, b, :], uw.flatten()) / (np.dot(uw.flatten(), uw.flatten())) for b in range(nB)])
            else:
                v_new = v
            
            # Update w: for each c, T_remaining[:,:,c] ≈ outer(u, v)
            uv = np.outer(u, v)
            if np.linalg.norm(uv) > 1e-15:
                w_new = np.array([np.dot(T_remaining[:, :, c], uv.flatten()) / (np.dot(uv.flatten(), uv.flatten())) for c in range(nC)])
            else:
                w_new = w
            
            u, v, w = u_new, v_new, w_new
            
            # Compute error
            T_approx = np.einsum('a,b,c->abc', u, v, w)
            err = np.linalg.norm(T_remaining - T_approx)
            if err < best_rank1_err:
                best_rank1_err = err
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        u, v, w = best_u, best_v, best_w
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Subtract this rank-1 term
        T_remaining -= np.einsum('a,b,c->abc', u, v, w)
        
        # Check if we've converged
        if np.linalg.norm(T_remaining) < 1e-10:
            break
    
    if np.linalg.norm(T_remaining) < 1e-10:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        return U, V, W, np.linalg.norm(T_remaining)
    
    return None

def _try_rank_integer_search(rank, n, m, p, time_budget=20.0):
    """Try to find a decomposition with integer/half-integer entries.
    Search over small integer values for the entries."""
    start_time = time.time()
    
    # For small ranks, we can try to find integer solutions
    # The tensor entries are 0 or 1, so the decomposition should have rational entries
    
    # Try a structured approach: assume the decomposition has a specific form
    # For 2x3x3x5, the known minimum rank is unknown, but let's try various approaches
    
    # One approach: use the fact that matrix multiplication can be decomposed
    # using algorithms like Winograd's for small cases
    
    # For 2x3 * 3x5, we can think of it as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # etc.
    
    # Let's try a direct approach with small integer values
    # The entries of U, V, W should be from {-2,-1,-0.5,0,0.5,1,2}
    
    candidates = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    # This is a huge search space, so we need to be clever
    # Let's try a local search approach starting from the schoolbook decomposition
    
    # Start with schoolbook
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # and projecting out
    
    # Actually, let's try a different approach: use the structure of the problem
    # to find a low-rank decomposition
    
    # For 2x3 * 3x5, we have 10 output elements, each is a sum of 3 products
    # Total 30 products. Can we share computations?
    
    # Let's try rank 15, 20, 25, etc.
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication 2x3 by 3x5.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with schoolbook decomposition (rank 30)
    rank_schoolbook = n * m * p
    U = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # Try ranks from 15 to 29
    for rank in range(15, rank_schoolbook):
        if time.time() > deadline - 30.0:
            break
        
        # Try random ALS
        result = _try_rank_random(rank, n, m, p, num_trials=20, 
                                  time_budget=min(10.0, max(1.0, deadline - time.time() - 20.0)))
        if result is not None:
            U_arr, V_arr, W_arr, err = result
            if err < 1e-10:
                best_rank = rank
                best_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist()
                }
                break  # Found a valid decomposition, stop
        
        # Try greedy
        if time.time() > deadline - 20.0:
            break
        result = _try_rank_greedy(rank, n, m, p, 
                                  time_budget=min(5.0, max(1.0, deadline - time.time() - 15.0)))
        if result is not None:
            U_arr, V_arr, W_arr, err = result
            if err < 1e-10:
                best_rank = rank
                best_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist()
                }
                break
    
    # Verify the best result
    if best_result is not None:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        if not _verify_decomposition(U_arr, V_arr, W_arr, n, m, p, tol=1e-10):
            # Fall back to schoolbook
            best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
