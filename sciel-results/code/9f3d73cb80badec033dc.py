import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    
    Strategy:
    1. Use a Greedy Rank-1 Subtraction approach with discrete candidate vectors.
    2. The candidates are restricted to small half-integers {-1, -0.5, 0, 0.5, 1}.
    3. At each step, select the rank-1 component that most reduces the residual tensor norm.
    4. Stop when the residual is zero (exact decomposition found).
    5. If no exact solution is found within time, fall back to the schoolbook rank-64 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    dim_u = n * m  # 8
    dim_v = m * p  # 32
    dim_w = p * n  # 16
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to check if entries are half-integers
    def is_half_integer(M, tol=1e-6):
        M2 = M * 2
        return np.all(np.abs(M2 - np.round(M2)) < tol)

    # Greedy search
    def greedy_search(max_rank=30, time_limit=130.0):
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Candidate values for vectors
        # We restrict to small half-integers to keep the search space manageable
        # and to increase the chance of finding an exact half-integer solution.
        # However, the vectors can be longer.
        # We will generate random candidate vectors from the set {-1, -0.5, 0, 0.5, 1}
        # and try to fit them.
        
        # To make it faster, we can use a heuristic:
        # For each step, we want to find u, v, w such that <u \otimes v \otimes w, residual> is large.
        # This is equivalent to maximizing the tensor contraction.
        
        # We will perform random restarts of the greedy process.
        best_rank = max_rank + 1
        best_U, best_V, best_W = None, None, None
        
        for restart in range(50):
            if time.time() > time_limit:
                break
                
            residual = T.copy()
            U_cols = []
            V_cols = []
            W_cols = []
            
            for r in range(max_rank):
                if time.time() > time_limit:
                    break
                
                # Find best u, v, w
                # We sample random vectors from the discrete set
                # To be efficient, we don't iterate over all combinations (too many).
                # We use a stochastic approach: sample many random vectors and pick the best.
                
                best_val = -np.inf
                best_u, best_v, best_w = None, None, None
                
                # Number of random trials per rank step
                trials = 200
                
                for _ in range(trials):
                    # Generate random vectors with entries in {-1, -0.5, 0, 0.5, 1}
                    # We can bias towards non-zero to avoid trivial solutions
                    u = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_u)
                    v = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_v)
                    w = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_w)
                    
                    # Skip if all zeros
                    if np.all(u == 0) or np.all(v == 0) or np.all(w == 0):
                        continue
                    
                    # Compute contraction: sum_{i,j,k} residual[i,j,k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
                    # We can vectorize this.
                    # Reshape residual to (n, m, p)
                    # u to (n, m), v to (m, p), w to (p, n)
                    
                    u_r = u.reshape(n, m)
                    v_r = v.reshape(m, p)
                    w_r = w.reshape(p, n)
                    
                    # Contraction
                    # val = sum_{i,j,k} residual[i,j,k] * u_r[i,j] * v_r[j,k] * w_r[k,i]
                    # This is a 3-way tensor contraction.
                    # We can do it in steps:
                    # temp1[i,k] = sum_j residual[i,j,k] * u_r[i,j] * v_r[j,k]
                    # val = sum_{i,k} temp1[i,k] * w_r[k,i]
                    
                    # temp1 = np.einsum('ijk,ij,jk->ik', residual, u_r, v_r)
                    # val = np.einsum('ik,ki->', temp1, w_r)
                    
                    # For speed, we can use matrix multiplications if we reshape, but einsum is fine for small dims.
                    try:
                        temp1 = np.einsum('ijk,ij,jk->ik', residual, u_r, v_r)
                        val = np.einsum('ik,ki->', temp1, w_r)
                    except:
                        continue
                    
                    if val > best_val:
                        best_val = val
                        best_u = u.copy()
                        best_v = v.copy()
                        best_w = w.copy()
                
                if best_u is None:
                    break
                
                # Subtract the rank-1 component
                # We want to subtract best_val / (norm_sq) * component?
                # Actually, we want to minimize ||residual - alpha * u \otimes v \otimes w||^2
                # The optimal alpha is <residual, u \otimes v \otimes w> / ||u \otimes v \otimes w||^2
                # Since we are searching for exact integer/half-integer solutions, 
                # the coefficients in the decomposition are the entries of U, V, W.
                # The decomposition is sum_c U[:,c] \otimes V[:,c] \otimes W[:,c].
                # So we are adding the term U[:,c] \otimes V[:,c] \otimes W[:,c] to the sum.
                # We want T = sum_c ...
                # So we should subtract the best fit from the residual.
                
                # Calculate optimal alpha for this u,v,w
                # norm_sq = ||u \otimes v \otimes w||^2 = ||u||^2 * ||v||^2 * ||w||^2
                norm_sq = np.dot(u, u) * np.dot(v, v) * np.dot(w, w)
                if norm_sq < 1e-9:
                    continue
                
                alpha = best_val / norm_sq
                
                # The new columns for U, V, W are alpha * u, v, w? 
                # No, the term is U[:,c] * V[:,c] * W[:,c].
                # We can distribute alpha arbitrarily among u, v, w.
                # To keep entries small, we might want to put alpha in one of them.
                # However, we are checking for half-integers at the end.
                # Let's just store u, v, w and scale them later?
                # No, the decomposition is linear.
                # T = sum_c (U_c \otimes V_c \otimes W_c).
                # We found u, v, w that approximate the residual.
                # We set U_c = u, V_c = v, W_c = w * alpha?
                # Or we can just accumulate the sum.
                
                # Let's just subtract the best fit from the residual.
                # residual = residual - alpha * (u \otimes v \otimes w)
                
                u_r = u.reshape(n, m)
                v_r = v.reshape(m, p)
                w_r = w.reshape(p, n)
                
                # component[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
                # We can compute this via einsum
                component = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
                
                residual = residual - alpha * component
                
                # Store the columns
                # We store u, v, and alpha*w as the columns?
                # The term is u \otimes v \otimes (alpha * w)
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(alpha * w)
                
                # Check if residual is zero
                if np.allclose(residual, 0, atol=1e-9):
                    # We found an exact decomposition of rank r+1
                    # But we need to check if the entries are half-integers.
                    # The entries of U, V, W are u, v, alpha*w.
                    # u and v are half-integers.
                    # alpha might not be.
                    # We need to check if the final U, V, W matrices have half-integer entries.
                    
                    U_mat = np.column_stack(U_cols)
                    V_mat = np.column_stack(V_cols)
                    W_mat = np.column_stack(W_cols)
                    
                    if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                        if verify(U_mat, V_mat, W_mat):
                            return U_mat, V_mat, W_mat, r + 1
                    
                    # If not half-integer, we might still be close.
                    # But the greedy approach with discrete u,v,w doesn't guarantee half-integer alpha.
                    # This is a weakness.
                    # We can try to snap alpha*w to half-integers?
                    # If we snap, the residual might not be zero.
                    
                    # For now, if we reach zero residual, we check validity.
                    # If valid, return.
                    # If not, we continue searching? No, residual is zero, so no more terms needed.
                    # But if it's not half-integer, it's invalid for the task.
                    # We should probably restart or try a different alpha distribution.
                    
                    # Let's try to distribute alpha differently.
                    # We can scale u, v, w such that the product is the same, but entries are half-integers.
                    # This is a Diophantine-like problem.
                    # For now, we just return if valid.
                    
                    pass
            # End rank loop
        # End restart loop
        
        return None, None, None, max_rank + 1

    # Run greedy search
    U, V, W, rank = greedy_search(max_rank=28, time_limit=130.0)
    
    if U is not None:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    R = n * m * p
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
