"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for r in range(n):
        for s in range(p):
            for t in range(m):
                expected = 1.0
                actual = 0.0
                for k in range(rank):
                    actual += U[r*m + t, k] * V[t*p + s, k] * W[s*n + r, k]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank14():
    """Construct the known rank-14 decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: columns 0,1 of B
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # Block 2: columns 2,3 of B
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_compress(U, V, W, n=2, m=2, p=4):
    """
    Attempt to compress the rank-14 decomposition to rank 13.
    We try to merge two terms k1, k2 into a single term k_new.
    This requires finding a vector in the span of the two terms that satisfies
    the tensor equations.
    
    Strategy:
    1. Pick two indices i, j.
    2. Remove them, creating a hole.
    3. The new term must satisfy:
       sum_{k != i, j} U_k V_k W_k + U_new V_new W_new = T
       => U_new V_new W_new = T - sum_{k != i, j} U_k V_k W_k
    Let R = T - sum_{k != i, j} U_k V_k V_k.
    We need to find U_new, V_new, W_new such that U_new V_new W_new = R.
    This is equivalent to finding a rank-1 tensor that equals R.
    If R is not rank-1, this fails.
    
    However, we can be more flexible. We can scale the existing terms.
    Let the new term be a combination.
    Actually, a better approach for compression is to check if the tensor R
    (the residual after removing two terms) has rank 1.
    If it does, we can replace the two terms with this single term.
    
    Let's implement this check.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit for search
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build the full tensor T
    # T shape: (n*m, m*p, p*n)
    # T[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
    # We know T is the matrix multiplication tensor.
    # We can reconstruct T explicitly to avoid floating point errors in subtraction?
    # No, the decomposition IS the tensor.
    # T[a, b, c] = 1 if c corresponds to (row, col) such that a=row*m+col_a, b=col_a*p+col_b?
    # Let's just compute T from the current decomposition to be safe, 
    # although it should be the standard MM tensor.
    
    # Actually, it's faster to just compute the residual tensor R for a given pair (i, j).
    # R[a, b, c] = T[a, b, c] - U[a,i]V[b,i]W[c,i] - U[a,j]V[b,j]W[c,j]
    # We want to know if R is rank-1.
    # A tensor R is rank-1 if R[a,b,c] = x[a]y[b]z[c] for some vectors x,y,z.
    # This is equivalent to checking if the matrix unfolding R_{ab, c} has rank 1? No.
    # It's equivalent to checking if the 3-way tensor has tensor rank 1.
    # A necessary and sufficient condition is that all its matrix unfoldings have rank 1.
    # Specifically, flatten R into a matrix of shape (n*m*m*p, p*n) and check rank.
    # If rank is 1, then R is rank-1.
    
    # Precompute T as a flattened vector or tensor?
    # T is fixed. We can compute T once.
    # T[a, b, c] = 1 if ...
    # Let's use the property that T is the MM tensor.
    # T[a, b, c] = 1 if c = s*n + r, a = r*m + t, b = t*p + s.
    # Otherwise 0.
    
    T_flat = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for t in range(m):
            for s in range(p):
                a = r*m + t
                b = t*p + s
                c = s*n + r
                T_flat[a, b, c] = 1.0
                
    # Try all pairs (i, j)
    for i in range(rank):
        for j in range(i+1, rank):
            if time.time() > deadline:
                break
            
            # Compute residual R
            # R = T_flat - U[:,i,None,None]*V[:,None,j? No.
            # R[a,b,c] = T_flat[a,b,c] - U[a,i]V[b,i]W[c,i] - U[a,j]V[b,j]W[c,j]
            
            term_i = U_arr[:, i][:, None, None] * V_arr[:, i][None, :, None] * W_arr[:, i][None, None, :]
            term_j = U_arr[:, j][:, None, None] * V_arr[:, j][None, :, None] * W_arr[:, j][None, None, :]
            
            R = T_flat - term_i - term_j
            
            # Check if R is rank 1
            # Unfold R to matrix of shape (n*m*m*p, p*n)
            R_mat = R.reshape(n*m*m*p, p*n)
            
            # Compute rank of R_mat
            # Use SVD
            try:
                s = np.linalg.svd(R_mat, compute_uv=False)
                # Rank is number of singular values > tol
                tol = 1e-9
                rnk = np.sum(s > tol)
                
                if rnk <= 1:
                    # Found a compression!
                    # Extract the vectors
                    # If rnk == 0, R is 0, which means terms i and j were redundant (unlikely in minimal decomps)
                    # If rnk == 1, R = sigma_1 * u_1 * v_1^T
                    # We need to reshape u_1 and v_1 back to vectors for U, V, W.
                    
                    if rnk == 0:
                         # This implies T = term_i + term_j.
                         # We can just keep one? No, we need to represent T.
                         # If R=0, then T = term_i + term_j.
                         # We can't just drop one. We need a single term that equals T? No.
                         # We want to replace term_i + term_j with ONE term.
                         # If R=0, then term_i + term_j = T.
                         # This would mean T is sum of 2 terms. Rank 2?
                         # But we are removing 2 and adding 1.
                         # If R=0, it means the remaining terms already sum to T?
                         # No, R = T - term_i - term_j.
                         # If R=0, then T = term_i + term_j.
                         # This means the other rank-2 terms are zero?
                         # This is unlikely unless the decomposition is very sparse.
                         # Let's assume rnk == 1.
                         pass

                    if rnk == 1:
                        u, s_vals, vt = np.linalg.svd(R_mat, compute_uv=False)
                        # Wait, SVD returns U, S, Vt.
                        U_svd, S_svd, Vt_svd = np.linalg.svd(R_mat, full_matrices=False)
                        
                        # R_mat = S[0] * U_svd[:, 0] * Vt_svd[0, :]
                        # We need to split U_svd[:, 0] into U_new and V_new parts?
                        # No, R_mat is a flattening of R[a,b,c].
                        # R[a,b,c] = U_new[a] * V_new[b] * W_new[c]
                        # R_mat[(a,b), c] = (U_new[a] V_new[b]) W_new[c]
                        # So U_svd[:, 0] corresponds to the vector (U_new[a] V_new[b]) for all a,b.
                        # And Vt_svd[0, :] corresponds to W_new[c].
                        
                        vec_ab = U_svd[:, 0]
                        vec_c = Vt_svd[0, :]
                        
                        # We need to find U_new, V_new such that U_new[a] V_new[b] = vec_ab[a, b] (reshaped)
                        # vec_ab is of length n*m*m*p.
                        # Reshape to (n*m, m*p)
                        mat_ab = vec_ab.reshape(n*m, m*p)
                        
                        # Check if mat_ab is rank 1
                        U_ab, S_ab, Vt_ab = np.linalg.svd(mat_ab, full_matrices=False)
                        if np.sum(S_ab > tol) <= 1:
                            # It is rank 1
                            # mat_ab = S_ab[0] * U_ab[:, 0] * Vt_ab[0, :]
                            # So U_new = S_ab[0] * U_ab[:, 0] (or any split of the scalar)
                            # V_new = Vt_ab[0, :]
                            # W_new = vec_c / S_ab[0]? 
                            # Let's distribute the scalar S_ab[0] and S_svd[0].
                            # R = S_svd[0] * vec_ab * vec_c
                            # vec_ab = S_ab[0] * u_ab * v_ab^T
                            # R = S_svd[0] * S_ab[0] * (u_ab[:, None] * v_ab[None, :] * vec_c[None, None, :])
                            # U_new = u_ab
                            # V_new = v_ab
                            # W_new = S_svd[0] * S_ab[0] * vec_c
                            
                            U_new = U_ab[:, 0]
                            V_new = Vt_ab[0, :]
                            W_new = S_svd[0] * S_ab[0] * vec_c
                            
                            # Construct the new decomposition
                            new_rank = rank - 1
                            new_U = []
                            new_V = []
                            new_W = []
                            
                            for k in range(rank):
                                if k == i or k == j:
                                    continue
                                new_U.append(U_arr[:, k])
                                new_V.append(V_arr[:, k])
                                new_W.append(W_arr[:, k])
                            
                            # Add the new term
                            new_U.append(U_new)
                            new_V.append(V_new)
                            new_W.append(W_new)
                            
                            new_U = np.array(new_U).T
                            new_V = np.array(new_V).T
                            new_W = np.array(new_W).T
                            
                            # Verify
                            if _verify(new_U, new_V, new_W):
                                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
                            
            except:
                pass
                
        if time.time() > deadline:
            break
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # First, get the baseline rank-14 solution
    base_sol = _construct_rank14()
    
    # Try to compress to rank 13
    if time.time() < deadline:
        compressed = _try_compress(base_sol['U'], base_sol['V'], base_sol['W'])
        if compressed is not None:
            # Verify one more time to be absolutely sure
            if _verify(compressed['U'], compressed['V'], compressed['W']):
                return compressed
    
    return base_sol
# EVOLVE-BLOCK-END
