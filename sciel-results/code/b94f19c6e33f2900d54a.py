"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Build the matrix multiplication tensor
    2. Use SVD to find the numerical rank (lower bound)
    3. Try to construct a valid decomposition at that rank with integer/half-integer entries
    4. If needed, incrementally increase rank and use local search
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (all entries are 1 in the standard formulation)
    # 
    # Decomposition: T[i,j,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Wait, that's not right. Let me reconsider.
    #
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1, and we need:
    # sum_r U[i,j,r] * V[j,k,r] = W[i,k,r] ... no
    #
    # Actually the correct formulation:
    # We want: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] = delta_{i,i'}*delta_{k,k'}
    # 
    # No. The standard formulation is:
    # The tensor T is a 3-tensor with T[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes columns of A / rows of B: 0..m-1  
    # - k indexes columns of B (and C): 0..p-1
    #
    # T[i,j,k] = 1 for all valid i,j,k
    #
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    #
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]... 
    #
    # No wait. The bilinear form interpretation:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_k' B[j,k'] V[j*p+k',r]) * W[k*n+i,r]
    #
    # This doesn't quite work. Let me use the standard definition:
    #
    # A tensor T of format (n*m, m*p, p*n) can be decomposed as:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # For matrix multiplication, T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if i=i' and k=k' and j=j', else 0
    #
    # So T[(i,j), (j,k), (k,i)] = 1, and T is zero otherwise.
    
    # Build the tensor as a 3D array: T[a, b, c]
    # a = i*m + j for i in [0,n), j in [0,m)
    # b = j*p + k for j in [0,m), k in [0,p)
    # c = k*n + i for k in [0,p), i in [0,n)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now find the minimal rank decomposition
    # Reshape T to a 2D matrix for SVD: flatten two modes, keep third
    # T reshaped as (nm*mp, pn) or (nm, mp*pn) etc.
    
    # Try different unfoldings to get rank estimates
    T_mat1 = T.reshape(nm * mp, pn)
    T_mat2 = T.reshape(nm, mp * pn)
    T_mat3 = T.reshape(mp * pn, nm)
    
    # Compute SVD to estimate rank
    try:
        s1 = np.linalg.svd(T_mat1, compute_uv=False)
        s2 = np.linalg.svd(T_mat2, compute_uv=False)
        s3 = np.linalg.svd(T_mat3, compute_uv=False)
        
        # Numerical rank: count singular values above tolerance
        tol = 1e-10
        rank1 = np.sum(s1 > tol)
        rank2 = np.sum(s2 > tol)
        rank3 = np.sum(s3 > tol)
        
        estimated_rank = max(rank1, rank2, rank3)
    except Exception:
        estimated_rank = nm * mp * pn  # fallback to schoolbook
    
    estimated_rank = max(1, int(estimated_rank))
    
    # The schoolbook rank is n*m*p = 84
    schoolbook_rank = n * m * p
    
    # Start from estimated_rank and try to find a valid decomposition
    # For exact decompositions with integer/half-integer entries,
    # we need to be careful about the rounding.
    
    # First, try to get a decomposition at the estimated rank using SVD
    best_result = None
    
    # Try ranks from estimated_rank up to schoolbook_rank
    # But be mindful of time
    max_rank_to_try = min(schoolbook_rank, estimated_rank + 20)
    
    for rank in range(estimated_rank, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        try:
            # Get SVD decomposition at this rank
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat1, full_matrices=False)
            
            # Truncate to desired rank
            U_trunc = U_svd[:, :rank] * s_vals[:rank]
            Vt_trunc = Vt_svd[:rank, :]
            
            # Reshape back
            U_3d = U_trunc.reshape(nm, rank)
            V_3d = Vt_trunc.reshape(mp, rank)
            
            # Now we need W such that T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This means: for each c, T[:, :, c] = U * diag(W[c,:]) * V^T
            # Or: T[:, :, c] = U * (W[c, :] * V^T) where * is element-wise
            # 
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # So T[:, :, c] = U @ diag(W[c,:]) @ V.T
            # 
            # We can solve for W by: for each c, T[:,:,c] = U @ diag(W[c]) @ V.T
            # => T[:,:,c] @ V = U @ diag(W[c]) @ V.T @ V
            # If V.T @ V is invertible, we can solve
            #
            # Alternatively, use the fact that:
            # T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
            # Let M = U * V (element-wise product), shape (nm, mp, rank) reshaped...
            # Actually M[r] = U[:,r] * V[:,r] reshaped to (nm*mp,)
            # Then T[:,:,c] reshaped to (nm*mp,) = M @ W[c,:]
            
            # Build M: M[r] = (U[:,r] * V[:,r]).flatten()
            M = np.zeros((nm * mp, rank))
            for r in range(rank):
                M[:, r] = (U_3d[:, r] * V_3d[:, r]).flatten()
            
            # For each c, solve: M @ W[c,:] = T[:,:,c].flatten()
            W_3d = np.zeros((pn, rank))
            for c in range(pn):
                target = T[:, :, c].flatten()
                # Solve least squares: M @ w = target
                w, residuals, rank_ls, sv = np.linalg.lstsq(M, target, rcond=None)
                W_3d[c, :] = w
            
            # Now verify the decomposition
            # Check: sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c]
            error = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = np.sum(U_3d[a, :] * V_3d[b, :] * W_3d[c, :])
                        error += abs(val - T[a, b, c])
            
            if error < 1e-8:
                # Valid decomposition! Now try to round to integer/half-integer
                U_rounded = np.round(U_3d * 2) / 2
                V_rounded = np.round(V_3d * 2) / 2
                W_rounded = np.round(W_3d * 2) / 2
                
                # Verify rounded version
                error_rounded = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                            error_rounded += abs(val - T[a, b, c])
                
                if error_rounded < 1e-8:
                    # Found a valid integer/half-integer decomposition
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                else:
                    # Rounding didn't work, try to adjust
                    # Use the unrounded values but snap to nearest half-integer
                    # and verify
                    pass
            
            # If SVD-based approach didn't give a clean result, 
            # try the schoolbook decomposition at this rank if rank >= schoolbook_rank
            if rank >= schoolbook_rank:
                # Use schoolbook
                U_sb = [[0] * rank for _ in range(nm)]
                V_sb = [[0] * rank for _ in range(mp)]
                W_sb = [[0] * rank for _ in range(pn)]
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_sb[i * m + j][col] = 1
                            V_sb[j * p + k][col] = 1
                            W_sb[k * n + i][col] = 1
                            col += 1
                
                # Verify
                error_sb = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = sum(U_sb[a][r] * V_sb[b][r] * W_sb[c][r] for r in range(rank))
                            error_sb += abs(val - T[a, b, c])
                
                if error_sb < 1e-10:
                    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
                    # This is valid, but we want minimal rank, so keep trying
                    break
        
        except Exception:
            continue
    
    # If we found a schoolbook decomposition, return it
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
    U_sb = [[0] * rank for _ in range(nm)]
    V_sb = [[0] * rank for _ in range(mp)]
    W_sb = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
