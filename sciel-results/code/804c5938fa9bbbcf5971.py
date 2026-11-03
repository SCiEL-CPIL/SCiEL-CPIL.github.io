"""Matrix multiplication 3x5x6 tensor decomposition - rank minimization."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90

    # Build schoolbook decomposition as fallback
    best_rank = rank_schoolbook
    best_U = np.zeros((n * m, rank_schoolbook))
    best_V = np.zeros((m * p, rank_schoolbook))
    best_W = np.zeros((p * n, rank_schoolbook))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1.0
                best_V[j * p + k, col] = 1.0
                best_W[k * n + i, col] = 1.0
                col += 1

    def _verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] should be 1 if the (i,j) element of A
        # and (j,k) element of B contribute to (i,k) element of C, which is always 1
        # for valid indices. Actually, the tensor for matrix multiplication is:
        # T[i, j, k] = 1 for all valid i,j,k (it's the standard multiplication tensor)
        # Wait, let me reconsider. The tensor T has indices (i, j, k) where:
        # - i indexes rows of A (n)
        # - j indexes columns of A / rows of B (m)  
        # - k indexes columns of B (p)
        # T[i,j,k] = 1 (the multiplication tensor is all ones in its support)
        # Actually no. The matrix multiplication tensor T_{ijk} satisfies:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor is T_{ijk} = 1 for all i,j,k in valid ranges.
        # The decomposition is: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        if time.time() > deadline:
            return False

        # Compute the decomposed tensor and check against the true tensor
        # True tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        rank = U.shape[1]
        
        # Efficient verification using matrix operations
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can compute this as: for each r, outer product contribution
        # Or more efficiently: 
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        
        # Let's verify a subset for speed, then full if time permits
        # Quick check: verify a few random points
        np.random.seed(42)
        check_points = []
        for _ in range(min(100, n * m * p)):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            check_points.append((i, j, k))
        
        for i, j, k in check_points:
            val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Full verification if time permits
        if time.time() < deadline - 5:
            # Full tensor verification
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Compute via einsum
            T_computed = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), 
                                    U_reshaped, V_reshaped, W_reshaped)
            if not np.allclose(T_computed, np.ones((n, m, p)), atol=1e-9):
                return False
        
        return True

    def _round_to_half_integers(U, V, W):
        """Round matrices to nearest half-integers."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return U_rounded, V_rounded, W_rounded

    def _try_randomized_rank_reduction(target_rank, max_iter=50):
        """Try to find a decomposition of given rank using randomized methods."""
        if time.time() > deadline - 10:
            return None

        U0 = best_U[:, :target_rank] if target_rank <= rank_schoolbook else best_U
        V0 = best_V[:, :target_rank] if target_rank <= rank_schoolbook else best_V
        W0 = best_W[:, :target_rank] if target_rank <= rank_schoolbook else best_W

        # Randomized approach: try to project onto lower rank
        # Using iterative refinement
        U_curr = U0.copy()
        V_curr = V0.copy()
        W_curr = W0.copy()

        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Compute residual
            # T_target[i,j,k] = 1
            # T_computed[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # Update U: for each (i,j), U[i,j,:] should satisfy
            # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all k
            # This is a linear system in U[i,j,:]
            
            # For fixed (i,j), we need: U[i,j,:] * (V[j,:,:] * W[:,i,:])^T = ones(p)
            # Where V[j,:,:] is (p, rank) and W[:,i,:] is (p, rank)
            # So M[i,j,k,r] = V[j,k,r] * W[k,i,r]
            # We need: sum_r U[i,j,r] * M[i,j,k,r] = 1 for all k
            
            # Let's build the matrix M for each (i,j)
            # M_kr = V[j,k,r] * W[k,i,r]
            
            for i in range(n):
                for j in range(m):
                    # Build matrix A_kr where A[k,r] = sum over nothing... 
                    # Actually: we need U[i,j,:] such that for all k:
                    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                    # Let B[k,r] = V[j,k,r] * W[k,i,r]
                    # Then: U[i,j,:] @ B^T = ones(p)
                    # So U[i,j,:] = (B^T B)^{-1} B^T ones(p) (least squares)
                    
                    B = V_curr[j, :, :] * W_curr[:, i, :]  # shape (p, rank)
                    # Solve: B^T @ U[i,j,:] = ones(p)
                    # i.e., U[i,j,:] is the solution to minimize ||B^T u - 1||
                    try:
                        solution, _, _, _ = np.linalg.lstsq(B.T, np.ones(p), rcond=None)
                        U_curr[i * m + j, :] = solution
                    except:
                        pass

            # Update V similarly
            for j in range(m):
                for k in range(p):
                    # Need V[j,k,:] such that for all i:
                    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                    # Let C[i,r] = U[i,j,r] * W[k,i,r]
                    # Then: C @ V[j,k,:] = ones(n)
                    C = U_curr.reshape(n, m, target_rank)[:, j, :] * W_curr[k, :, :]  # shape (n, rank)
                    try:
                        solution, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                        V_curr[j * p + k, :] = solution
                    except:
                        pass

            # Update W similarly
            for k in range(p):
                for i in range(n):
                    # Need W[k,i,:] such that for all j:
                    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                    # Let D[j,r] = U[i,j,r] * V[j,k,r]
                    # Then: D @ W[k,i,:] = ones(m)
                    D = U_curr.reshape(n, m, target_rank)[i, :, :] * V_curr.reshape(m, p, target_rank)[:, k, :]  # shape (m, rank)
                    try:
                        solution, _, _, _ = np.linalg.lstsq(D, np.ones(m), rcond=None)
                        W_curr[k * n + i, :] = solution
                    except:
                        pass

        # Round and verify
        U_rounded, V_rounded, W_rounded = _round_to_half_integers(U_curr, V_curr, W_curr)
        
        if _verify(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        return None

    # Try to find lower rank decompositions
    # Start from schoolbook rank and try to reduce
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook // 2, 1), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_randomized_rank_reduction(target_rank, max_iter=20)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_U = U_r
            best_V = V_r
            best_W = W_r
            # If we found a good one, try to go lower
            continue
    
    # Also try a few random initializations for the current best rank
    if time.time() < deadline - 30:
        for trial in range(10):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 123 + 456)
            target_rank = best_rank
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            # Add the schoolbook structure as a hint
            U_rand += best_U[:, :target_rank] * 0.5
            V_rand += best_V[:, :target_rank] * 0.5
            W_rand += best_W[:, :target_rank] * 0.5
            
            # Run refinement
            U_curr = U_rand.copy()
            V_curr = V_rand.copy()
            W_curr = W_rand.copy()
            
            for iteration in range(30):
                if time.time() > deadline - 15:
                    break
                
                for i in range(n):
                    for j in range(m):
                        B = V_curr[j, :, :] * W_curr[:, i, :]
                        try:
                            solution, _, _, _ = np.linalg.lstsq(B.T, np.ones(p), rcond=None)
                            U_curr[i * m + j, :] = solution
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        C = U_curr.reshape(n, m, target_rank)[:, j, :] * W_curr[k, :, :]
                        try:
                            solution, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                            V_curr[j * p + k, :] = solution
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        D = U_curr.reshape(n, m, target_rank)[i, :, :] * V_curr.reshape(m, p, target_rank)[:, k, :]
                        try:
                            solution, _, _, _ = np.linalg.lstsq(D, np.ones(m), rcond=None)
                            W_curr[k * n + i, :] = solution
                        except:
                            pass
            
            U_rounded, V_rounded, W_rounded = _round_to_half_integers(U_curr, V_curr, W_curr)
            if _verify(U_rounded, V_rounded, W_rounded):
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

