"""Optimized matrix multiplication tensor decomposition for 2x3x4."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all (i,j,k) in the multiplication pattern
    # Actually, the multiplication tensor is: T[i,j,k] = delta_{i,k} is wrong
    # The correct tensor: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entries are: T[i,j,k] = 1 for all i,j,k (it's the identity in the bilinear form)
    # Wait, no. The tensor is defined such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # This is because (A*B)[i,k] = sum_j A[i,j]*B[j,k], and we want this to equal
    # sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Hmm, let me reconsider.
    
    # The standard formulation: We want to compute C = A @ B where A is n×m, B is m×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Tensor decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j*p+k', c] * B[j,k']) * W[k*n+i, c]
    # Wait, that's not right either. Let me use the standard trace convention.
    #
    # With W[k*n+i, r], the decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    #
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] summed over r equals delta_{j,j} = 1 for the matching j
    # Actually: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all (i,j,k)
    # No wait, that's not right. Let me think again.
    #
    # The bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_r f_r(A) * g_r(B) * h_r(C_position)
    # where f_r(A) = sum_j U[i*m+j, r] A[i,j], g_r(B) = sum_j V[j*p+k, r] B[j,k]
    # and h_r is just W[k*n+i, r]
    #
    # So: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k])
    # = sum_j A[i,j] B[j,k]
    #
    # This requires: for all (i,j,k), sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    # Wait no, it should be that the coefficient of A[i,j]*B[j,k] is 1, and coefficients of A[i,j']*B[j',k] for j'≠j or other combinations are 0.
    #
    # Actually the correct condition is:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 if j=j (always true in this context)
    # No, the tensor entry T[i,j,k] should equal 1 for the "diagonal" terms.
    #
    # Let me use the standard: The multiplication tensor has T[i,j,k] = 1 for all i,j,k,
    # and we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k] = 1 for all (i,j,k).
    #
    # Wait, that can't be right because then we'd just need rank 1.
    # 
    # I think I'm confusing myself. Let me look at this differently.
    #
    # The matrix multiplication tensor M_{n,m,p} is a tensor in (C^{n*m})^* ⊗ (C^{m*p})^* ⊗ (C^{n*p})
    # where M[(i,j), (j',k), (i',k')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    #
    # So M[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    #
    # In our indexing: U has rows indexed by (i,j) with i in [0,n), j in [0,m)
    # V has rows indexed by (j',k) with j' in [0,m), k in [0,p)
    # W has rows indexed by (i',k') with i' in [0,n), k' in [0,p)
    #
    # The condition is: sum_r U[(i,j), r] * V[(j',k), r] * W[(i',k'), r] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    #
    # In the trace convention W[k*n+i, r], the row index for W is k*n+i where i is the row index and k is the column index of C.
    # So W[(i',k'), r] corresponds to W[k'*n + i', r].
    #
    # So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    # But wait, i and i' are the same variable? No, let me use different symbols.
    #
    # Let's say: for all (a, b, c) where a in [0,n), b in [0,m), c in [0,p):
    #   sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] = 1
    # And for all other combinations (a', b', c) with a'≠a or b'≠b:
    #   sum_r U[a'*m+b', r] * V[b'*p+c, r] * W[c*n+a, r] = 0
    #
    # Hmm, this is getting complicated. Let me just use the standard formulation.
    #
    # The tensor T is defined by T[i,j,k] = 1 for all (i,j,k) in the "matching" positions.
    # Specifically, T[(i,j), (j',k), (i',k')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    #
    # In flattened form: T[U_idx, V_idx, W_idx] where:
    # U_idx = i*m + j, V_idx = j'*p + k, W_idx = k'*n + i'
    # T[U_idx, V_idx, W_idx] = 1 if i=i', j=j', k=k' else 0
    #
    # So for a given (U_idx, V_idx, W_idx):
    # i = U_idx // m, j = U_idx % m
    # j' = V_idx // p, k = V_idx % p
    # i' = W_idx % n, k' = W_idx // n
    # T = 1 if i==i' and j==j' and k==k' else 0

    def build_target_tensor():
        """Build the target tensor T of shape (n*m, m*p, p*n)"""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_idx = i * m + j
                    V_idx = j * p + k
                    W_idx = k * n + i
                    T[U_idx, V_idx, W_idx] = 1.0
        return T
    
    T = build_target_tensor()
    
    best_rank = n * m * p  # Schoolbook fallback
    best_factors = None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(6, 12, 8) = 12
    # Known best for 2x3x4 is 20
    min_rank = max(n * m, m * p, n * p)
    
    for rank in range(min_rank, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        n_restarts = 50 if rank <= 22 else 20
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(restart * 1000 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS optimization
            max_iters = 500
            tol = 1e-12
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row i, solve least squares
                for idx in range(n * m):
                    # T[idx, :, :] = sum_r U[idx, r] * V[:, r] * W[:, r] (outer product)
                    # We want: sum_r U[idx, r] * (V[:, r] * W[:, r]) = T[idx, :, :]
                    # This is: U[idx, :] @ (V * W)^T flattened... 
                    # Let's form the matrix: M[r, s] where s indexes (V_idx, W_idx)
                    # M = np.einsum('vr,wr->r,s', V, W) doesn't work directly
                    # Better: for each r, the contribution is V[:, r] * W[:, r] (element-wise)
                    # So the "matrix" for this row is: A[r, (v_idx, w_idx)] = V[v_idx, r] * W[w_idx, r]
                    # We want: U[idx, r] such that sum_r U[idx, r] * A[r, s] = T[idx, s]
                    # This is a linear system: A @ U[idx, :] = T[idx, :].flatten()
                    
                    # A has shape (rank, m*p * p*n)
                    # A[r, v_idx * p*n + w_idx] = V[v_idx, r] * W[w_idx, r]
                    A = np.einsum('vr,wr->r,vw', V, W)  # shape (rank, m*p, p*n)
                    A_flat = A.reshape(rank, -1)  # shape (rank, m*p*p*n)
                    target = T[idx, :, :].flatten()
                    
                    # Solve: A_flat.T @ U[idx, :] = target
                    # Or: U[idx, :] = pinv(A_flat) @ target
                    try:
                        U[idx, :] = np.linalg.lstsq(A_flat.T, target, rcond=None)[0]
                    except:
                        U[idx, :] = np.random.randn(rank) * 0.1
                
                if time.time() > deadline - 5:
                    break
                
                # Update V
                for idx in range(m * p):
                    A = np.einsum('ur,wr->r,uw', U, W)  # shape (rank, n*m, p*n)
                    A_flat = A.reshape(rank, -1)
                    target = T[:, idx, :].flatten()
                    try:
                        V[idx, :] = np.linalg.lstsq(A_flat.T, target, rcond=None)[0]
                    except:
                        V[idx, :] = np.random.randn(rank) * 0.1
                
                if time.time() > deadline - 5:
                    break
                
                # Update W
                for idx in range(p * n):
                    A = np.einsum('ur,vr->r,uv', U, V)  # shape (rank, n*m, m*p)
                    A_flat = A.reshape(rank, -1)
                    target = T[:, :, idx].flatten()
                    try:
                        W[idx, :] = np.linalg.lstsq(A_flat.T, target, rcond=None)[0]
                    except:
                        W[idx, :] = np.random.randn(rank) * 0.1
                
                # Check convergence
                if it % 10 == 0:
                    error = np.abs(np.einsum('ir,jr,kr->ijk', U, V, W) - T)
                    if np.max(error) < tol:
                        found = True
                        break
            
            if found:
                best_rank = rank
                # Convert to list format
                best_factors = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found:
            break
    
    # Fallback to schoolbook if no better found
    if best_factors is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_factors = {"U": U, "V": V, "W": W}
    
    return best_factors
