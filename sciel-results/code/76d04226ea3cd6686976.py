"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # The tensor T[i,j,k] = delta[i,j,k] for matrix multiplication
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to verify that for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # and for all other combinations, it equals 0.
    
    # Actually, the matrix multiplication tensor has entries:
    # T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k] -> C[i,k]
    # The standard tensor is: T[i,j,k] = delta_{i,j,k} where the "1" entries are at positions
    # that correspond to valid (i,j,k) triples for the multiplication.
    
    # More precisely, for matrix multiplication of (n x m) * (m x p) -> (n x p):
    # The tensor has T[i, j, k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p)
    # Wait, no. The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) and 0 otherwise? 
    # No, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor is a 3-tensor T of shape (n*p, m, m) or similar.
    # Actually, the standard formulation: we have a tensor T of shape (n, m, p) where
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # But actually all (i,j,k) with valid indices contribute, so T[i,j,k] = 1 for all valid (i,j,k).
    # That can't be right because then the rank would just be n*m*p which is the schoolbook.
    
    # Let me re-read the problem. The tensor decomposition for matrix multiplication:
    # We want to write the multiplication as: C = A * B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The tensor T is of shape (n*p, m, m) or we can think of it as a 3-way tensor.
    # Actually, the standard way: the multiplication tensor is in the space 
    # (R^{n x p})^* \otimes R^{n x m} \otimes R^{m x p}
    # 
    # A decomposition of rank r means:
    # T = sum_{c=0}^{r-1} W_c \otimes U_c \otimes V_c
    # where U_c is in R^{n x m}, V_c is in R^{m x p}, W_c is in R^{n x p}
    # 
    # In the flattened form:
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Note: W[k*n+i, c] - the index is k*n + i, meaning the output C is stored in column-major? 
    # Or rather, the (i,k) entry of C is at position k*n + i.
    
    # The condition is: for all i in [0,n), k in [0,p):
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # In terms of the tensor:
    #   sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] summed over j gives C[i,k]
    #   But actually it's: for each c, the contribution is W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_k V[j*p+k,c] * B[j,k])
    #   No, that's not how it works.
    
    # Let me think more carefully. The decomposition says:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * ... 
    # No, that's not right.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # We have bilinear maps. The multiplication is a bilinear map:
    # (A, B) -> C where C = A*B
    # A is n x m, B is m x p, C is n x p.
    # 
    # The tensor T is in (R^{n*p}) \otimes (R^{n*m}) \otimes (R^{m*p})
    # T = sum_{i,k,j} e_{i,k} \otimes e_{i,j} \otimes e_{j,k}
    # where e_{i,k} is the basis vector for C[i,k], e_{i,j} for A[i,j], e_{j,k} for B[j,k].
    # 
    # A rank-r decomposition is:
    # T = sum_{c=0}^{r-1} w_c \otimes u_c \otimes v_c
    # where w_c is in R^{n*p}, u_c in R^{n*m}, v_c in R^{m*p}
    # 
    # This means: for any A, B:
    # C = sum_c (w_c^T C) * (u_c^T A) * (v_c^T B)
    # Wait, no. Let me use the standard formulation.
    
    # If T = sum_c w_c \otimes u_c \otimes v_c, then for inputs A (flattened) and B (flattened):
    # C = sum_c (w_c \cdot C_target) ... no.
    
    # Actually: the bilinear map is:
    # C = sum_c (u_c^T A) * (v_c^T B) * w_c
    # where A, B are flattened vectors, and C is the flattened output.
    # 
    # So: C_flat = sum_c (U[:,c]^T A_flat) * (V[:,c]^T B_flat) * W[:,c]
    # 
    # In matrix form: C_flat = W * diag(V^T B_flat) * (U^T A_flat) ... no.
    # C_flat = W * ( (U^T A_flat) * (V^T B_flat) ) where * is element-wise
    # 
    # So: C_flat[i] = sum_c W[i,c] * (sum_j U[j,c] * A_flat[j]) * (sum_k V[k,c] * B_flat[k])
    
    # For this to equal matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # With C_flat indexed by k*n + i (as per W[k*n+i, r]):
    # C_flat[k*n + i] = sum_j A[i,j] B[j,k]
    # 
    # And A_flat indexed by i*m + j: A_flat[i*m + j] = A[i,j]
    # B_flat indexed by j*p + k: B_flat[j*p + k] = B[j,k]
    
    # So we need: for all i, k:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j'*p+k', c] * B[j',k'])
    # = sum_j A[i,j] B[j,k]
    
    # This must hold for all A, B. So:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    # = sum_j A[i,j] B[j,k]
    
    # For this to be true for all A, B, we need:
    # The coefficient of A[i',j'] * B[j'',k''] on the LHS must equal 
    # delta_{i',i} * delta_{j',j''} * delta_{k'',k} on the RHS.
    
    # LHS coefficient of A[i',j'] * B[j'',k'']:
    # sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c]
    
    # So we need: for all i, k, i', j', j'', k'':
    # sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c] = delta_{i',i} * delta_{j',j''} * delta_{k'',k}
    
    # This is the exact condition. Let's verify with this.
    
    # For efficiency, let's construct the full tensor and check.
    # T[i, k, i', j', j'', k''] = sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c]
    # should equal delta_{i',i} * delta_{j',j''} * delta_{k'',k}
    
    # But this is a 6-dimensional tensor which is too large to check directly.
    # Instead, let's check by evaluating on random matrices.
    
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        A_flat = A.flatten()  # shape (n*m,), index i*m+j
        B_flat = B.flatten()  # shape (m*p,), index j*p+k
        
        # Compute C using the decomposition
        # C_flat[k*n + i] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A_flat[i*m+j]) * (sum_{j',k'} V[j'*p+k', c] * B_flat[j'*p+k'])
        
        # u_vals[c] = sum_j U[i*m+j, c] * A[i,j] for each i, c
        # Actually, u_vals[i, c] = sum_j U[i*m+j, c] * A[i,j]
        u_vals = np.zeros((n, rank))
        for i in range(n):
            for j in range(m):
                u_vals[i, :] += A[i, j] * U[i*m+j, :]
        
        v_vals = np.zeros((p, rank))
        for k in range(p):
            for j in range(m):
                v_vals[k, :] += B[j, k] * V[j*p+k, :]
        
        # C_flat[k*n + i] = sum_c W[k*n+i, c] * u_vals[i, c] * v_vals[k, c]
        C_flat = np.zeros(p * n)
        for k in range(p):
            for i in range(n):
                C_flat[k*n + i] = np.dot(W[k*n+i, :], u_vals[i, :] * v_vals[k, :])
        
        C_computed = C_flat.reshape(n, p, order='F')  # column-major: index k*n+i
        
        if not np.allclose(C_computed, C_true, atol=1e-9):
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal decomposition
    # Start with schoolbook rank n*m*p = 18
    # Try to reduce
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Schoolbook decomposition
    def schoolbook():
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
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    if _verify_decomposition(U, V, W, n, m, p):
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try rank 17, 16, ... down to some minimum
    # For small matrices, we can try random initializations and gradient descent
    # But gradient descent on non-convex problems is tricky.
    
    # Let's try a simpler approach: use the known fact that for 2x3x3, 
    # the rank is at most 18 (schoolbook). 
    # There might be a rank 17 or lower decomposition.
    
    # Try random search for lower rank
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute gradient
                # Loss: sum over all (i,k,i',j',j'',k'') of 
                # (sum_c W[k*n+i,c] U[i'*m+j',c] V[j''*p+k'',c] - delta_{i',i}delta_{j',j''}delta_{k'',k})^2
                # This is too expensive. Use random matrix evaluation instead.
                
                A = np.random.randn(n, m)
                B = np.random.randn(m, p)
                C_true = A @ B
                
                A_flat = A.flatten()
                B_flat = B.flatten()
                
                u_vals = np.zeros((n, target_rank))
                for i in range(n):
                    for j in range(m):
                        u_vals[i, :] += A[i, j] * U[i*m+j, :]
                
                v_vals = np.zeros((p, target_rank))
                for k in range(p):
                    for j in range(m):
                        v_vals[k, :] += B[j, k] * V[j*p+k, :]
                
                C_flat = np.zeros(p * n)
                for k in range(p):
                    for i in range(n):
                        C_flat[k*n + i] = np.dot(W[k*n+i, :], u_vals[i, :] * v_vals[k, :])
                
                C_computed = C_flat.reshape(n, p, order='F')
                error = np.sum((C_computed - C_true) ** 2)
                
                if error < 1e-6:
                    break
                
                # Gradient computation is complex; skip for now
                # Just do random perturbation
                U += lr * np.random.randn(n * m, target_rank)
                V += lr * np.random.randn(m * p, target_rank)
                W += lr * np.random.randn(p * n, target_rank)
            
            # Check if we found a valid decomposition
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = target_rank
                break
    
    # Return the best found
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
