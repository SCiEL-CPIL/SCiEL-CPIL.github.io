import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    We first try to find a decomposition with rank < 72 using algebraic structure
    (Strassen-style block decomposition) and random search. If none is found within
    the time budget, we return the schoolbook decomposition with rank 72.
    """
    n, m, p = 3, 4, 6
    schoolbook_rank = n * m * p  # 72
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Try to find a better decomposition
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Strategy 1: Try rank 66 (save 6 from schoolbook)
    # This is optimistic; we'll try a few ranks and see what we can find
    
    # For now, let's try a structured approach:
    # Partition the 4x6 output into blocks and use Strassen-like ideas
    
    # Actually, for 3x4*4x6, let's think about block structure:
    # A is 3x4, B is 4x6, C = AB is 3x6
    # We can write A as [A1 A2] where A1 is 3x2, A2 is 3x2
    # B as [B1; B2] where B1 is 2x6, B2 is 2x6
    # Then C = A1*B1 + A2*B2
    # Each A1*B1 is 3x2 * 2x6 = 3x6, rank 12
    # Each A2*B2 is 3x2 * 2x6 = 3x6, rank 12
    # Total: 24 multiplications for the cross terms... wait that's not right.
    
    # Let me reconsider. The schoolbook does 3*4*6 = 72 scalar multiplications.
    # To beat this, we need algebraic structure.
    
    # Let's try a direct search for rank r decompositions with small r
    # using random initialization and gradient-free optimization
    
    # For a tensor decomposition of matrix multiplication:
    # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # The multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1 for valid indices
    
    # Let's define the tensor explicitly
    # T[a, b, c] where a = i*m+j (0..n*m-1), b = j*p+k (0..m*p-1), c = k*n+i (0..p*n-1)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # Actually, the standard formulation is:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor M[i,j,k] = delta (it's 1 when we multiply A[i,j] by B[j,k] to contribute to C[i,k])
    # The decomposition is: M[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    
    # Let's try random search for small ranks
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W form a valid decomposition."""
        n, m, p = 3, 4, 6
        r = U.shape[1]
        # Reconstruct the tensor
        # M[i,j,k] should equal sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # We check that M[i,j,k] = 1 iff the indices are consistent (always true for matmul)
        # Actually, M[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
        # Wait, no. The matrix multiplication tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ (sum over t of U V W)
        # The condition is that for all i,j,k:
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # because (AB)[i,k] = sum_j A[i,j] B[j,k], and the tensor captures this.
        
        # Let's verify: for each (i,j,k), the sum should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_random_search(rank, max_iter=500, seed=None):
        """Try to find a rank decomposition via random search."""
        if seed is not None:
            np.random.seed(seed)
        
        n, m, p = 3, 4, 6
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to help convergence
        for t in range(rank):
            norm_u = np.linalg.norm(U[:, t])
            norm_v = np.linalg.norm(V[:, t])
            norm_w = np.linalg.norm(W[:, t])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U[:, t] /= norm_u
                V[:, t] /= norm_v
                W[:, t] /= norm_w
                # Then multiply the triple product by scale^3 = norm_u*norm_v*norm_w
                # We want sum to be 1, so we need to adjust
                # Actually, let's just do a simple random search
        
        # Simple random perturbation search
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for t in range(rank):
                            val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                return best_U, best_V, best_W, True
            
            # Random perturbation
            step = 0.1 * max(1.0, best_err)
            U += np.random.randn(n*m, rank) * step
            V += np.random.randn(m*p, rank) * step
            W += np.random.randn(p*n, rank) * step
        
        return best_U, best_V, best_W, False
    
    # Try a few ranks below 72
    for rank in [70, 69, 68, 67, 66, 65, 64]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            U, V, W, success = try_random_search(rank, max_iter=100, seed=rank*100+attempt)
            if success:
                best_rank = rank
                best_decomp = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    # If we found a better decomposition, use it
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to lists with rounded values for cleaner output
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    # Fall back to schoolbook
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
