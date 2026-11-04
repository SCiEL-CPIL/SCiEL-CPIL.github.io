"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check that the decomposition is exact
    # For matrix multiplication C = A @ B where A is n x m, B is m x p
    # The tensor T[i,j,k] = sum over rank of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i,j,k are valid indices (which they always are for valid matrix mult)
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # because (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's verify by checking that for random matrices, the reconstruction works
    # But that's expensive. Instead, let's verify the tensor equation directly.
    
    # The tensor for matrix multiplication n x m by m x p is:
    # T[i, j, k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
    # And the decomposition gives:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Let's check a few random instances
    rng = np.random.default_rng(42)
    for _ in range(20):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        # Reconstruct using the decomposition
        C_recon = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + 0, r] * 0  # This isn't right
                # Actually, let me think about this more carefully
                # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
                # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
                # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
                # The product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This should give 1 for the "coefficient" of A[i,j]*B[j,k] in C[i,k]
                # So C[i,k] = sum_j (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) * A[i,j]*B[j,k]
                # = sum_j 1 * A[i,j]*B[j,k] = (A@B)[i,k]
                pass
        
        # Let me just directly verify the tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using optimization
    # The matrix multiplication tensor for n x m by m x p has known lower bounds
    # For 2x4 by 4x5: the rank is at least max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Actually the Strassen-like bounds give better estimates
    # Let's try ranks from 20 upward
    
    # For small problems, we can try to find exact decompositions
    # using numerical optimization or known constructions
    
    # Known: for 2x2 by 2x2, rank 7 (Strassen)
    # For 2x3 by 3x2, rank 10 or 11?
    # For 2x4 by 4x5, let's try various ranks
    
    # Let's try a greedy/optimization approach for specific ranks
    # We'll use the fact that the tensor is multilinear in each factor
    
    # Try to find a rank-r decomposition for decreasing r
    # Start from a reasonable upper bound and go down
    
    # The trivial upper bound is n*m*p = 40
    # A better bound: min over factorizations... 
    # For n=2, m=4, p=5: 
    #   - Schoolbook: 40
    #   - We can try to exploit the small n=2
    
    # Let's try a different approach: since n=2 is small, we can use
    # the fact that we're multiplying 2x4 by 4x5.
    
    # Actually, let me think about this differently. The rank of the matrix
    # multiplication tensor [n,m,p] is the minimum number of multiplications
    # needed. For [2,4,5], this is not well-known to me exactly.
    
    # Let me try to construct a decomposition of rank 20 (which is m*p, the 
    # largest of the three "side products") using a clever scheme.
    
    # For n=2: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k]
    # This is just two independent dot products of length 4 with each row of B.
    # So we need 2*4*5 = 40 multiplications naively.
    # But we can share some work...
    
    # Actually for n=2, m=4, p=5, the problem is essentially:
    # Given A (2x4) and B (4x5), compute C (2x5).
    # C[0,:] = A[0,:] @ B  (dot product of 4-vectors)
    # C[1,:] = A[1,:] @ B
    # Each row of C is a 4x5 matrix-vector product, which needs 4*5=20 mults.
    # Total: 40 mults naively.
    
    # Can we do better? The two rows of A are different, so we can't directly
    # share multiplications between them unless we use Strassen-like tricks.
    
    # For a single dot product of length 4 (m=4), we need 4 multiplications.
    # For two dot products of length 4 with the same B columns but different A rows,
    # we need 8 multiplications per column of B, so 8*5=40 total.
    
    # Unless... we can use the structure of B. But B is arbitrary.
    
    # So for this specific problem, I think the rank might actually be 40,
    # or close to it. Let me try to see if any rank < 40 works.
    
    # Let me try rank 39, 38, ... using a simple optimization.
    
    # Actually, let me try a smarter approach. Since n=2 is very small,
    # let's parameterize the decomposition differently.
    
    # For rank r, we need:
    # U: 8 x r, V: 20 x r, W: 10 x r
    # such that for all i in {0,1}, j in 0..3, k in 0..4:
    # sum_{t=0}^{r-1} U[i*4+j, t] * V[j*5+k, t] * W[k*2+i, t] = 1
    
    # This is a system of 40 equations in 8r + 20r + 10r = 38r unknowns.
    # For r=40: 1520 unknowns, 40 equations -> very underdetermined
    # For r=20: 760 unknowns, 40 equations -> still very underdetermined
    
    # The constraint is that the tensor equation must hold exactly.
    # This is a set of polynomial equations (degree 3 in the unknowns if we
    # consider all variables, but actually it's bilinear in pairs).
    
    # Let me try a numerical approach: fix some variables and solve for others.
    
    # Strategy: Try to find a rank-r decomposition by iterative refinement.
    # Start with a random initialization and use gradient descent on the
    # residual sum of squares.
    
    def try_rank(r, max_iters=5000, tol=1e-12):
        """Try to find a rank-r decomposition using optimization."""
        if r >= best_rank:
            return None
        
        rng = np.random.default_rng(hash(r) % 2**32)
        
        # Initialize randomly
        U = rng.standard_normal((n*m, r)) * 0.1
        V = rng.standard_normal((m*p, r)) * 0.1
        W = rng.standard_normal((p*n, r)) * 0.1
        
        # Use Adam-like optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        best_loss = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            # Compute residual
            # Residual[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1
            # Gradient w.r.t. U[i*m+j, t]: V[j*p+k,t] * W[k*n+i,t] summed over j,k... 
            # Actually for each (i,j,k): d/dU[i*m+j,t] = V[j*p+k,t]*W[k*n+i,t]
            # But U[i*m+j,t] appears in multiple (i,j,k) terms (for different k)
            
            # Let's compute gradients efficiently
            # grad_U[i*m+j, t] = sum_k V[j*p+k, t] * W[k*n+i, t] * (residual[i,j,k])
            # But residual[i,j,k] = sum_t' U[i*m+j,t']*V[j*p+k,t']*W[k*n+i,t'] - 1
            
            # This is getting complex. Let me use a simpler approach.
            
            # Compute all residuals
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for tt in range(r):
                            val += U[i*m+j, tt] * V[j*p+k, tt] * W[k*n+i, tt]
                        residual[i, j, k] = val - 1.0
            
            loss = np.sum(residual**2)
            if loss < best_loss:
                best_loss = loss
                if loss < tol:
                    # Found a solution!
                    return (U.copy(), V.copy(), W.copy())
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = residual[i, j, k]
                        for tt in range(r):
                            grad_U[i*m+j, tt] += res * V[j*p+k, tt] * W[k*n+i, tt]
                            grad_V[j*p+k, tt] += res * U[i*m+j, tt] * W[k*n+i, tt]
                            grad_W[k*n+i, tt] += res * U[i*m+j, tt] * V[j*p+k, tt]
            
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U**2
            U -= lr_t * m_U / (np.sqrt(v_U) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V**2
            V -= lr_t * m_V / (np.sqrt(v_V) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W**2
            W -= lr_t * m_W / (np.sqrt(v_W) + eps)
        
        # Check if we found a good solution
        if best_loss < 1e-8:
            return (U, V, W)
        return None
    
    # Try various ranks
    # The theoretical minimum is at least max(n*m, m*p, n*p) but that's not quite right
    # For matrix multiplication [n,m,p], the rank is at least n*m, m*p, and n*p
    # Actually it's at least the max of those, but there are better bounds.
    # For [2,4,5]: max(8, 20, 10) = 20
    
    # Let's try from 20 up to 39
    for r in range(20, min(40, best_rank)):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=2000)
        if result is not None:
            U_opt, V_opt, W_opt = result
            # Verify
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                best_rank = r
                best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                break
    
    return best_result
# EVOLVE-BLOCK-END
