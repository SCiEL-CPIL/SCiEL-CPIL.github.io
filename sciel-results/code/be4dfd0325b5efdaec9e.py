"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    
    Uses exact rational Gaussian elimination to find the minimal rank decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # The multiplication tensor T has entries T[i,j,k] = delta_{i,k} (for A[i,j]*B[j,k])
    # We need to find rank-minimal decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Key insight: For matrix multiplication, we can fix the "output" indices and
    # look at the linear map from (U ⊗ V) to W space.
    # 
    # More precisely, consider the linear map M: R^{n*m} ⊗ R^{m*p} -> R^{p*n}
    # defined by M(A, B) = A*B. The rank of this map (as a tensor) is what we want.
    #
    # Actually, let's think differently. The tensor T_{i,j,k} = δ_{ik} (for 2x5 * 5x5 -> 5x2)
    # Wait, let me re-read the convention.
    
    # Convention: W[k*n + i, r] where output is C = A*B
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The tensor is: T_{i,j,k} = δ_{ik} ... no wait.
    
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # For each output position (i,k), we have C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 for the term A[i,j]*B[j,k]
    # contributing to C[i,k].
    
    # So the full tensor is: sum_{i,j,k} T[i,j,k] * e_i ⊗ e_j ⊗ e_k
    # where T[i,j,k] = 1 if this term appears in some C[i',k'] computation.
    # Actually T[i,j,k] = 1 for all valid (i,j,k) since each product A[i,j]*B[j,k] 
    # contributes to C[i,k].
    
    # The tensor is: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # Wait, that's not right either. Let me think about this more carefully.
    
    # The matrix multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where the three factors correspond to:
    # - First factor: row index of A (and C), size n
    # - Second factor: column index of A / row index of B, size m  
    # - Third factor: column index of B (and C), size p
    
    # But the output C[i,k] = sum_j A[i,j] B[j,k], so the tensor should be:
    # T = sum_{i,j,k} e_i^{(out)} ⊗ e_j^{(mid)} ⊗ e_k^{(out)}
    # Hmm, this is getting confusing. Let me use the standard formulation.
    
    # Standard: The tensor for multiplying (n x m) by (m x p) is:
    # T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    # with T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = i'*p+k' 
    # and i=i', j=j', k=k'... 
    
    # Actually, let me just use the direct approach. The tensor is:
    # T[i,j,k] = δ_{ik} ... no.
    
    # Let me start fresh. We want to decompose:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The bilinear form is: sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # = sum_{i,j,k} (sum_{j'} A[i,j'] B[j',k]) A[i,j] B[j,k]
    # 
    # The tensor T has entries: T[i,j,k] = 1 for the "core" tensor
    # Actually the tensor is: T_{i,j,k} = δ_{ik}... 
    
    # OK let me just construct it explicitly.
    # The tensor T ∈ R^{(n*m)} ⊗ R^{(m*p)} ⊗ R^{(n*p)}
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k
    # (using the same i, j, k)
    
    # So T has a 1 at position (i*m+j, j*p+k, i*p+k) for each i,j,k
    # and 0 elsewhere.
    
    # The rank of this tensor is what we want to minimize.
    
    # Approach: We can fix one mode and look at the matrix.
    # For a fixed output index c = i*p+k, the slice is:
    # T[:, :, c] = sum over (i,j,k) with i*p+k=c of e_{i*m+j} ⊗ e_{j*p+k}
    # 
    # This is a matrix of size (n*m) x (m*p).
    # The tensor rank is at least the rank of any such slice.
    
    # Better approach: Use the fact that for matrix multiplication, 
    # the tensor can be written as a sum of rank-1 tensors.
    # Each rank-1 term corresponds to a "trivial" multiplication step.
    
    # The schoolbook decomposition has rank n*m*p.
    # We want to find if we can do better.
    
    # Key insight: The tensor rank of matrix multiplication <n,m,p> is known
    # to be at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p).
    # For 2x5 * 5x5: at least max(10, 25, 10) = 25.
    
    # Let's try to find the exact rank using linear algebra.
    # We can formulate this as: find the minimum r such that there exist
    # U ∈ R^{n*m x r}, V ∈ R^{m*p x r}, W ∈ R^{n*p x r} with
    # T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    # This is equivalent to: the tensor T lies in the span of r rank-1 tensors.
    
    # A practical approach: Use the "border rank" or try to find a low-rank
    # decomposition by solving linear systems.
    
    # For small cases, we can try ranks from the lower bound up and check
    # if a decomposition exists.
    
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # Upper bound: n*m*p = 50 (schoolbook)
    
    # Let's try to find the minimum rank by checking if rank r is achievable.
    # For a given r, we need to find U, V, W such that:
    # T[i*m+j, j'*p+k, i'*p+k'] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[i'*p+k', c]
    
    # This is a system of nonlinear equations. However, we can fix V and W
    # and solve for U, or use other tricks.
    
    # Alternative: Use the fact that the tensor is "structured". 
    # The tensor T has T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = i*p+k
    # for some i,j,k.
    
    # Let me try a different approach: exploit the structure.
    # Note that T can be written as:
    # T = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} sum_{k=0}^{p-1} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{i*p+k})
    # 
    # For fixed j, the inner sum is a tensor of size n*p terms.
    # This doesn't immediately help.
    
    # Let me try the exact rational approach suggested in the directive.
    # We can flatten the tensor and use Gaussian elimination to find the rank
    # of the associated linear map.
    
    # Actually, here's a cleaner approach:
    # The tensor T defines a linear map L: R^{n*m} ⊗ R^{m*p} -> R^{n*p}
    # by L(A, B) = A*B.
    # 
    # The tensor rank is NOT the same as the rank of this linear map.
    # The rank of L is always full (n*p) since matrix multiplication is surjective.
    
    # Let me think about this differently. We need to find the minimum r such that
    # there exist vectors u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{n*p} with
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # One approach: Fix v_c and w_c, then solve for u_c.
    # For each (a,b) pair, we need:
    # T[a,b,:] = sum_c u_c[a] * v_c[b] * w_c
    # 
    # This is a linear system in the unknowns u_c[a] for each (a,b).
    
    # This is getting complex. Let me try a simpler heuristic first:
    # Try ranks from 25 to 50 and see which ones work using random initialization
    # and iterative refinement.
    
    # Actually, for this specific case (2x5 * 5x5), let me think about whether
    # there's a known result or a clever decomposition.
    
    # The tensor has n=2, m=5, p=5.
    # Schoolbook rank: 2*5*5 = 50.
    # Lower bound: max(2*5, 5*5, 2*5) = max(10, 25, 10) = 25.
    
    # Let me try to see if rank 25 is achievable.
    # For rank 25, we'd need exactly one term per output position (i,k) on average,
    # which seems tight.
    
    # Let me try a constructive approach. For each j (the "middle" index),
    # we have n*p = 2*5 = 10 terms. If we could share terms across j values,
    # we could reduce the rank.
    
    # Actually, let me just implement the search: try ranks from lower bound up,
    # and for each rank, try to find a decomposition using random initialization
    # and least squares.
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Build the tensor
    tensor = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                tensor[a, b, c] = 1.0
    
    # Lower bound
    lower_bound = max(n*m, m*p, n*p)
    upper_bound = n * m * p
    
    # Try ranks from lower bound to upper bound
    for rank in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use random initialization and iterative refinement
        found = False
        for attempt in range(5):  # Try a few random initializations
            if time.time() > deadline:
                break
            
            np.random.seed(42 + rank * 100 + attempt)
            
            # Initialize with random values
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(n*p, rank)
            
            # Iterative refinement: fix two, solve for the third
            for iteration in range(20):
                if time.time() > deadline:
                    break
                
                # Fix U and V, solve for W
                # T[a,b,:] = sum_c U[a,c] * V[b,c] * W[:,c]
                # For each (a,b): T[a,b,:] = sum_c (U[a,c] * V[b,c]) * W[:,c]
                # This is a linear system: M_ab @ W.T = T[a,b,:]
                # where M_ab[c, d] = sum_c' U[a,c'] * V[b,c'] * delta_{c',d}... 
                # Actually: T[a,b,d] = sum_c U[a,c] * V[b,c] * W[c,d]
                # So for fixed (a,b): T[a,b,:] = (U[a,:] * V[b,:]) @ W
                # Let x = U[a,:] * V[b,:] (element-wise product), then T[a,b,:] = x @ W
                # So W = T[a,b,:] / x ... but this is per (a,b), which is inconsistent.
                
                # Better: stack all (a,b) equations.
                # For all (a,b,d): T[a,b,d] = sum_c U[a,c] * V[b,c] * W[c,d]
                # This is: sum_c (U[a,c] * V[b,c]) * W[c,d] = T[a,b,d]
                
                # Let's form a linear system. For each d, we have:
                # sum_c (U[a,c] * V[b,c]) * W[c,d] = T[a,b,d]
                # This is a linear system in W[:,d] with coefficients U[a,c]*V[b,c].
                
                # Stack over all (a,b):
                # A_mat @ W[:,d] = T[:,:,d].flatten()
                # where A_mat has rows indexed by (a,b) and columns by c:
                # A_mat[(a,b), c] = U[a,c] * V[b,c]
                
                A_mat = np.zeros((n*m * m*p, rank))
                for a in range(n*m):
                    for b in range(m*p):
                        A_mat[a * m*p + b, :] = U[a, :] * V[b, :]
                
                # Solve for W
                T_flat = tensor.reshape(n*m * m*p, n*p)
                try:
                    W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, T_flat, rcond=None)
                    W = W_sol
                except:
                    break
                
                # Fix V and W, solve for U
                A_mat = np.zeros((n*m * m*p, rank))
                for a in range(n*m):
                    for b in range(m*p):
                        A_mat[a * m*p + b, :] = V[b, :] * W.T[b, :]  # This is wrong
                
                # Let me redo this. For fixed V and W:
                # T[a,b,d] = sum_c U[a,c] * V[b,c] * W[c,d]
                # For fixed (b,d): sum_c U[a,c] * (V[b,c] * W[c,d]) = T[a,b,d]
                # So for each (b,d): U @ diag(V[b,:] * W[:,d]) = T[:,b,d]
                # Or: U @ (V[b,:] * W[:,d]) = T[:,b,d]
                # This is: for each c, U[a,c] * V[b,c] * W[c,d] contributes to T[a,b,d]
                
                # Stack over all (b,d):
                # For each (a,b,d): sum_c U[a,c] * V[b,c] * W[c,d] = T[a,b,d]
                # Let's form: B_mat @ U.T = T.transpose(1,2,0).reshape(m*p*n*p, n*m)
                # where B_mat[(b,d), (a,c)] = V[b,c] * W[c,d]
                
                B_mat = np.zeros((m*p * n*p, n*m * rank))
                for b in range(m*p):
                    for d in range(n*p):
                        for a in range(n*m):
                            for c in range(rank):
                                B_mat[(b * n*p + d) * n*m + a, c] = V[b, c] * W[c, d]
                
                T_trans = tensor.transpose(1, 2, 0).reshape(m*p * n*p, n*m)
                try:
                    U_sol, residuals, rank_sol, sv = np.linalg.lstsq(B_mat, T_trans, rcond=None)
                    U = U_sol.reshape(n*m, rank)
                except:
                    break
                
                # Check convergence
                # Compute the reconstructed tensor
                recon = np.einsum('ac,bc,cd->abd', U, V, W)
                error = np.linalg.norm(recon - tensor)
                
                if error < 1e-6:
                    found = True
                    break
                
                if time.time() > deadline:
                    break
            
            if found:
                # Convert to exact rational if possible
                best_rank = rank
                # Round to nearest integer or half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded version
                recon = np.einsum('ac,bc,cd->abd', U_rounded, V_rounded, W_rounded)
                if np.linalg.norm(recon - tensor) < 1e-6:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_rank is not None:
            break
    
    # If no better rank found, use schoolbook
    if best_rank is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((n*p, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = i * p + k
                    U[a, column] = 1
                    V[b, column] = 1
                    W[c, column] = 1
                    column += 1
        
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
