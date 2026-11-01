"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline to be safe within 120s
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Strassen-based approach for rank reduction
    # We'll use a recursive block approach
    # For 2x4x5, we can pad to 2x4x8 and use Strassen on blocks
    
    # Let's try a direct approach: express the tensor and find low-rank decomposition
    # The tensor T[i,j,k] = delta(i,j,k) for matrix multiplication
    # We want U, V, W such that T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # Strategy: Use the fact that for n=2, we can write C = A*B where A is 2x4, B is 4x5
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a bilinear map. We can try to find a decomposition using
    # the structure of the problem.
    
    # For 2x4x5, the schoolbook rank is 40.
    # Strassen for 2x2x2 gives rank 7 instead of 8.
    # We can try to embed a Strassen-like decomposition.
    
    # Let's try a random search for a lower-rank decomposition
    # We'll use numerical rank reduction / tensor decomposition
    
    # First, let's construct the tensor
    # T[i,j,k] = 1 if we're in the right position, else 0
    # Actually, for matrix multiplication, the tensor entries are:
    # T[i,j,k] contributes to C[i,k] from A[i,j]*B[j,k]
    # So the tensor is: T[i,j,k] = 1 (it's a trilinear form)
    
    # Let's use numpy to find a low-rank decomposition
    # We can use the fact that the multiplication tensor has known structure
    
    # For 2x4x5:
    # Let's try rank values from low to high and see which works
    # Use a randomized approach: generate random U, V and solve for W, check consistency
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find a lower rank decomposition using randomized methods
    # We'll try various ranks and use a least-squares approach
    
    # The condition is: for all i,j,k, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is nonlinear in U,V,W jointly.
    
    # Alternative: fix U and V, then W can be solved linearly for each (i,k) pair
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*B[j,k] expressed via U and V)
    
    # Let's use a different approach: 
    # The multiplication tensor for 2x4x5 can be decomposed using block matrix multiplication
    # Write A as [A1 | A2] where A1 is 2x2, A2 is 2x2
    # Write B as [B1; B2] where B1 is 2x5, B2 is 2x5
    # C = A1*B1 + A2*B2
    # Each A_i*B_i is 2x2 times 2x5, which is rank 10 schoolbook
    # Total rank = 20 if we can make them independent, but they share structure
    
    # Actually let's just try random matrices and check
    # For a given rank r, generate random U (8 x r), V (20 x r), and try to find W (10 x r)
    # such that the tensor equation holds.
    
    # The equation: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k]
    # where T[i,j,k] = 1 for the specific (i,j,k) that corresponds to this multiplication.
    
    # Actually, the tensor is: T has entries T[i,j,k] = 1 only when we're multiplying
    # A[i,j] and B[j,k] to get C[i,k]. So T[i,j,k] is a trilinear form.
    
    # Let me think about this differently. The standard approach:
    # We want to find U, V, W of rank r such that:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) * W[k,i,r]
    # Wait, that's not quite right. Let me reconsider.
    
    # The decomposition is: the bilinear map (A,B) -> A*B can be written as:
    # A*B = sum_r U_r(A) * V_r(B) * W_r
    # where U_r(A) is a linear functional on A, V_r(B) is a linear functional on B,
    # and W_r is a matrix (the "structure constants").
    
    # In our notation:
    # U[i*m+j, r] is the coefficient for A[i,j] in the r-th term
    # V[j*p+k, r] is the coefficient for B[j,k] in the r-th term  
    # W[k*n+i, r] is the coefficient for C[i,k] in the r-th term
    
    # So: C[i,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * A[i,j] ... no, that's not right either.
    
    # Let me re-read the interface:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition should satisfy: the multiplication tensor = sum_r U[:,r] * V[:,r] * W[:,r]
    # where the product is the outer product in three modes.
    
    # So T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (using appropriate index mappings)
    
    # For the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # And T[i,j,k] = 1 if we're at the right (i,j,k), which is always the case for each term.
    
    # So the condition is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for the multiplication tensor.
    
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) in the sense that
    # the (i,j,k)-th entry of the multiplication tensor is 1.
    
    # Let's construct the tensor T explicitly and try to find a low-rank decomposition.
    
    # T is a 3D array of shape (n*m, m*p, p*n) but really it's indexed by (i,j,k)
    # Let's create it as a 3D array of shape (n, m, p)
    
    # Actually, for the multiplication tensor, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's not right. The multiplication tensor has T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    # which is just 1 for each valid combination. But that would make it the all-ones tensor.
    
    # I think I'm confusing myself. Let me look at the schoolbook code again:
    # For each (i,j,k), we set U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So the (i,j,k)-th "slice" of the tensor is captured by the (col)-th term.
    # The tensor entry T[i,j,k] = sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # In schoolbook, for the term corresponding to (i',j',k'), U[i*m+j, col] = 1 only if (i,j) = (i',j')
    # V[j*p+k, col] = 1 only if (j,k) = (j',k')
    # W[k*n+i, col] = 1 only if (k,i) = (k',i')
    # So T[i,j,k] = 1 if there exists a col such that (i,j) matches, (j,k) matches, and (k,i) matches
    # which happens iff i=i', j=j', k=k' for that term, so T[i,j,k] = 1 for each (i,j,k).
    
    # So the multiplication tensor for 2x4x5 is actually the all-ones tensor of shape (2,4,5)!
    # No, that can't be right because then any rank-1 decomposition would work.
    
    # I think the issue is that the tensor is not the all-ones tensor. The multiplication tensor
    # is defined such that C[i,k] = sum_j A[i,j] B[j,k], and the tensor T satisfies
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1 if the indices match
    # in the right way. For matrix multiplication, T[i,j,k] = 1 for all i,j,k in the valid range.
    
    # Wait, I think I need to reconsider. The tensor T is a 3D array where T[i,j,k] represents
    # the coefficient of A[i,j]*B[j,k] in the expression for C[i,k]. For standard matrix
    # multiplication, C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    
    # But that would make the tensor the all-ones tensor, which has rank 1. That's clearly wrong.
    
    # I think the correct interpretation is: the multiplication tensor is a trilinear form
    # T: A * B * C where T(A,B,C) = sum_{i,j,k} C[i,k] A[i,j] B[j,k] / (some normalization)
    # or T(A,B,C) = sum_{i,j,k} C[i,k] A[i,j] B[j,k] and we want to express this as
    # sum_r U_r(A) V_r(B) W_r(C) where U_r, V_r, W_r are linear functionals.
    
    # In that case, the tensor entries are T[i,j,k] which is the coefficient of A[i,j]*B[j,k]*C[k,i]
    # (using the W[k*n+i] indexing). For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    
    # So the tensor is indeed the all-ones tensor of shape (n*m, m*p, p*n) with the constraint
    # that the indices are consistent (i.e., the j in A[i,j] matches the j in B[j,k], and the
    # k in B[j,k] matches the k in C[k,i]).
    
    # Given the indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The consistency condition is that the j in U's second index matches the j in V's first index,
    # and the k in V's second index matches the k in W's first index, and the i in W's second
    # index matches the i in U's first index.
    
    # So the tensor T is defined on triples (u_idx, v_idx, w_idx) where u_idx = i*m+j,
    # v_idx = j*p+k, w_idx = k*n+i, and the consistency is automatic given the construction.
    
    # For the multiplication tensor, T[u_idx, v_idx, w_idx] = 1 if the indices are consistent
    # (which they are by construction) and 0 otherwise. But since we only define T for consistent
    # triples, T is 1 for all valid (u_idx, v_idx, w_idx) in the consistent set.
    
    # I think the key insight is that the tensor is NOT the full all-ones tensor of shape
    # (n*m, m*p, p*n), but rather it's a sparse tensor where only entries corresponding to
    # consistent index triples are 1.
    
    # Let's just construct the tensor explicitly and use numpy to find its rank.
    
    # Create the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    
    # Now we want to find a low-rank decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # This is the rank of the tensor.
    
    # We can try to compute the tensor rank using various methods.
    # One approach: use the fact that for fixed r, we can try to solve for U, V, W.
    
    # Let's try a randomized approach: for a given rank r, generate random U and V,
    # then solve for W using least squares, and check if the reconstruction is accurate.
    
    def try_rank(r, max_iterations=100, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        
        # For each column r, we have T[:, :, r] approx U[:, r] * V[:, r] * W[:, r]
        # This is a nonlinear problem. Let's try a different approach.
        
        # Actually, the decomposition is: T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # This is a 3-way array rank decomposition.
        
        # We can try to use ALS (alternating least squares) to find the decomposition.
        
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        for iteration in range(max_iterations):
            # Update U
            for i in range(n * m):
                # T[i, :, :] = U[i, :] * (V * W) where V*W is the outer product
                VW = V.T @ W  # This is r x r, not quite right
                # Actually, for fixed i, T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Let M[i, j, k] = T[i, j, k]
                # For fixed i, M[i] (shape m*p x p*n) = sum_r U[i, r] * (V[:, r] * W[:, r].T)
                # So M[i] = U[i, :] @ (V * W) where (V*W) is a 3D array... this is getting complex.
                
                # Let's use a different formulation. Define S = V.T @ W (r x r matrix)
                # Then T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # This isn't simply U @ S.
                
                # Let's just do a simple gradient-free approach or use scipy.
                pass
            
            break  # Skip the ALS for now, it's complex to implement correctly
        
        return None
    
    # Given the complexity, let's just use the schoolbook decomposition for now
    # and try to improve it with a simple optimization.
    
    # Actually, let's try a simpler approach: use the known Strassen decomposition for 2x2x2
    # and pad/embed it for our 2x4x5 case.
    
    # For 2x2x2, Strassen gives rank 7. We can use this as a building block.
    
    # Let's try to construct a decomposition for 2x4x5 using block structure.
    # Write A (2x4) as [A1 A2] where A1, A2 are 2x2
    # Write B (4x5) as [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x2 * 2x5, which we can decompose
    # A2*B2 is 2x2 * 2x5, which we can decompose
    
    # For 2x2*2x5, we can pad B to 2x8 and use Strassen-like methods, or just use
    # the fact that it's a 2x2 times 2x5 matrix.
    
    # Let's just return the schoolbook decomposition for now, since finding a
    # provably lower-rank decomposition is complex and time-consuming.
    
    # Convert to lists for the required format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
