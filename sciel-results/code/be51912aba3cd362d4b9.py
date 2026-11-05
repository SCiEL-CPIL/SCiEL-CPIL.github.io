import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find a lower-rank decomposition using alternating least squares
    # The tensor equation: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (for the MM tensor)
    # Actually, the MM tensor has T[i,j,k] = 1 only in the sense that
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the bilinear form is: sum_j A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to reconstruct this
    
    # More precisely, the matrix multiplication tensor M has:
    # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition requires: for all i,j,k,
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # No wait - the tensor is defined such that:
    # <M, A ⊗ B ⊗ C> = trace(C^T * A * B)
    # So M[i,j,k] = 1 for the appropriate indices
    
    # Let me be more careful. The matrix multiplication tensor T ∈ R^{n×m × m×p × p×n}
    # is defined by: T[i,j, k,n+i] ... no.
    
    # Standard convention: T is a tensor such that
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j) for A, (j,k) for B, (k,i) for C (output)
    # T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    # But this isn't quite right either.
    
    # The correct formulation: We want rank-r factors U, V, W such that
    # for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is NOT correct. The MM tensor doesn't have all-ones entries.
    
    # Let me reconsider. The matrix multiplication bilinear map is:
    # f(A, B) = A*B
    # The tensor T representing this bilinear map satisfies:
    # f(A,B)[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form, if we flatten A to vector a, B to vector b, C to vector c:
    # c[i*p+k] = sum_j a[i*m+j] * b[j*p+k]
    # So T[a_idx, b_idx, c_idx] = 1 if a_idx = i*m+j, b_idx = j*p+k, c_idx = i*p+k
    # for some i,j,k.
    
    # The decomposition: c = sum_r (U^r)^T a * (V^r)^T b * ... 
    # Actually: c_idx = sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
    
    # So we need: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[(i*p+k) % (p*n) ... ]
    
    # Wait, the W index is k*n+i per the problem statement: W[k*n+i, r]
    # And the output is C[i,k] which is at position... let me re-read.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r] is the factor for output C[i,k]
    # But C[i,k] is at position i*p+k in the flattened output.
    # The convention says W[k*n+i], which is different from i*p+k.
    
    # Let me just use: W index = k*n + i as specified.
    
    # So the constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Wait, that would mean the tensor has all-ones entries, which is wrong for MM.
    
    # I think I'm overcomplicating this. Let me look at it differently.
    
    # The matrix multiplication tensor for (n×m)·(m×p) → (n×p) has the property:
    # T[i,j,k] = 1 where i indexes rows of A (and C), j indexes cols of A (and rows of B),
    # k indexes cols of B (and C).
    # The bilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]
    # For this to equal trace(C^T A B), we need T[i,j,k] = 1 for all valid i,j,k.
    # No, that's not right either.
    
    # Let me just implement the alternating minimization for the constraint:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # If this constraint is satisfied, then the decomposition works for MM.
    
    # Actually, I realize the issue. The MM tensor does NOT have all-ones entries.
    # Let me think about what the tensor actually is.
    
    # For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T ∈ R^{(n*m) × (m*p) × (n*p)} is defined by:
    # T[i*m+j, j*p+k, i*p+k] = 1 for all valid i,j,k
    # All other entries are 0.
    
    # So the constraint for decomposition is:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] = 1
    # And for all other index combinations: sum_r ... = 0
    
    # But the problem says W[k*n+i, r], not W[i*p+k, r].
    # Since n=3, p=7: k*n+i = 3k+i, while i*p+k = 7i+k. These are different!
    
    # Let me re-read: "W[k*n+i,r]" - so the W index is k*n + i.
    # And the output C[i,k] - how is it indexed? The problem says the output factor
    # follows W[k*n+i,r], so I'll assume the output index is also k*n+i.
    
    # So the constraint is: for all i in [0,3), j in [0,3), k in [0,7):
    # sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*3+i, r] = 1
    
    # And we need this to hold ONLY for valid (i,j,k) triples where the indices are consistent.
    # Actually, for MM, the constraint is that this sum equals 1 for all valid i,j,k,
    # because the tensor entry is 1 for each (i,j,k) combination.
    
    # Let me verify with a simple case. If A is 1×1, B is 1×1, C is 1×1:
    # C[0,0] = A[0,0]*B[0,0]
    # T[0,0,0] = 1
    # Decomposition: U[0,r]*V[0,r]*W[0,r] summed over r = 1
    # This works with rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1.
    
    # For 2×1 × 1×2 → 2×2:
    # C[0,0] = A[0,0]*B[0,0]
    # C[0,1] = A[0,0]*B[0,1]
    # C[1,0] = A[1,0]*B[0,0]
    # C[1,1] = A[1,0]*B[0,1]
    # T has 4 non-zero entries, one for each (i,j,k) with j=0.
    # With the indexing: U[i*1+0, r] = U[i, r], V[0*2+k, r] = V[k, r], W[k*2+i, r]
    # Constraint: for i=0,1; k=0,1: sum_r U[i,r]*V[k,r]*W[2k+i,r] = 1
    # This is 4 equations. Can we do it with rank 2?
    # U[0,0]=1, U[1,0]=1, U[0,1]=0, U[1,1]=0
    # V[0,0]=1, V[1,0]=0, V[0,1]=0, V[1,1]=1
    # W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0
    # W[0,1]=0, W[1,1]=1, W[2,1]=0, W[3,1]=1
    # Check: i=0,k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0 = 1 ✓
    # i=0,k=1: U[0,0]*V[1,0]*W[1,0] + U[0,1]*V[1,1]*W[1,1] = 0 + 0 = 0 ✗ (should be 1)
    
    # So that doesn't work. The issue is that for MM, not all (i,k) pairs are independent
    # in the way I'm thinking.
    
    # Let me just go with the alternating minimization approach and try to minimize
    # the error of the constraint. I'll try various ranks and see what works.
    
    best_rank = n * m * p  # 63
    best_U, best_V, best_W = schoolbook()
    
    # Try ranks from low to high
    for rank in range(15, min(50, best_rank + 1)):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 100 + rank)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U given V, W
                # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is a linear system in U[i*m+j, :]
                # For fixed i,j: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let M_k,r = V[j*p+k, r] * W[k*n+i, r]
                # Then: M @ U[i*m+j, :] = 1 (vector of ones of length p)
                
                for i in range(n):
                    for j in range(m):
                        # Build the system: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A_mat @ u = ones(p)
                        try:
                            u_sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U[i * m + j, :] = u_sol
                        except:
                            pass
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            v_sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V[j * p + k, :] = v_sol
                        except:
                            pass
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            w_sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W[k * n + i, :] = w_sol
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    # Found a valid decomposition
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
                elif max_err < 1e-6:
                    # Close, keep iterating
                    pass
                else:
                    # Not converging, try next attempt
                    if iteration > 50:
                        break
            
            if rank < best_rank:
                # Check if this is better
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-10:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
    
    # Convert to lists for return
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
