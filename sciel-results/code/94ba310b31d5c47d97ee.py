"""Program for matrix multiplication 2x3 x 3x6 tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We'll use a block-based approach.
    # For each column block of B (there are 3 blocks of width 2),
    # we decompose the multiplication A[:, j] * B[j, block] into rank-11 terms.
    # A[:, j] is 2x1, B[j, block] is 1x2.
    # So we need 2x1 * 1x2 = 2x2, which has rank 4 (but we use 7 via Strassen-like)
    # Wait, 2x1 * 1x2 is just outer product of a 2-vector and a 2-vector.
    # That's rank 1 in tensor sense? No, the multiplication tensor for 2x1 * 1x2 is
    # sum_{i,j} e_{i}^{A} e_{j}^{B} e_{i}^{C} e_{j}^{C} ... 
    # Actually, let me think about this differently.
    #
    # The full multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed j, the contribution to C is A[:,j] * B[j,:] which is a 2x6 matrix
    # of rank 1 (outer product). But we're computing all j simultaneously.
    #
    # Let me use a different strategy: directly enumerate the rank-1 terms.
    # The naive decomposition has 2*3*6 = 36 terms: for each (i,j,k), term = A[i,j]*B[j,k] contributes to C[i,k].
    # We can try to reduce this.
    #
    # Strategy: Group by j. For each j in {0,1,2}, we have A[:,j] (2-vector) and B[j,:] (6-vector).
    # The product A[:,j] * B[j,:] gives a 2x6 matrix. We need to represent this as a sum of rank-1 terms
    # in the tensor sense, but since we're summing over j, we can combine.
    #
    # Actually, the key insight: for each j, the sub-tensor for A[:,j] * B[j,:] -> C[:, :] (2x6)
    # can be decomposed. The tensor is sum_{i,k} e_i^A e_k^B e_{i,k}^C for fixed j... 
    # No, let me be more careful.
    #
    # Let me just use a direct search approach with numpy.
    
    # Build the multiplication tensor T[i,j,k; r] where r indexes C
    # T has shape (n*m, m*p, p*n) = (6, 18, 12)
    # T[i,j,k, r] = 1 if r corresponds to C[(k//p)*n + (i%m)] ... wait, let me be careful with indexing.
    #
    # A is flat: index a = i*m + j, where i in [0,n), j in [0,m)
    # B is flat: index b = j*p + k, where j in [0,m), k in [0,p)
    # C is flat: index c = k*n + i, where k in [0,p), i in [0,n)
    #
    # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j,k), and c corresponds to (k,i)
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now we want to find a low-rank decomposition.
    # Let's use randomized SVD or a greedy approach to find rank-1 approximations.
    
    # First, let's check the naive rank:
    # The tensor T can be written as sum of rank-1 tensors.
    # Each "element" T[a,b,c] corresponds to a triple (a,b,c) where the indices are linked.
    
    # Let's try a greedy rank-1 decomposition approach.
    # We'll iteratively find the best rank-1 approximation to the residual.
    
    def tensor_to_matrix(T3d, mode):
        """Reshape 3D tensor to 2D matrix by flattening two modes."""
        if mode == 0:
            return T3d.reshape(T3d.shape[0], T3d.shape[1] * T3d.shape[2])
        elif mode == 1:
            return T3d.transpose(1, 0, 2).reshape(T3d.shape[1], T3d.shape[0] * T3d.shape[2])
        else:
            return T3d.transpose(2, 0, 1).reshape(T3d.shape[2], T3d.shape[0] * T3d.shape[1])
    
    residual = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    max_rank = 50
    tol = 1e-10
    
    for rank_iter in range(max_rank):
        if np.max(np.abs(residual)) < tol:
            break
        
        # Try to find a good rank-1 approximation.
        # Use the fact that for matrix multiplication tensors, the optimal rank-1 terms
        # often have sparse structure. Let's try a heuristic.
        
        # For each mode, compute the matrix unfolding and find the dominant direction.
        M0 = tensor_to_matrix(residual, 0)
        M1 = tensor_to_matrix(residual, 1)
        M2 = tensor_to_matrix(residual, 2)
        
        # Use power iteration on each mode to find approximate directions.
        # But this may not give an exact rank-1 fit.
        
        # Alternative: since this is a structured tensor, let's try specific patterns.
        # For matrix multiplication, rank-1 terms often have the form:
        # u supported on A indices for some (i,j), v on B indices for (j,k), w on C for (k,i)
        # or combinations.
        
        # Let's try a more direct approach: enumerate possible "patterns" of support.
        # This is complex, so let's use a numerical optimization approach.
        
        # Find u, v, w that maximize <residual, u ⊗ v ⊗ w>
        # Use alternating least squares or a simple grid search over sparse patterns.
        
        # For now, let's use a randomized approach with structured sparse vectors.
        best_u, best_v, best_w = None, None, None
        best_score = -np.inf
        
        # Try combinations of small support sets
        # This is expensive, so let's limit the search.
        
        # Actually, let's just use the known structure. For 2x3x6, let's try to find
        # a decomposition using the block structure.
        
        # For each j, the sub-tensor involves A[:,j], B[j,:], and all of C.
        # Let me just use a fixed construction based on Strassen-like ideas.
        
        break  # Fallback to manual construction
    
    # Since the automatic approach is complex, let me construct a known valid decomposition.
    # I'll use a rank-30 construction based on block decomposition.
    
    # For each column pair (2b, 2b+1) of B, we have a 2x3x2 sub-problem.
    # 2x3x2 matrix multiplication: A (2x3) * B (3x2) = C (2x2)
    # This can be done in rank 11 (naive is 12, so we save 1).
    # Actually, let me just use rank 12 per block for simplicity, giving 36 total,
    # but that's worse than 33.
    
    # Let me use the same approach as the original but verify it works.
    # The original uses rank 33 with 11 terms per block.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block b in {0,1,2}, corresponding to columns 2b, 2b+1 of B
    # Local A: 2x3, indices 0-5
    # Local B: 3x2, indices 0-5
    # Local C: 2x2, indices 0-3
    
    # Strassen-like 2x2x2 terms for the first two columns/rows
    block_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00)
        ({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
        # Scalar terms for third row/col
        ({2: 1}, {4: 1}, {0: 1}),
        ({2: 1}, {5: 1}, {1: 1}),
        ({5: 1}, {4: 1}, {2: 1}),
        ({5: 1}, {5: 1}, {3: 1}),
    ]
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(block_terms):
            col = block * 11 + t_idx
            
            for idx, val in u_dict.items():
                U[idx][col] = val
            
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_rec = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    if not np.allclose(T_rec, T, atol=1e-10):
        # Fallback: use naive rank-36 decomposition
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][r] = 1
                    V[b][r] = 1
                    W[c][r] = 1
    
    return {"U": U, "V": V, "W": W}
