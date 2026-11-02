import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices:
    # A: 2x2 -> flat i*2+j
    # B: 2x2 -> flat j*2+k
    # C: 2x2 -> flat i*2+k (local), mapped to W convention k*n+i globally
    
    # Strassen components:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    # For each column-pair j (0..3) and row-pair s (0..1):
    #   A block: rows 0,1; cols 2s, 2s+1
    #   B block: rows 2s, 2s+1; cols 2j, 2j+1
    #   C block: rows 0,1; cols 2j, 2j+1
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Map local A indices (i_loc*2+j_loc) to global (i_g*4+j_g)
            a_map = [
                0 * 4 + (2 * s + 0),   # i=0, j=2s
                0 * 4 + (2 * s + 1),   # i=0, j=2s+1
                1 * 4 + (2 * s + 0),   # i=1, j=2s
                1 * 4 + (2 * s + 1),   # i=1, j=2s+1
            ]
            
            # Map local B indices (j_loc*2+k_loc) to global (j_g*8+k_g)
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),   # j_g=2s, k_g=2j
                (2 * s + 0) * 8 + (2 * j + 1),   # j_g=2s, k_g=2j+1
                (2 * s + 1) * 8 + (2 * j + 0),   # j_g=2s+1, k_g=2j
                (2 * s + 1) * 8 + (2 * j + 1),   # j_g=2s+1, k_g=2j+1
            ]
            
            # Map local C indices (i_loc*2+k_loc) to W convention (k_g*n + i_g)
            c_map = [
                (2 * j + 0) * 2 + 0,   # i_g=0, k_g=2j
                (2 * j + 1) * 2 + 0,   # i_g=0, k_g=2j+1
                (2 * j + 0) * 2 + 1,   # i_g=1, k_g=2j
                (2 * j + 1) * 2 + 1,   # i_g=1, k_g=2j+1
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor as a matrix: each rank-1 term r contributes
    # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to find vectors c in R^56 such that sum_r c_r * T_r = 0
    # This is equivalent to finding the null space of the "tensor matrix"
    
    # Flatten each rank-1 term into a vector of length n*m*p = 64
    # T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
    # The tensor equation: sum_{i,j,k} A[i,j] B[j,k] C[k,i] = 0 for all A,B
    # is equivalent to: for each (i,j,k), sum_r c_r * U[i,r] * V[j,r] * W[k,r] = 0
    
    # Build matrix M of size (n*m*p) x rank where M[(i,j,k), r] = U[i,r]*V[j,r]*W[k,r]
    # But n*m*p = 64, rank = 56, so we have 64 equations in 56 unknowns
    # The null space of M^T (size 56 x 64) gives dependencies
    
    # Actually, we need: for the decomposition to be exact, we need
    # sum_r U[:,r] V[:,r]^T W[:,r]^T to equal the identity tensor
    # The tensor is T[i,j,k] = delta_{i,k} (for matrix mult)
    # So we need sum_r U[i,r] * V[j,r] * W[k,r] = delta_{i,j} * delta_{j,k} ... 
    
    # Wait, let me reconsider. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the "all ones" in the sense that every
    # triple (i,j,k) contributes to the product)
    # Actually no: the tensor for matrix mult is T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # The rank-1 decomposition says: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # So we need: sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k
    
    # Build the constraint matrix: for each (i,j,k), the equation is
    # sum_r (U[i,r] * V[j,r] * W[k,r]) * c_r = 1  (if we're looking for exact decomp)
    # But we already have an exact decomp, so we want to find c such that
    # sum_r c_r * U[i,r] * V[j,r] * W[k,r] = 0 for all i,j,k
    
    # This is: M @ c = 0 where M has rows indexed by (i,j,k) and columns by r
    # M[i,j,k, r] = U[i,r] * V[j,r] * W[k,r]
    
    # Size: (2*4*8) x 56 = 64 x 56
    # Since 64 > 56, the null space of M (as a 64x56 matrix) is trivial
    # But we want null space of M^T? No...
    
    # Actually, the tensor T is in R^(n x m x p) = R^(2 x 4 x 8)
    # Each rank-1 term is u_r ⊗ v_r ⊗ w_r
    # The span of these 56 rank-1 tensors is a subspace of R^64
    # If they are linearly dependent, there exists c != 0 with sum c_r * T_r = 0
    
    # To check linear independence, build matrix where column r is vec(T_r)
    # vec(T_r) has entries: T_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]
    
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Find null space of M (64 x 56). If rank(M) < 56, there are dependencies.
    # Use SVD to find the null space
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=True)
        # Null space is spanned by columns of Vt_svd.T corresponding to small singular values
        tol = 1e-10
        rank_m = np.sum(s_vals > tol)
        
        if rank_m < total_rank:
            # There are (total_rank - rank_m) linear dependencies
            # Null space vectors are the last (56 - rank_m) rows of Vt_svd
            null_space = Vt_svd[rank_m:, :].T  # shape: 56 x (56 - rank_m)
            null_dim = null_space.shape[1]
            
            # We can use these to reduce the rank
            # New basis: take first rank_m columns as independent, 
            # but we need to be careful about exactness
            
            # Actually, if sum c_r T_r = 0, then we can express one rank-1 term
            # as a linear combination of others, reducing the rank.
            
            # For each null vector c, we can eliminate one term
            # Let's find which terms to eliminate
            
            # Simple approach: for each null vector, find the component with largest
            # absolute value, and express that term in terms of the others
            # Then remove that term from the decomposition
            
            # But this gets complicated. Let's try a different approach:
            # Use the null space to project and get a minimal representation
            
            # Alternative: if the 56 rank-1 tensors are linearly dependent,
            # we can find a basis of rank < 56 that spans the same tensor.
            # The new rank-1 terms would be linear combinations of the old ones,
            # but that doesn't directly give us rank-1 decompositions.
            
            # For exact rank reduction, we need to find a different set of rank-1
            # terms. This is harder.
            
            # Let's try: use the null space to identify redundant terms and
            # attempt to merge them.
            
            # For now, let's just report the rank and see if we can do better
            # with a more sophisticated approach
            pass
    except np.linalg.LinAlgError:
        pass
    
    # The above approach is complex. Let's try a simpler heuristic:
    # Check if there are obvious symmetries or repeated patterns that can be exploited.
    
    # For the 2x4 x 4x8 case with 8 blocks of 2x2 Strassen, the rank is 56.
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # and at most n*m*p = 64 (schoolbook).
    
    # Strassen for 2x2x2 has rank 7, and we have 8 such blocks, giving 56.
    # It's possible that some of these blocks can be shared or combined.
    
    # Let's try to verify the current solution is correct and return it
    # If we find a way to reduce rank, we'll implement it in future iterations.
    
    if time.time() - start_time > deadline:
        pass  # Deterministic and fast; this guard is precautionary
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
