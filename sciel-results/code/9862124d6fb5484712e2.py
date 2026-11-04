"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as iproduct

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Build the tensor T[i,j,k] = delta for M_{2,4,4}
    # T is indexed by (a_idx, b_idx, c_idx) where:
    # a_idx in [0, 8) for A (2x4), b_idx in [0, 16) for B (4x4), c_idx in [0, 8) for C (2x4)
    # T[a,b,c] = 1 if C[c] = A[a] * B[b] in schoolbook, i.e.,
    # c = i*p + j, a = i*m + k, b = k*p + j  =>  T[i*m+k, k*p+j, i*p+j] = 1
    
    # The tensor equation: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # where W uses trace convention: W[k*n+i, r] corresponds to output C[i*n+k]... 
    # Actually let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k*n+i) where i is row of C, k is col of C
    # C is 2x4, so i in [0,2), k in [0,4), W index = k*2+i
    
    # U shape: (n*m, rank) = (8, rank) - A has 8 entries
    # V shape: (m*p, rank) = (16, rank) - B has 16 entries  
    # W shape: (p*n, rank) = (8, rank) - C has 8 entries, indexed as k*n+i
    
    # T[a, b, c] where a = i_A*m + k_A (i_A in [0,n), k_A in [0,m))
    #                   b = k_B*p + j_B (k_B in [0,m), j_B in [0,p))
    #                   c = k_C*n + i_C (i_C in [0,n), k_C in [0,p))
    # T[a,b,c] = 1 iff i_A == i_C and k_A == k_B and j_B == k_C
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i*m + k
                    b = k*p + j
                    c = j*n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor()
    
    # Verify current rank-28 solution
    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        # Compute sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a 3-tensor contraction
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(result, T, atol=1e-9)
    
    def build_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        products = [
            ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
            ([ (2, 1), (3, 1) ], [ (0, 1) ]),
            ([ (0, 1) ], [ (1, 1), (3, -1) ]),
            ([ (3, 1) ], [ (2, 1), (0, -1) ]),
            ([ (0, 1), (1, 1) ], [ (3, 1) ]),
            ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
            ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])
        ]
        recon = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        def add_strassen_block(U_indices, V_indices, W_indices, col_offset):
            for p_idx in range(7):
                x_coeffs, y_coeffs = products[p_idx]
                for x_i, x_c in x_coeffs:
                    U[U_indices[x_i]][col_offset + p_idx] += x_c
                for y_i, y_c in y_coeffs:
                    V[V_indices[y_i]][col_offset + p_idx] += y_c
                for z_i in range(4):
                    coeff = recon[z_i][p_idx]
                    if coeff != 0:
                        W[W_indices[z_i]][col_offset + p_idx] += coeff
        
        U_idx_1 = [0, 1, 4, 5]
        V_idx_1 = [0, 1, 4, 5]
        W_idx_1 = [0, 2, 1, 3]
        add_strassen_block(U_idx_1, V_idx_1, W_idx_1, 0)
        
        U_idx_2 = [2, 3, 6, 7]
        V_idx_2 = [8, 9, 12, 13]
        W_idx_2 = [0, 2, 1, 3]
        add_strassen_block(U_idx_2, V_idx_2, W_idx_2, 7)
        
        U_idx_3 = [0, 1, 4, 5]
        V_idx_3 = [2, 3, 6, 7]
        W_idx_3 = [4, 6, 5, 7]
        add_strassen_block(U_idx_3, V_idx_3, W_idx_3, 14)
        
        U_idx_4 = [2, 3, 6, 7]
        V_idx_4 = [10, 11, 14, 15]
        W_idx_4 = [4, 6, 5, 7]
        add_strassen_block(U_idx_4, V_idx_4, W_idx_4, 21)
        
        return U, V, W
    
    U28, V28, W28 = build_rank28()
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for SVD-based rank reduction
    U_np = np.array(U28, dtype=np.float64)
    V_np = np.array(V28, dtype=np.float64)
    W_np = np.array(W28, dtype=np.float64)
    
    # The decomposition satisfies: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # If we can find a linear transformation that reduces the effective rank...
    # Actually, rank reduction is non-trivial. Let's try a different approach.
    
    # Try rank-24: use a custom algorithm
    # Idea: For C1 = A1*B11 + A2*B21 and C2 = A1*B12 + A2*B22
    # Use 4 Strassen products with shared structure
    
    # Actually, let's try to use the "additive" structure more cleverly.
    # C1 = A1*B11 + A2*B21: This is a 2x2 * 2x2 + 2x2 * 2x2
    # We can view this as (A1, A2) * [[B11, 0], [0, B21]] but that's not a standard matmul.
    
    # Alternative: use rank-7 for each of 4 products but try to merge.
    # The 28 rank-1 terms are: for each of 4 blocks, 7 Strassen terms.
    # Can any two terms from different blocks be merged? Only if their U, V, W vectors are proportional.
    
    # Let's try a numerical rank-reduction approach:
    # Stack the 28 rank-1 tensors and see if any are linearly dependent in the tensor space.
    
    # Better approach: try to find a rank-r decomposition directly via optimization
    # for r = 24, 25, 26, 27
    
    def try_rank_reduction(U0, V0, W0, target_rank, max_time=60):
        """Try to reduce rank from current to target_rank using gradient-free search."""
        U0 = np.array(U0, dtype=np.float64)
        V0 = np.array(V0, dtype=np.float64)
        W0 = np.array(W0, dtype=np.float64)
        r0 = U0.shape[1]
        
        # Start with target_rank columns (keep first target_rank, then optimize)
        # Actually, better: try random subsets or perturbations
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        U = U0[:, :target_rank].copy()
        V = V0[:, :target_rank].copy()
        W = W0[:, :target_rank].copy()
        
        # This won't work directly since dropping columns breaks the identity.
        # Instead, try to find a rank-target_rank decomposition from scratch using
        # the structure.
        
        # For now, just return None if we can't find one
        return None
    
    # Let's try a more systematic approach:
    # The 28 terms come in 4 groups of 7. Within each group, the 7 terms form a
    # Strassen decomposition. Between groups, can we find linear combinations?
    
    # Let's check: are the 28 rank-1 tensors linearly independent?
    # Tensor space has dimension 8*16*8 = 1024. 28 vectors in 1024-dim space are
    # almost certainly independent. So we can't just drop terms.
    
    # However, we might be able to find a DIFFERENT decomposition with lower rank.
    # Let's try a parametric search for rank-24.
    
    # Actually, let me try a known result: for 2xn x nxn, the rank is at most
    # 2 * (2n-1) * something... Let me just try to implement a better algorithm.
    
    # New idea: Use the fact that we're multiplying 2x4 by 4x4.
    # Partition differently: A = [A_top; A_bot] (each 1x4), B = [[B1,B2],[B3,B4]] (each 2x2)
    # C_top = A_top * B  (1x4 * 4x4 = 1x4)
    # C_bot = A_bot * B  (1x4 * 4x4 = 1x4)
    # Each row of C is a dot product of a 1x4 vector with 4x4 matrix = 4 dot products of length 4.
    # That's schoolbook: 8*4 = 32 multiplications. Not helpful.
    
    # Let me try yet another approach: use the rank-28 solution and attempt to
    # find a lower-rank equivalent via algebraic manipulation.
    
    # For now, return the rank-28 solution which we know is correct.
    # But let's add a time check and return early.
    
    if time.time() > deadline - 10:
        return {"U": U28, "V": V28, "W": W28}
    
    # Try to verify and return
    if verify(U28, V28, W28):
        return {"U": U28, "V": V28, "W": W28}
    
    # Subtensor restriction: embed 2x4x4 into 4x4x4, take rank-29 Strassen,
    # restrict to subspace where second row of A is zero.
    # 4x4x4 Strassen: partition into 2x2 blocks, 7 products + 1 extra.
    # 4x4x4 tensor T4[a4,b4,c4] where a4=i4*4+k4, b4=k4*4+j4, c4=j4*4+i4
    # We want T2[a2,b2,c2] where a2=i2*4+k2 (i2 in [0,2)), b2=k2*4+j2, c2=j2*4+i2
    # Embedding: set row index of A in 4x4x4: i4 = i2 (top 2 rows of 4x4 A)
    # The 4x4x4 Strassen has 29 terms. Restrict to i4 < 2 (first 2 rows of A).
    
    # 4x4x4 Strassen-29 decomposition:
    # Partition A4 = [[A11,A12],[A21,A22]], B4 = [[B11,B12],[B21,B22]], C4 = [[C11,C12],[C21,C22]]
    # each 2x2. Standard Strassen gives 7 products for each 2x2 block of C.
    # But the 29-rank decomposition for 4x4x4 uses a more global structure.
    
    # Actually, the known rank-29 for 4x4x4: use 2x2 Strassen on the 2x2 blocks
    # but with some sharing. Let me construct it explicitly.
    
    # For 4x4x4, partition into 4 2x2 blocks for A, B, C.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each C block is sum of two 2x2*2x2 products.
    # Naive: 4 blocks * 2 products * 7 Strassen = 56. Too high.
    
    # Better: view C11 and C21 as (A11, A21) * [[B11,0],[0,B11]]... no.
    # Actually C11 = A11*B11 + A12*B21 and C21 = A21*B11 + A22*B21
    # = [A11 A12; A21 A22] * [[B11, 0],[0, B21]]... no that's not right either.
    
    # Let me just use the standard 4x4x4 Strassen-29:
    # It's based on the fact that 4x4 matmul can be done in 29 multiplications.
    # One construction: use the 2x2 Strassen algorithm recursively but with
    # a clever choice that shares some products.
    
    # Actually, for our purpose, let me try a different approach:
    # Directly search for a rank-24 or rank-25 decomposition using the
    # algebraic structure of the 2x4x4 problem.
    
    # Key insight: C[i,j] = sum_k A[i,k]*B[k,j]
    # For i=0: C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j] + A[0,2]*B[2,j] + A[0,3]*B[3,j]
    # For i=1: C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j] + A[1,2]*B[2,j] + A[1,3]*B[3,j]
    
    # Group by B rows:
    # C[:,j] = B[0,j]*A[:,0] + B[1,j]*A[:,1] + B[2,j]*A[:,2] + B[3,j]*A[:,3]
    # where A[:,k] is a 2-vector (column k of A)
    
    # So C[:,j] = sum_k B[k,j] * A[:,k]
    # This is a sum of 4 outer products (1x2 vectors times 2-vectors)... 
    # Actually B[k,j] is scalar, A[:,k] is 2-vector, so B[k,j]*A[:,k] is 2-vector.
    # C[:,j] is 2-vector = sum of 4 scaled 2-vectors.
    
    # The tensor T[a,b,c]: a = i*4+k (i=row of A, k=col of A = row of B)
    #                   b = k*4+j (k=row of B, j=col of B)
    #                   c = j*4+i (j=col of B, i=row of C)
    # T[a,b,c] = 1 iff the indices are consistent.
    
    # Let me try a direct rank-reduction search using optimization.
    # Start from rank-28 and try to find a rank-24 solution.
    
    # Approach: use the fact that we can write the tensor as a sum of rank-1 terms.
    # Try to find 24 rank-1 terms (u_r, v_r, w_r) such that sum_r u_r ⊗ v_r ⊗ w_r = T.
    
    # This is a non-convex optimization. Let me try a simple approach:
    # Start from the rank-28 solution, randomly drop 4 terms, and try to
    # adjust the remaining 24 to compensate.
    
    # Actually, let me try a cleaner algebraic approach.
    # The 2x4x4 problem has the structure:
    # C = A * B where A is 2x4, B is 4x4.
    
    # Write A = [A1 | A2] where A1, A2 are 2x2.
    # Write B = [[B11 | B12], [B21 | B22]] where each is 2x2.
    # C = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    
    # Let P1 = A1*B11, P2 = A2*B21, P3 = A1*B12, P4 = A2*B22.
    # C_left = P1 + P2, C_right = P3 + P4.
    
    # Each P is a 2x2 product. If we use Strassen for each, that's 4*7=28.
    # But P1 and P3 share A1, and P2 and P4 share A2.
    # P1 and P2 share nothing in B (different B blocks).
    # P1 and P3 share A1 but different B blocks.
    
    # Can we combine P1+P2 into a single lower-rank computation?
    # P1 + P2 = A1*B11 + A2*B21
    # This is not a standard matrix product. It's a "mixed" product.
    
    # However, note that:
    # [A1 A2] * [[B11, 0], [0, B21]] = [A1*B11 + 0, 0 + A2*B21]... no, that's not right.
    # [A1 A2] is 2x4, [[B11,0],[0,B21]] is 4x4.
    # Product is 2x4: [A1*B11 + A2*0, A1*0 + A2*B21] = [A1*B11, A2*B21]
    # That gives us P1 and P2 separately, not their sum.
    
    # Alternative: use the "additive" trick.
    # A1*B11 + A2*B21 = (A1+A2)*B11 + A2*(B21-B11)
    # = (A1+A2)*B11 + A2*B21 - A2*B11
    # = (A1+A2)*B11 + A2*(B21-B11)
    # That's 2 products: (A1+A2)*B11 and A2*(B21-B11).
    # Each is 2x2, so 7+7=14 for the left half. Similarly 14 for right half. Total 28.
    # No improvement.
    
    # Try: A1*B11 + A2*B21 = A1*B21 + A2*B11 + (A1-A2)*(B11-B21)
    # = A1*B21 + A2*B11 + (A1-A2)*(B11-B21)
    # That's 3 products: 3*7=21 for left, 21 for right. Total 42. Worse.
    
    # Let me try yet another grouping:
    # C = A*B. Write B in columns: B = [b0 | b1 | b2 | b3] where bk is 4x1.
    # C[:,j] = A * b_j = sum_k A[:,k] * b_k[j]
    # C[:,j] is 2x1 = sum of 4 outer products of 2x1 and 1x1.
    # Total: 4 columns * 4 terms = 16 rank-1 terms? No, each term is rank 1 in the tensor sense.
    # T[A[:,k], B[k,j], C[:,j]]... 
    
    # Actually the schoolbook is 32 terms (one per (i,k,j) triple).
    # Strassen reduces this to 28 by combining terms within each 2x2 block.
    
    # Let me try to find if rank 24 is achievable by a direct search.
    # Use a simple local search: start from rank-28, perturb, try to drop terms.
    
    # For now, let me just return the rank-28 solution which we know works.
    return {"U": U28, "V": V28, "W": W28}
# EVOLVE-BLOCK-END
