"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (evaluator kills at 150s)
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    n_m = n * m  # 8
    m_p = m * p  # 28
    p_n = p * n  # 14
    schoolbook_rank = n * m * p  # 56
    
    # Build the schoolbook decomposition
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n_m, rank), dtype=float)
        V = np.zeros((m_p, rank), dtype=float)
        W = np.zeros((p_n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    best_result, best_rank = build_schoolbook()
    
    # Verify schoolbook is valid
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Check: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # And for non-matching triples, should be 0
        # Actually the tensor T_{n,m,p} has T[i,j,k] = 1 if indices match the multiplication pattern
        # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor entries are T[i,j,k] = 1 (for all i,j,k) representing the trilinear form
        # Actually no. The matrix multiplication tensor T_{n,m,p} is defined as:
        # T_{n,m,p}[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
        # and T[i,j,k] = 1 for all valid i,j,k? No...
        # 
        # Let me re-read the task. The tensor decomposition for matrix multiplication:
        # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This is a trilinear form. The tensor T has entries indexed by (i, j, k)
        # where the "free" indices are i (row of C), k (col of C), and j is the summation index.
        # 
        # In the standard formulation, the matrix multiplication tensor M_{n,m,p} is a
        # 3-tensor with M_{n,m,p}[i, j, k] = delta_{i,j,k} in some sense... 
        # 
        # Actually, the standard way: the tensor T_{n,m,p} is defined such that
        # contracting T with vectors u (length n*m), v (length m*p), w (length p*n)
        # gives the bilinear/trilinear form.
        #
        # More precisely: T is a 3-tensor of dimensions (n*m, m*p, p*n).
        # T[a, b, c] where a = i*m + j (A index), b = j*p + k (B index), c = k*n + i (C index)
        # T[a, b, c] = 1 if the indices are consistent (i.e., the j in a matches the j in b,
        # and the k in b matches the k in c, and the i in c matches the i in a), else 0.
        #
        # So T[i*m+j, j*p+k, k*n+i] = 1, and T is 0 elsewhere.
        #
        # The rank of T is the minimum r such that T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c
        # where U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n.
        #
        # So we need: for all (a, b, c), T[a,b,c] = sum_{t=1}^{r} U[a,t] * V[b,t] * W[c,t]
        
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check that non-matching entries are 0
        # Sample some non-matching entries
        for _ in range(200):
            a = np.random.randint(0, n_m)
            b = np.random.randint(0, m_p)
            c = np.random.randint(0, p_n)
            # Check if this is a matching entry
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (j_a == j_b and k_b == k_c and i_c == i_a):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
        return True
    
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        # Should not happen, but just in case
        pass
    
    # Try to find a lower rank decomposition using a simple iterative approach
    # For small reductions (rank 55, 54, ...), we can try perturbation-based search
    
    # Strategy: Start from schoolbook, try to merge two terms or introduce a linear combination
    # that cancels out. This is related to finding a "rank 1 perturbation" that allows
    # reduction.
    
    # A more systematic approach: use the fact that if the tensor has a certain symmetry
    # or structure, we might be able to reduce rank. For 2x4x7, let's think about it as
    # 2x4x4 + 2x4x3.
    
    # Actually, let's try a direct numerical approach: 
    # Initialize with a slightly perturbed schoolbook and try gradient-based optimization
    # on the constraint that the tensor is exactly decomposed.
    
    # Alternative: try specific algebraic constructions.
    
    # For now, let's try a simple search: can we express the tensor with rank 55?
    # This would require finding 55 rank-1 tensors whose sum equals T.
    
    # One approach: take the schoolbook 56 terms and try to "merge" two of them by
    # adding a clever 56th term that allows two original terms to be replaced by one.
    
    # Given the time constraints and the difficulty, let's try a few specific constructions:
    
    # Construction 1: Exploit the structure T_{2,4,7} = T_{2,4,4} ⊕ T_{2,4,3}
    # Rank(T_{2,4,4}) <= 32 (actually known to be 32 for 2x4x4? Let me check)
    # Actually, for 2x2x2, rank is 7. For 2x4x4, it's not simply 4*7=28.
    # The rank of T_{2,4,4} is at most 32 (schoolbook) but could be lower.
    
    # Let's try a different approach: randomized search with local improvement.
    
    def try_random_search(target_rank, max_iter=500):
        """Try to find a decomposition of given rank using random initialization + refinement."""
        if target_rank >= best_rank:
            return None, best_rank
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n_m, target_rank)
            V = np.random.randn(m_p, target_rank)
            W = np.random.randn(p_n, target_rank)
            
            # This is a very difficult non-convex problem. Random initialization + 
            # simple optimization is unlikely to work in the time budget.
            # Let's skip this and try something more structured.
            break
        
        return None, best_rank
    
    # Try structured reductions based on known results
    # For 2x4x7: 
    # - Schoolbook: 56
    # - Any known lower bound? The border rank and rank for small dimensions are studied.
    # - According to some sources, the rank of T_{2,4,4} is 32 (same as schoolbook? or lower?)
    
    # Actually, I recall that for T_{2,2,2}, rank = 7 (Strassen).
    # For T_{2,4,4}, one can use a block decomposition.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into [A1 A2] where A1, A2 are 2x2.
    # Split B into [[B11, B12], [B21, B22]] where each is 2x2.
    # Split C into [C1 C2] where C1, C2 are 2x2.
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # Each 2x2x2 multiplication has rank 7.
    # So we need 4 multiplications of 2x2x2, giving rank 4*7 = 28? 
    # Wait, but the inputs overlap? No, A1, A2, B11, B12, B21, B22 are all independent.
    # So rank(T_{2,4,4}) <= 4 * 7 = 28.
    
    # That's a big reduction from 32! Let me verify this.
    
    # For T_{2,4,4}:
    # A is 2x4: A = [[a11, a12, a13, a14], [a21, a22, a23, a24]]
    # B is 4x4: B = [[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34], [b41, b42, b43, b44]]
    # C is 2x4: C = A*B
    
    # C[0,0] = a11*b11 + a12*b21 + a13*b31 + a14*b41
    # C[0,1] = a11*b12 + a12*b22 + a13*b32 + a14*b42
    # etc.
    
    # If I group: A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C1 = C[:, 0:2], C2 = C[:, 2:4]
    # C1 = A1 * B11 + A2 * B21  (each is 2x2 * 2x2 = 2x2)
    # C2 = A1 * B12 + A2 * B22
    
    # Each 2x2*2x2=2x2 multiplication has rank 7 (Strassen).
    # So C1 requires 2 Strassen multiplications (one for A1*B11, one for A2*B21) = 14 terms
    # C2 requires 2 Strassen multiplications = 14 terms
    # Total: 28 terms.
    
    # So rank(T_{2,4,4}) <= 28.
    
    # Now for T_{2,4,7}:
    # A is 2x4, B is 4x7, C is 2x7.
    # Split B into B_left (4x4) and B_right (4x3).
    # C = A * [B_left, B_right] = [A*B_left, A*B_right]
    # C_left is 2x4, C_right is 2x3.
    # rank(T_{2,4,7}) <= rank(T_{2,4,4}) + rank(T_{2,4,3})
    # rank(T_{2,4,4}) <= 28 (from above)
    # rank(T_{2,4,3}): A is 2x4, B is 4x3, C is 2x3.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x3) and B2 (2x3).
    # C = A1*B1 + A2*B2, where each is 2x2 * 2x3 = 2x3.
    # What's the rank of 2x2 * 2x3 = 2x3?
    # This is T_{2,2,3} (or rather, the tensor for 2x2 times 2x3).
    # Hmm, this is not a standard cube. Let me think differently.
    
    # Actually, 2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] B[j,k] for j=0,1.
    # This has 2*2*3 = 12 "schoolbook" terms.
    # Can we do better? 
    # For each output column k (3 columns), we have C[:,k] = A * B[:,k].
    # A is 2x2, B[:,k] is 2x1, C[:,k] is 2x1.
    # This is just a 2x2 times 2x1 multiplication, which has rank 4 (no Strassen-like reduction for rectangular).
    # So 3 columns * 4 = 12 terms.
    # So rank(T_{2,2,3} type) = 12? Or can we share structure across columns?
    # 
    # Actually, the tensor for A(2x2) * B(2x3) = C(2x3) can be seen as:
    # For each (i, j, k): contribution A[i,j] * B[j,k].
    # The tensor is sum over j of (e_i ⊗ e_j ⊗ e_k) * ... 
    # It's 12 rank-1 terms (one for each i,j,k combination where the indices are "active").
    # I don't think there's an obvious Strassen-like reduction here since the inner dimension is 2
    # and we're computing 2 outputs per column with only 2 products each.
    # 
    # So rank for this part is 12.
    # Total for T_{2,4,3}: 2 * 12 = 24.
    
    # So rank(T_{2,4,7}) <= 28 + 24 = 52.
    
    # That's better than 56! Let me implement this.
    
    # First, let me implement Strassen for 2x2x2.
    def strassen_2x2(A, B):
        """Strassen decomposition for 2x2 matrix multiplication. Returns the 7 products."""
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]
        e, f = B[0, 0], B[0, 1]
        g, h = B[1, 0], B[1, 1]
        # 7 Strassen products
        p1 = (a + d) * (e + h)
        p2 = (c + d) * e
        p3 = a * (f - h)
        p4 = d * (g - e)
        p5 = (a + b) * h
        p6 = (c - a) * (e + f)
        p7 = (b - d) * (g + h)
        C = np.array([
            [p1 + p4 - p5 + p7, p3 + p5],
            [p2 + p4, p1 - p2 + p3 + p6]
        ])
        return p1, p2, p3, p4, p5, p6, p7
    
    # For the tensor decomposition, I need to express each Strassen product as a rank-1 term
    # in the overall 2x4x4 or 2x4x7 decomposition.
    
    # Let me build the decomposition for T_{2,4,4} using 4 Strassen blocks.
    # A is 2x4: indices (i, j) where i in {0,1}, j in {0,1,2,3}
    # B is 4x4: indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: indices (i, k) where i in {0,1}, k in {0,1,2,3}
    
    # Block 1: A1 = A[:, 0:2] (2x2), B11 = B[0:2, 0:2] (2x2), contributes to C[:, 0:2] (2x2)
    # Block 2: A2 = A[:, 2:4] (2x2), B21 = B[2:4, 0:2] (2x2), contributes to C[:, 0:2] (2x2)
    # Block 3: A1 = A[:, 0:2] (2x2), B12 = B[0:2, 2:4] (2x2), contributes to C[:, 2:4] (2x2)
    # Block 4: A2 = A[:, 2:4] (2x2), B22 = B[2:4, 2:4] (2x2), contributes to C[:, 2:4] (2x2)
    
    # For T_{2,4,7}:
    # A is 2x4, B is 4x7, C is 2x7
    # Split B into B[:, 0:4] (4x4) and B[:, 4:7] (4x3)
    # C[:, 0:4] = A * B[:, 0:4] -> use 4 Strassen blocks (rank 28)
    # C[:, 4:7] = A * B[:, 4:7] -> this is 2x4 * 4x3 = 2x3
    
    # For the 2x4 * 4x3 part:
    # Split A into A[:, 0:2] and A[:, 2:4], each 2x2
    # B[:, 4:7] is 4x3, split into B[0:2, 4:7] (2x3) and B[2:4, 4:7] (2x3)
    # C[:, 4:7] = A[:, 0:2] * B[0:2, 4:7] + A[:, 2:4] * B[2:4, 4:7]
    # Each 2x2 * 2x3 = 2x3 multiplication has rank 12 (schoolbook, no known reduction)
    # Total: 2 * 12 = 24
    
    # Grand total: 28 + 24 = 52
    
    # Let me build this decomposition.
    
    def build_block_decomposition():
        """Build a decomposition of T_{2,4,7} with rank 52 using Strassen for 2x2x2 blocks."""
        rank = 52
        U = np.zeros((n_m, rank), dtype=float)  # 8 x 52
        V = np.zeros((m_p, rank), dtype=float)  # 28 x 52
        W = np.zeros((p_n, rank), dtype=float)  # 14 x 52
        
        col = 0
        
        # Helper: add a Strassen 2x2x2 block
        # a_offset: offset in A (2x4) for the 2x2 block (row, col_in_2x4)
        # b_offset: offset in B (4x7) for the 2x2 or 2x3 block
        # c_offset: offset in C (2x7) for the output 2x2 or 2x3 block
        
        def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
            """Add Strassen terms for a 2x2 * 2x2 = 2x2 block.
            a_row_offset: which row block of A (0 or 1, each is 2 rows)
            a_col_offset: which column block of A (0 or 1, each is 2 cols)
            b_row_offset: which row block of B (0 or 1, each is 2 rows)
            b_col_offset: which col block of B (for 2x2: 0 or 1, each 2 cols)
            c_row_offset: which row block of C (0 or 1)
            c_col_offset: which col block of C (for 2x2: 0 or 1, each 2 cols)
            """
            nonlocal col
            # Strassen 7 products for 2x2*2x2=2x2
            # A block is at rows [a_row_offset*2, a_row_offset*2+2), cols [a_col_offset*2, a_col_offset*2+2)
            # B block is at rows [b_row_offset*2, b_row_offset*2+2), cols [b_col_offset*2, b_col_offset*2+2)
            # C block is at rows [c_row_offset*2, c_row_offset*2+2), cols [c_col_offset*2, c_col_offset*2+2)
            
            # Strassen products:
            # p1 = (a+d)*(e+h) where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
            # p2 = (c+d)*e
            # p3 = a*(f-h)
            # p4 = d*(g-e)
            # p5 = (a+b)*h
            # p6 = (c-a)*(e+f)
            # p7 = (b-d)*(g+h)
            
            # For each product, we need to add entries to U, V, W
            # The "A part" is a linear combination of A entries
            # The "B part" is a linear combination of B entries
            # The "C part" distributes to C entries
            
            # Let me define the A-index, B-index, C-index for each Strassen term
            # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in the 2x2 block
            # Global A index: (c_row_offset*2 + a_local_row)*4 + (c_col_offset*2 + a_local_col)
            # Wait, A is 2x4, so global index = i*4 + j where i is row (0,1) and j is col (0,1,2,3)
            # In the block, local row r (0,1) maps to global row c_row_offset*2 + r? No.
            # A is 2x4. The two row blocks are rows 0-1 (only 2 rows total, so it's not split by row).
            # Wait, I think I'm confusing myself. Let me reconsider.
            
            # A is 2x4. There's only 2 rows. The "blocks" are column blocks:
            # A[:, 0:2] and A[:, 2:4]. So a_col_offset is 0 or 1.
            # a_row_offset should always be 0 since there's only one row block.
            
            # Let me redefine:
            # For A (2x4): column block a_col_block (0 or 1), local col lc (0 or 1)
            # Global A index = i*4 + (a_col_block*2 + lc)
            
            # For B (4x7): row block b_row_block (0 or 1), local row lr (0 or 1)
            #              col block b_col_block, local col lc (0 or 1)
            # Global B index = j*7 + k where j is row (0-3), k is col (0-6)
            # j = b_row_block*2 + lr
            # k = b_col_block*2 + lc (for 2x2 blocks)
            
            # For C (2x7): row block c_row_block, local row lr (0 or 1) -- but only 2 rows, so c_row_block=0
            #              col block c_col_block, local col lc (0 or 1)
            # Global C index = k*n + i = k*2 + i where k is col (0-6), i is row (0,1)
            
            # This is getting complicated. Let me just directly assign indices.
            
            # For a 2x2*2x2=2x2 Strassen block:
            # A submatrix: rows [r0, r0+1] of the 2x4 A, cols [c0, c0+1]
            #   where r0 = 0 (since A has only 2 rows), c0 = a_col_block*2
            # B submatrix: rows [b0, b0+1] of the 4x7 B, cols [bc0, bc0+1]
            #   where b0 = b_row_block*2, bc0 = b_col_block*2
            # C submatrix: rows [0, 1] of the 2x7 C, cols [cc0, cc0+1]
            #   where cc0 = c_col_block*2
            
            a_r0 = 0
            a_c0 = a_col_offset * 2
            b_r0 = b_row_offset * 2
            b_c0 = b_col_offset * 2
            c_r0 = 0
            c_c0 = c_col_offset * 2
            
            # A indices in the block:
            # A[0, a_c0+0] -> global A index = 0*4 + (a_c0+0) = a_c0
            # A[0, a_c0+1] -> global A index = 0*4 + (a_c0+1) = a_c0+1
            # A[1, a_c0+0] -> global A index = 1*4 + (a_c0+0) = 4 + a_c0
            # A[1, a_c0+1] -> global A index = 1*4 + (a_c0+1) = 4 + a_c0+1
            
            # B indices:
            # B[b_r0+0, b_c0+0] -> global B index = (b_r0+0)*7 + (b_c0+0)
            # B[b_r0+0, b_c0+1] -> global B index = (b_r0+0)*7 + (b_c0+1)
            # B[b_r0+1, b_c0+0] -> global B index = (b_r0+1)*7 + (b_c0+0)
            # B[b_r0+1, b_c0+1] -> global B index = (b_r0+1)*7 + (b_c0+1)
            
            # C indices:
            # C[0, c_c0+0] -> global C index = (c_c0+0)*2 + 0 = c_c0
            # C[0, c_c0+1] -> global C index = (c_c0+1)*2 + 0 = c_c0+2
            # C[1, c_c0+0] -> global C index = (c_c0+0)*2 + 1 = c_c0+1
            # C[1, c_c0+1] -> global C index = (c_c0+1)*2 + 1 = c_c0+3
            
            # Strassen terms:
            # p1 = (A00+A11)*(B00+B11)
            # p2 = (A10+A11)*B00
            # p3 = A00*(B01-B11)
            # p4 = A11*(B10-B00)
            # p5 = (A00+A01)*B11
            # p6 = (A10-A00)*(B00+B01)
            # p7 = (A01-A11)*(B10+B11)
            
            # C00 = p1 + p4 - p5 + p7
            # C01 = p3 + p5
            # C10 = p2 + p4
            # C11 = p1 - p2 + p3 + p6
            
            # For each p, we add a column to U, V, W
            # U column has non-zero entries at the A indices involved
            # V column has non-zero entries at the B indices involved
            # W column has non-zero entries at the C indices, with coefficients from the C formula
            
            A00 = a_c0
            A01 = a_c0 + 1
            A10 = 4 + a_c0
            A11 = 4 + a_c0 + 1
            
            B00 = b_r0 * 7 + b_c0
            B01 = b_r0 * 7 + b_c0 + 1
            B10 = (b_r0 + 1) * 7 + b_c0
            B11 = (b_r0 + 1) * 7 + b_c0 + 1
            
            C00 = c_c0
            C01 = c_c0 + 2
            C10 = c_c0 + 1
            C11 = c_c0 + 3
            
            # p1
            U[A00, col] = 1; U[A11, col] = 1
            V[B00, col] = 1; V[B11, col] = 1
            W[C00, col] = 1; W[C11, col] = 1
            col += 1
            
            # p2
            U[A10, col] = 1; U[A11, col] = 1
            V[B00, col] = 1
            W[C10, col] = 1
            col += 1
            
            # p3
            U[A00, col] = 1
            V[B01, col] = 1; V[B11, col] = -1
            W[C01, col] = 1
            col += 1
            
            # p4
            U[A11, col] = 1
            V[B10, col] = 1; V[B00, col] = -1
            W[C00, col] = 1; W[C10, col] = 1
            col += 1
            
            # p5
            U[A00, col] = 1; U[A01, col] = 1
            V[B11, col] = 1
            W[C00, col] = -1; W[C01, col] = 1
            col += 1
            
            # p6
            U[A10, col] = 1; U[A00, col] = -1
            V[B00, col] = 1; V[B01, col] = 1
            W[C11, col] = 1
            col += 1
            
            # p7
            U[A01, col] = 1; U[A11, col] = -1
            V[B10, col] = 1; V[B11, col] = 1
            W[C00, col] = 1
            col += 1
            
        # Now for the 2x2 * 2x3 = 2x3 block (no Strassen, just schoolbook)
        def add_rect_block_2x2_2x3(a_col_offset, b_row_offset, b_col_offset, c_col_offset):
            """Add schoolbook terms for a 2x2 * 2x3 = 2x3 block.
            a_col_offset: column block of A (0 or 1)
            b_row_offset: row block of B (0 or 1)
            b_col_offset: column offset in B (4 for the right part)
            c_col_offset: column offset in C (4 for the right part)
            """
            nonlocal col
            a_c0 = a_col_offset * 2
            b_r0 = b_row_offset * 2
            b_c0 = b_col_offset
            c_c0 = c_col_offset
            
            # A is 2x2: A[i, a_c0+j] for i,j in {0,1}
            # B is 2x3: B[b_r0+j, b_c0+k] for j in {0,1}, k in {0,1,2}
            # C is 2x3: C[i, c_c0+k] for i in {0,1}, k in {0,1,2}
            # C[i,k] = sum_j A[i, a_c0+j] * B[b_r0+j, b_c0+k]
            
            A00 = a_c0
            A01 = a_c0 + 1
            A10 = 4 + a_c0
            A11 = 4 + a_c0 + 1
            
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        a_idx = i * 4 + (a_c0 + j)
                        b_idx = (b_r0 + j) * 7 + (b_c0 + k)
                        c_idx = (c_c0 + k) * 2 + i
                        U[a_idx, col] = 1
                        V[b_idx, col] = 1
                        W[c_idx, col] = 1
                        col += 1
        
        # T_{2,4,4} part: 4 Strassen blocks
        # C[:, 0:4] = A[:, 0:2]*B[0:2, 0:2] + A[:, 2:4]*B[2:4, 0:2] + 
        #             A[:, 0:2]*B[0:2, 2:4] + A[:, 2:4]*B[2:4, 2:4]
        # Block (a_col=0, b_row=0, b_col=0, c_col=0): A[:,0:2]*B[0:2,0:2] -> C[:,0:2]
        add_strassen_block(0, 0, 0, 0)  # a_col=0, b_row=0, b_col=0, c_col=0
        # Block (a_col=1, b_row=1, b_col=0, c_col=0): A[:,2:4]*B[2:4,0:2] -> C[:,0:2]
        add_strassen_block(1, 1, 0, 0)
        # Block (a_col=0, b_row=0, b_col=1, c_col=1): A[:,0:2]*B[0:2,2:4] -> C[:,2:4]
        add_strassen_block(0, 0, 1, 1)
        # Block (a_col=1, b_row=1, b_col=1, c_col=1): A[:,2:4]*B[2:4,2:4] -> C[:,2:4]
        add_strassen_block(1, 1, 1, 1)
        
        # T_{2,4,3} part: 2x2*2x3 blocks
        # C[:, 4:7] = A[:, 0:2]*B[0:2, 4:7] + A[:, 2:4]*B[2:4, 4:7]
        # Block (a_col=0, b_row=0): A[:,0:2]*B[0:2,4:7] -> C[:,4:7]
        add_rect_block_2x2_2x3(0, 0, 4, 4)
        # Block (a_col=1, b_row=1): A[:,2:4]*B[2:4,4:7] -> C[:,4:7]
        add_rect_block_2x2_2x3(1, 1, 4, 4)
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    block_result, block_rank = build_block_decomposition()
    
    # Verify the block decomposition
    if verify(block_result["U"], block_result["V"], block_result["W"]):
        best_result = block_result
        best_rank = block_rank
    
    # Check time and return
    if time.time() > deadline - 2:
        pass
    
    return best_result
# EVOLVE-BLOCK-END

