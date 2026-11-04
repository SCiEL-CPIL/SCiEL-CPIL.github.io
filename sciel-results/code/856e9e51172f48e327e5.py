import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # n=4, m=4, p=9
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def reduce_rank(U_cols, V_cols, W_cols, max_time):
        """Try to reduce rank by finding linear dependencies among terms."""
        import numpy as np
        from fractions import Fraction
        
        R = len(U_cols)
        if R <= 1:
            return U_cols, V_cols, W_cols, R
        
        # Build constraint matrix: for each output (i,k), the sum over r of U[r,i]*V[r,k]*W[r,k*n+i] should equal delta
        # Actually, the tensor equation is:
        # T[i,j,k] = sum_r U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i]
        # We need to find if we can set some rank terms to zero while preserving the decomposition.
        
        # Alternative: look for linear dependencies in the set of rank-1 tensors.
        # Each rank term r defines a tensor T_r with entries T_r[i,j,k] = U[r,i*m+j]*V[r,j*p+k]*W[r,k*n+i]
        # If sum_r alpha_r * T_r = 0, we can potentially eliminate one term.
        
        # Build a matrix where each row is a flattened rank-1 tensor
        # Number of entries in tensor: n*m*p = 4*4*9 = 144
        num_tensor_entries = n * m * p
        
        # For efficiency, only consider a subset of tensor entries as constraints
        # Start with all 144 entries
        constraint_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Build matrix M where M[entry, r] = U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        # But this is nonlinear in the coefficients. Instead, we need a different approach.
        
        # Better approach: The decomposition is valid if for all (i,j,k):
        # sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1 if (i,j,k) is a valid product term, else 0
        # Actually for matrix multiplication, T[i,j,k] = 1 if the "product" makes sense, which is always 1 for standard MM.
        # Wait, for standard MM: C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor for MM)
        
        # The decomposition must satisfy:
        # For all (i,k): sum_j sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
        # This is automatically satisfied if the rank-1 tensors sum to the MM tensor.
        
        # To find dependencies: consider the bilinear structure.
        # Fix one factor, say W. Then we need:
        # sum_r W[r, k*n+i] * (U[:, i*m+j] outer V[:, j*p+k]) = delta
        # This is getting complex. Let me try a simpler approach.
        
        # Simple approach: try to find if any rank term can be expressed as a linear combination of others
        # by checking if removing it and adjusting others still gives the same tensor.
        
        # Even simpler: use the fact that if we have R rank-1 tensors summing to T,
        # and there's a linear relation sum_r alpha_r * T_r = 0, then we can potentially
        # reduce the rank by expressing one T_r in terms of the others.
        
        # Build the matrix of rank-1 tensor values (flattened)
        # M[entry_idx, r] = value of rank-r tensor at entry_idx
        M = np.zeros((num_tensor_entries, R), dtype=object)
        
        for entry_idx, (i, j, k) in enumerate(constraint_indices):
            for r in range(R):
                M[entry_idx, r] = U_cols[r][i*m+j] * V_cols[r][j*p+k] * W_cols[r][k*n+i]
        
        # Find null space of M (or its transpose)
        # If M has rank < R, there are linear dependencies.
        # Use SVD to find the null space dimension
        try:
            # Convert to float for SVD
            M_float = np.array([[float(M[i,j]) for j in range(R)] for i in range(num_tensor_entries)])
            U_svd, s_vals, Vt = np.linalg.svd(M_float, full_matrices=False)
            
            # Count singular values above tolerance
            tol = 1e-10
            rank_M = sum(1 for s in s_vals if s > tol)
            
            nullity = R - rank_M
            if nullity > 0:
                # There are linear dependencies. Try to exploit them.
                # The null space vectors give alpha such that sum_r alpha[r]*T_r = 0
                # We can try to eliminate terms by finding a null vector with a large component.
                
                # Get null space vectors
                null_vecs = Vt[rank_M:]
                
                # For each null vector, try to eliminate the term with the largest |alpha|
                best_elimination = None
                best_null_vec = None
                for nv in null_vecs:
                    # Find index with largest absolute value
                    max_idx = np.argmax(np.abs(nv))
                    max_val = abs(nv[max_idx])
                    if max_val > 1e-6:
                        if best_elimination is None or max_val > best_elimination[1]:
                            best_elimination = (max_idx, max_val)
                            best_null_vec = nv
                
                if best_elimination is not None:
                    elim_idx = best_elimination[0]
                    nv = best_null_vec
                    # Scale so that nv[elim_idx] = 1
                    scale = nv[elim_idx]
                    nv_scaled = nv / scale
                    # nv_scaled[elim_idx] = 1
                    # sum_r nv_scaled[r] * T_r = 0
                    # T_elim = -sum_{r != elim} nv_scaled[r] * T_r
                    # We can't directly "remove" T_elim unless we can express it as a sum of rank-1 terms.
                    # This approach doesn't directly reduce rank. Let me try a different method.
                    pass
        except Exception:
            pass
        
        return U_cols, V_cols, W_cols, R
        
    # Strassen 2x2 terms
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u: A indices, v: B indices, w: C indices
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 - m4 + m6
    # c22 = -m1 + m2 + m3 + m6
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Each entry is (u_dict, v_dict, w_dict)
    # u_dict: {index: coeff} for A
    # v_dict: {index: coeff} for B
    # w_dict: {index: coeff} for C
    
    strassen_2x2 = [
        # m1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 49 terms for a 4x4 * 4x4 multiplication.
        A is 4x4, B is 4x4, C is 4x4.
        The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
        """
        # Helper to add one inner Strassen term
        def add_inner(A_blocks, B_blocks, C_blocks):
            # A_blocks: list of (br, bc, coeff) where br, bc are block row/col (0 or 1)
            # This maps to the 2x2 indices inside the block.
            
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Map inner 2x2 indices to global A indices
                # Inner index loc (0..3) -> (li, lj) where li=loc//2, lj=loc%2
                # Global index in A: (A_base_i + br*2 + li, A_base_j + bc*2 + lj)
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                # Map inner 2x2 indices to global B indices
                # Inner index loc -> (lj, lk)
                # Global index in B: (B_base_j + br*2 + lj, B_base_k + bc*2 + lk)
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                # Map inner 2x2 indices to global C indices
                # Inner index loc -> (li, lk)
                # Global index in C: (C_base_i + br*2 + li, C_base_k + bc*2 + lk)
                # W index: k*n + i
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        # Strassen 7 steps for 4x4 * 4x4
        # Blocks:
        # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
        # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
        # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        add_inner(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        
        # M2 = (A21+A22)B11 -> C21, C22
        add_inner(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        )
        
        # M3 = A11(B12-B22) -> C12, C22
        add_inner(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        )
        
        # M4 = A22(B21-B11) -> C11, C21
        add_inner(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        )
        
        # M5 = (A11+A12)B22 -> C11, C12
        add_inner(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        )
        
        # M6 = (A21-A11)(B11+B12) -> C22
        add_inner(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        )
        
        # M7 = (A12-A22)(B21+B22) -> C11
        add_inner(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )

    # Part 1: A * B1 (cols 0-3)
    # A is 4x4, B1 is 4x4 (cols 0-3), C1 is 4x4 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    # A is 4x4, B2 is 4x4 (cols 4-7), C2 is 4x4 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    
    rank = len(U_cols)

    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}


# Placeholder - no reduction attempted due to complexity
