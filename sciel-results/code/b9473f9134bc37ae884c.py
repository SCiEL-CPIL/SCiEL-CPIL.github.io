"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 3, 4, 5
    # Target rank 58
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 component
    # u_idx, v_idx, w_idx are the indices in U, V, W
    # coeff_u, coeff_v, coeff_w are the coefficients
    def add_component(u_idx, v_idx, w_idx, cu, cv, cw):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        U[u_idx][col] += cu
        V[v_idx][col] += cv
        W[w_idx][col] += cw
        col += 1

    # Helper to add Strassen decomposition for a 2x2x2 block
    # i_off, j_off, k_off are the offsets for A (i,j), B (j,k), C (k,i)
    # A is n x m, B is m x p, C is p x n
    # Local A is 2x2 at (i_off, j_off)
    # Local B is 2x2 at (j_off, k_off)
    # Local C is 2x2 at (k_off, i_off)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A22 + A11)(B11 + B22)
        # A indices: (i_off+1, j_off+1) and (i_off, j_off)
        # B indices: (j_off, k_off) and (j_off+1, k_off+1)
        # C indices: (k_off, i_off)
        
        # We need to be careful with the Strassen formulas.
        # Standard Strassen for C = AB:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We add 7 rank-1 terms.
        
        # 1. M1
        # A part: A11 (1) + A22 (1)
        # B part: B11 (1) + B22 (1)
        # C part: C11 (1)
        # This is not a single rank-1 term. It's a sum of 4 rank-1 terms?
        # No, Strassen reduces the number of *multiplications*.
        # In tensor decomposition, each multiplication $M_k$ corresponds to a rank-1 component?
        # No. $M_k = (\sum u_i A_i) (\sum v_j B_j)$.
        # This is a product of two linear combinations.
        # In the tensor decomposition $T = \sum r_k U_k \otimes V_k \otimes W_k$,
        # each $r_k$ is a scalar multiplication.
        # The Strassen algorithm computes 7 scalar products $M_1 \dots M_7$.
        # Each $M_k$ is a dot product of two vectors (linear combos of A and B entries).
        # So each $M_k$ corresponds to ONE rank-1 component in the tensor decomposition?
        # Let's check.
        # $M_1 = (A_{11} + A_{22})(B_{11} + B_{22})$.
        # This is not a rank-1 term in the tensor $T_{ijk} = \delta_{jk}$.
        # The tensor is $\sum_{j} A_{ij} B_{jk}$.
        # Strassen replaces $\sum_{j} A_{ij} B_{jk}$ with linear combos of $M_k$.
        # Each $M_k$ is a product of a linear combo of A's and a linear combo of B's.
        # So yes, each $M_k$ is a rank-1 component in the *extended* space?
        # No, the tensor decomposition is for the multiplication map.
        # The rank of the tensor is the number of scalar multiplications.
        # So each $M_k$ contributes 1 to the rank.
        # And the coefficients in U, V, W are the coefficients of the linear combos.
        
        # So for M1:
        # U coeffs: A11: 1, A22: 1
        # V coeffs: B11: 1, B22: 1
        # W coeffs: C11: 1
        
        # We add this as a single column in U, V, W.
        
        def add_rank1(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, coeff in zip(u_indices, u_coeffs):
                if coeff != 0:
                    U[idx][col] += coeff
            for idx, coeff in zip(v_indices, v_coeffs):
                if coeff != 0:
                    V[idx][col] += coeff
            for idx, coeff in zip(w_indices, w_coeffs):
                if coeff != 0:
                    W[idx][col] += coeff
            col += 1
            
        # Indices
        # A11: (i_off, j_off)
        # A12: (i_off, j_off+1)
        # A21: (i_off+1, j_off)
        # A22: (i_off+1, j_off+1)
        
        # B11: (j_off, k_off)
        # B12: (j_off, k_off+1)
        # B21: (j_off+1, k_off)
        # B22: (j_off+1, k_off+1)
        
        # C11: (k_off, i_off)
        # C12: (k_off, i_off+1)
        # C21: (k_off+1, i_off)
        # C22: (k_off+1, i_off+1)
        
        # M1
        add_rank1(
            [i_off*m + j_off, (i_off+1)*m + (j_off+1)],
            [j_off*p + k_off, (j_off+1)*p + (k_off+1)],
            [k_off*n + i_off],
            [1, 1], [1, 1], [1]
        )
        # M2
        add_rank1(
            [(i_off+1)*m + j_off, (i_off+1)*m + (j_off+1)],
            [j_off*p + k_off],
            [k_off+1*n + i_off],
            [1, 1], [1], [1]
        )
        # M3
        add_rank1(
            [i_off*m + j_off],
            [(j_off+1)*p + k_off, (j_off+1)*p + (k_off+1)],
            [k_off*n + (i_off+1)],
            [1], [1, -1], [1]
        )
        # M4
        add_rank1(
            [(i_off+1)*m + (j_off+1)],
            [j_off*p + (k_off+1), j_off*p + k_off],
            [k_off*n + i_off],
            [1], [1, -1], [1]
        )
        # M5
        add_rank1(
            [i_off*m + j_off, i_off*m + (j_off+1)],
            [(j_off+1)*p + (k_off+1)],
            [k_off*n + i_off],
            [1, 1], [1], [-1]
        )
        # M6
        add_rank1(
            [(i_off+1)*m + j_off, i_off*m + j_off],
            [j_off*p + k_off, j_off*p + (k_off+1)],
            [k_off+1*n + (i_off+1)],
            [1, -1], [1, 1], [1]
        )
        # M7
        add_rank1(
            [i_off*m + (j_off+1), (i_off+1)*m + (j_off+1)],
            [(j_off+1)*p + k_off, (j_off+1)*p + (k_off+1)],
            [k_off*n + i_off],
            [1, -1], [1, 1], [1]
        )

    # Helper to add schoolbook for small blocks
    # ni x nj x nk
    def add_schoolbook(i_off, j_off, k_off, ni, nj, nk):
        nonlocal col
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    add_component(u_idx, v_idx, w_idx, 1, 1, 1)

    # Apply Strassen to the 2x2x2 blocks
    # A is 3x4. Split rows 2+1, cols 2+2.
    # A11: (0,0) to (1,1) -> 2x2
    # A12: (0,2) to (1,3) -> 2x2
    # A21: (2,0) to (2,1) -> 1x2
    # A22: (2,2) to (2,3) -> 1x2
    
    # B is 4x5. Split rows 2+2, cols 2+3.
    # B11: (0,0) to (1,1) -> 2x2
    # B12: (0,2) to (1,4) -> 2x3
    # B21: (2,0) to (3,1) -> 2x2
    # B22: (2,2) to (3,4) -> 2x3
    
    # C is 5x3.
    
    # 1. A11 B11 (2x2x2) -> Strassen
    add_strassen(0, 0, 0)
    
    # 2. A12 B21 (2x2x2) -> Strassen
    # A12 is at (0, 2). B21 is at (2, 0).
    # Result C is at (0, 0) (cols 0-1 of C, rows 0-1 of A? No)
    # C index: (k, i).
    # B21 k is 0-1. A12 i is 0-1.
    # So C is at (0, 0) in the k-i grid?
    # k_off = 0, i_off = 0.
    add_strassen(0, 2, 0)
    
    # 3. A11 B12 (2x2x3) -> Schoolbook
    # A11 (0,0) 2x2. B12 (0,2) 2x3.
    # k_off = 2, i_off = 0.
    add_schoolbook(0, 0, 2, 2, 2, 3)
    
    # 4. A12 B22 (2x2x3) -> Schoolbook
    # A12 (0,2) 2x2. B22 (2,2) 2x3.
    # k_off = 2, i_off = 0.
    add_schoolbook(0, 2, 2, 2, 2, 3)
    
    # 5. A21 B11 (1x2x2) -> Schoolbook
    # A21 (2,0) 1x2. B11 (0,0) 2x2.
    # k_off = 0, i_off = 2.
    add_schoolbook(2, 0, 0, 1, 2, 2)
    
    # 6. A22 B21 (1x2x2) -> Schoolbook
    # A22 (2,2) 1x2. B21 (2,0) 2x2.
    # k_off = 0, i_off = 2.
    add_schoolbook(2, 2, 0, 1, 2, 2)
    
    # 7. A21 B12 (1x2x3) -> Schoolbook
    # A21 (2,0) 1x2. B12 (0,2) 2x3.
    # k_off = 2, i_off = 2.
    add_schoolbook(2, 0, 2, 1, 2, 3)
    
    # 8. A22 B22 (1x2x3) -> Schoolbook
    # A22 (2,2) 1x2. B22 (2,2) 2x3.
    # k_off = 2, i_off = 2.
    add_schoolbook(2, 2, 2, 1, 2, 3)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
