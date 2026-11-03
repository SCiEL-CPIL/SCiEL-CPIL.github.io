"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using a 4+1 block structure.
    # Total rank expected: 110.
    
    # Helper to add a rank-1 component to the lists
    # U is n*m x rank, V is m*p x rank, W is p*n x rank
    # Indices:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    U_components = []
    V_components = []
    W_components = []
    
    def add_component(u_row, u_val, v_row, v_val, w_row, w_val):
        # We store tuples (row, val) for each matrix
        # But since each component is rank 1, we just add to the current column
        pass

    # It is easier to build the matrices directly or as lists of columns.
    # Let's use lists of lists for U, V, W.
    # U: 25 x R
    # V: 25 x R
    # W: 25 x R
    
    # We don't know R yet, so we'll accumulate columns.
    cols = []
    
    def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        cols.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))

    # 1. 4x4x4 Block (Top-Left)
    # Indices i in 0..3, j in 0..3, k in 0..3
    # This is M_444. Rank 49.
    # Implemented as 7 Strassen blocks of 2x2x2.
    
    # Strassen 2x2x2 decomposition
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    def add_strassen_2x2(i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets for the 2x2 block
        # A is 2x2, B is 2x2, C is 2x2
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11 (with coefficients)
        
        # We need to represent the sum of products.
        # Each "multiplication" M_x is a rank 1 tensor? No.
        # M1 = (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is 4 rank-1 tensors.
        # But Strassen reduces the number of multiplications to 7.
        # The "multiplications" are scalar multiplications of linear combinations.
        # In the tensor decomposition, each scalar multiplication corresponds to one rank-1 term.
        # So M1 contributes 1 rank-1 term to the decomposition?
        # No, the term is $(A00 + A11) \otimes (B00 + B11) \otimes W_{coeff}$.
        # This is a rank-1 tensor with coefficients 1 for A00, 1 for A11, etc.
        # So yes, each of the 7 Strassen steps is ONE rank-1 tensor in the decomposition.
        
        # M1
        u_idx_1 = [i_off*m + j_off, (i_off+1)*m + (j_off+1)]
        v_idx_1 = [j_off*p + k_off, (j_off+1)*p + (k_off+1)]
        # Coefficients for U: 1, 1
        # Coefficients for V: 1, 1
        # W coefficients for C:
        # C00: +1, C01: 0, C10: 0, C11: -1 (Wait, C11 = M1 - M2 + M3 + M6)
        # Let's trace the W coefficients for each M.
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1
        # C11 -= M1
        w_idx_1 = [k_off*n + i_off, (k_off+1)*n + (i_off+1)]
        w_coeff_1 = [1, -1]
        
        # We need to add this as a single rank-1 component?
        # No, a rank-1 component has ONE non-zero entry in U, ONE in V, ONE in W.
        # Strassen's M1 involves multiple entries in U and V.
        # So M1 is NOT a rank-1 tensor in the standard basis.
        # It is a rank-1 tensor in the *transformed* basis?
        # No, the decomposition requires U, V, W to be matrices.
        # A term $u \otimes v \otimes w$ means $U_{u, c}=1, V_{v, c}=1, W_{w, c}=1$.
        # Strassen's method uses linear combinations.
        # This means we cannot just add 7 terms.
        # We must expand the linear combinations into basis vectors.
        
        # For M1 = (A00 + A11)(B00 + B11):
        # This is equivalent to 4 rank-1 tensors:
        # A00B00, A00B11, A11B00, A11B11.
        # So Strassen's "7 multiplications" actually correspond to 7 *sets* of rank-1 tensors?
        # No.
        # The rank of the tensor is the minimum number of rank-1 tensors.
        # Strassen's algorithm has 7 multiplications.
        # Each multiplication is of the form $L_A \cdot L_B$.
        # $L_A$ is a linear combination of A entries.
        # $L_B$ is a linear combination of B entries.
        # The product $L_A L_B$ is distributed to C entries.
        # This structure is exactly a rank-1 tensor decomposition if we view $L_A$ and $L_B$ as vectors.
        # So, for M1, we have:
        # $u = e_{A00} + e_{A11}$
        # $v = e_{B00} + e_{B11}$
        # $w = e_{C00} - e_{C11}$
        # This is ONE rank-1 tensor: $u \otimes v \otimes w$.
        # Yes!
        # So each Strassen step is ONE rank-1 tensor.
        # The entries of U, V, W will be 1, -1, etc.
        
        # So for M1:
        # U has 1 at A00, 1 at A11.
        # V has 1 at B00, 1 at B11.
        # W has 1 at C00, -1 at C11.
        
        # We need to add this to our columns.
        # Since our columns are sparse, we can represent each column as a list of (index, value) for U, V, W.
        
        def add_strassen_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            cols.append((u_indices, u_vals, v_indices, v_vals, w_indices, w_vals))
        
        # M1
        add_strassen_term(
            [i_off*m + j_off, (i_off+1)*m + (j_off+1)], [1, 1],
            [j_off*p + k_off, (j_off+1)*p + (k_off+1)], [1, 1],
            [k_off*n + i_off, (k_off+1)*n + (i_off+1)], [1, -1]
        )
        
        # M2 = (A10 + A11)B00
        add_strassen_term(
            [(i_off+1)*m + j_off, (i_off+1)*m + (j_off+1)], [1, 1],
            [j_off*p + k_off], [1],
            [(k_off+1)*n + i_off], [1] # C10 += M2
        )
        
        # M3 = A00(B01 - B11)
        add_strassen_term(
            [i_off*m + j_off], [1],
            [j_off*p + (k_off+1), (j_off+1)*p + (k_off+1)], [1, -1],
            [(k_off+1)*n + i_off], [1] # C01 += M3
        )
        
        # M4 = A11(B10 - B00)
        add_strassen_term(
            [(i_off+1)*m + (j_off+1)], [1],
            [(j_off+1)*p + k_off, j_off*p + k_off], [1, -1],
            [k_off*n + i_off, (k_off+1)*n + i_off], [1, 1] # C00 += M4, C10 += M4
        )
        
        # M5 = (A00 + A01)B11
        add_strassen_term(
            [i_off*m + j_off, i_off*m + (j_off+1)], [1, 1],
            [(j_off+1)*p + (k_off+1)], [1],
            [k_off*n + i_off, (k_off+1)*n + (i_off+1)], [-1, 1] # C00 -= M5, C11 += M5
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_strassen_term(
            [(i_off+1)*m + j_off, i_off*m + j_off], [1, -1],
            [j_off*p + k_off, j_off*p + (k_off+1)], [1, 1],
            [(k_off+1)*n + (i_off+1)], [1] # C11 += M6
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_strassen_term(
            [i_off*m + (j_off+1), (i_off+1)*m + (j_off+1)], [1, -1],
            [(j_off+1)*p + k_off, (j_off+1)*p + (k_off+1)], [1, 1],
            [k_off*n + i_off, (k_off+1)*n + (i_off+1)], [1, 1] # C00 += M7, C11 += M7
        )

    # Apply Strassen to the 4x4 block (2x2 blocks of 2x2)
    # Block (0,0): i=0,1; j=0,1; k=0,1
    add_strassen_2x2(0, 0, 0)
    # Block (0,1): i=0,1; j=2,3; k=0,1
    add_strassen_2x2(0, 2, 0)
    # Block (1,0): i=2,3; j=0,1; k=0,1
    add_strassen_2x2(2, 0, 0)
    # Block (1,1): i=2,3; j=2,3; k=0,1
    add_strassen_2x2(2, 2, 0)
    
    # Wait, Strassen for 4x4 requires 7 calls to 2x2 multiplication.
    # The 7 calls are not just on the diagonal blocks.
    # They are on specific combinations.
    # My previous logic of just applying to diagonal blocks is WRONG.
    # Strassen for 4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Here A11 is the top-left 2x2 block of A.
    # So I need to define the blocks first.
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    # And similarly for B and C.
    
    # I will rewrite the 4x4x4 decomposition correctly.
    
    cols = [] # Reset
    
    def add_4x4_strassen():
        # M1 = (A11 + A22)(B11 + B22)
        # A11+A22 involves 4 entries in A.
        # B11+B22 involves 4 entries in B.
        # C11 += M1, C22 -= M1
        # This is 1 rank-1 tensor?
        # U has 4 non-zeros, V has 4 non-zeros, W has 2 non-zeros.
        # Yes.
        
        # Indices for A11: (0,0), (0,1), (1,0), (1,1)
        # Indices for A22: (2,0), (2,1), (3,0), (3,1)
        # etc.
        
        # This is getting complex to hardcode.
        # Given the time constraint, I will stick to the 125 rank if I cannot easily implement 110.
        # But 110 is a significant improvement.
        
        # Let's try a simpler improvement: 3+2 split?
        # Rank of 3x3x3 is 23.
        # Rank of 2x2x2 is 7.
        
        # C33 = A33B33 + A32B23
        # R(M333) = 23.
        # R(M323) = ? (3x2x3). Rank 14?
        # C32 = A33B32 + A32B22
        # C23 = A23B33 + A22B23
        # C22 = A23B32 + A22B22
        
        # If R(M323) is 14, then:
        # C33: 23 + 14 = 37
        # C32: 14 + 7 = 21
        # C23: 14 + 7 = 21
        # C22: 14 + 7 = 21
        # Total: 37 + 21 + 21 + 21 = 100.
        
        # Is R(M323) = 14?
        # M323 is 3x2x3.
        # It is isomorphic to M322?
        # Rank of M322 is 14?
        # M222 is 7.
        # M322: 3x2x2.
        # Rank is 14?
        
        # If I can implement 3x3x3 (Rank 23) and the trivial blocks, I can get 100.
        # But I don't have the 3x3x3 decomposition hardcoded.
        
        # Given the complexity, I will return the 125 rank solution to ensure correctness and avoid timeout/bugs.
        # The prompt says "1.0 equals the seed program".
        # If I can't easily find a better one, I should keep the seed.
        
        pass

    # Fallback to standard 125
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
