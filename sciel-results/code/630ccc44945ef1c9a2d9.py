import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 tensor decomposition for 3x4x4 matrix multiplication.
    Uses block decomposition into four 3x2x2 multiplications.
    Each 3x2x2 is decomposed into a 1x2x2 (rank 4) and a 2x2x2 (rank 7, Strassen).
    Total rank = 4 * (4 + 7) = 44.
    """
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term: u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1

    # Helper for standard 1x2x2 multiplication
    # A is 1x2, B is 2x2, C is 1x2
    # A[0,0], A[0,1]
    # B[0,0], B[0,1], B[1,0], B[1,1]
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    def add_standard_1x2x2(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i):
        # A row i, cols j, j+1
        # B rows j, j+1, cols k, k+1
        # C row k, col i (Wait, C is 1x2, so row is fixed? No, C is part of the 3x4 result)
        # In the context of 3x2x2 block:
        # A_block is 3x2. We take row i of A_block.
        # B_block is 2x2.
        # C_block is 3x2. We produce row i of C_block.
        
        # A indices: i (global row in block), j, j+1 (global cols in block)
        # B indices: j, j+1 (global rows in block), k, k+1 (global cols in block)
        # C indices: k, k+1 (global cols in block), i (global row in block)
        
        # Term 1: A[i, j] * B[j, k] -> C[k, i]
        add_term(a_base_i + i, a_base_j + j, b_base_j + j, b_base_k + k, c_base_k + k, c_base_i + i)
        # Term 2: A[i, j+1] * B[j+1, k] -> C[k, i]
        add_term(a_base_i + i, a_base_j + j + 1, b_base_j + j + 1, b_base_k + k, c_base_k + k, c_base_i + i)
        # Term 3: A[i, j] * B[j, k+1] -> C[k+1, i]
        add_term(a_base_i + i, a_base_j + j, b_base_j + j, b_base_k + k + 1, c_base_k + k + 1, c_base_i + i)
        # Term 4: A[i, j+1] * B[j+1, k+1] -> C[k+1, i]
        add_term(a_base_i + i, a_base_j + j + 1, b_base_j + j + 1, b_base_k + k + 1, c_base_k + k + 1, c_base_i + i)

    # Helper for Strassen 2x2x2 multiplication
    # A is 2x2, B is 2x2, C is 2x2
    def add_strassen_2x2x2(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i):
        # A rows i, i+1; cols j, j+1
        # B rows j, j+1; cols k, k+1
        # C rows i, i+1; cols k, k+1
        
        # M1 = (A[i,j] + A[i+1,j]) * (B[j,k] + B[j+1,k])
        # We need to combine inputs.
        # Since we are building a linear decomposition, we can't just "add" inputs directly in the tensor factors
        # unless we expand the terms.
        # M1 = A[i,j]B[j,k] + A[i,j]B[j+1,k] + A[i+1,j]B[j,k] + A[i+1,j]B[j+1,k]
        # This would be 4 terms. But Strassen combines them into 1 term in the algorithm.
        # In tensor decomposition, a "term" is a rank-1 tensor $u \otimes v \otimes w$.
        # Strassen's algorithm uses linear combinations of inputs.
        # $M_1 = (A_{11}+A_{21})(B_{11}+B_{21})$
        # This is NOT a rank-1 tensor in the original variables $A_{ij}, B_{jk}$.
        # It is a product of linear forms.
        # To represent this as a sum of rank-1 tensors $A_{ij} B_{jk} C_{ik}$, we must expand it.
        # Wait. The tensor decomposition is over the field.
        # $T = \sum u_r \otimes v_r \otimes w_r$.
        # If we use Strassen, we are effectively using a different basis or a different algorithm.
        # The rank of the tensor is the minimum number of rank-1 tensors.
        # Strassen's algorithm shows that the rank is <= 7.
        # The 7 terms in Strassen's algorithm are:
        # $M_1 = (A_{11}+A_{21})(B_{11}+B_{21})$
        # This term corresponds to a rank-1 tensor in the *extended* space?
        # No. The tensor $T$ is fixed.
        # If rank is 7, there exist 7 rank-1 tensors summing to T.
        # The Strassen algorithm provides such a decomposition.
        # The vectors $u, v, w$ in the decomposition will have entries that are linear combinations of the standard basis vectors.
        
        # So, for M1:
        # $u$ vector has 1s at positions corresponding to $A_{11}$ and $A_{21}$.
        # $v$ vector has 1s at positions corresponding to $B_{11}$ and $B_{21}$.
        # $w$ vector has 1s at positions corresponding to $C_{11}$ and $C_{22}$?
        # Let's check the reconstruction.
        # $C_{11} = M_1 - M_4 + M_7$
        # $C_{12} = M_2 + M_6$
        # $C_{21} = M_3 + M_5$
        # $C_{22} = M_1 - M_5 + M_6$
        
        # So, for each M, we define u, v, w.
        # M1:
        # u: A[i,j]=1, A[i+1,j]=1
        # v: B[j,k]=1, B[j+1,k]=1
        # w: C[i,k]=1, C[i+1,k+1]=1 (from C11 and C22)
        
        add_term_uvw([a_base_i + i, a_base_j], [a_base_i + i + 1, a_base_j], 
                     [b_base_j + j, b_base_k + k], [b_base_j + j + 1, b_base_k + k],
                     [c_base_k + k, c_base_i + i], [c_base_k + k + 1, c_base_i + i + 1])
        
        # M2 = A[i,j] * B[j,k+1]
        # u: A[i,j]=1
        # v: B[j,k+1]=1
        # w: C[i,k+1]=1 (from C12)
        add_term_uvw([a_base_i + i, a_base_j], [], 
                     [b_base_j + j, b_base_k + k + 1], [],
                     [c_base_k + k + 1, c_base_i + i], [])

        # M3 = A[i+1,j+1] * B[j+1,k]
        # u: A[i+1,j+1]=1
        # v: B[j+1,k]=1
        # w: C[i+1,k]=1 (from C21)
        add_term_uvw([a_base_i + i + 1, a_base_j + j + 1], [], 
                     [b_base_j + j + 1, b_base_k + k], [],
                     [c_base_k + k, c_base_i + i + 1], [])

        # M4 = (A[i+1,j] + A[i+1,j+1]) * (B[j,k] + B[j,k+1])
        # u: A[i+1,j]=1, A[i+1,j+1]=1
        # v: B[j,k]=1, B[j,k+1]=1
        # w: C[i,k]=1 (from -M4 in C11)
        add_term_uvw([a_base_i + i + 1, a_base_j], [a_base_i + i + 1, a_base_j + j + 1], 
                     [b_base_j + j, b_base_k + k], [b_base_j + j, b_base_k + k + 1],
                     [c_base_k + k, c_base_i + i], [])
        # Note: M4 contributes -1 to C[i,k]. So we need to subtract it.
        # My add_term_uvw adds +1.
        # I will handle signs by adding multiple terms or adjusting coefficients.
        # Actually, I can just add a term with coefficient -1?
        # The interface requires integer/half-integer entries. -1 is allowed.
        
        # Let's refine add_term_uvw to accept coefficients.
        
        # M5 = A[i,j] * B[j+1,k+1]
        # u: A[i,j]=1
        # v: B[j+1,k+1]=1
        # w: C[i+1,k]=1 (from M5 in C21), C[i+1,k+1]=-1 (from -M5 in C22)
        # So w has 1 at C[i+1,k] and -1 at C[i+1,k+1]
        
        # M6 = (A[i,j] + A[i,j+1]) * (B[j+1,k] + B[j+1,k+1])
        # u: A[i,j]=1, A[i,j+1]=1
        # v: B[j+1,k]=1, B[j+1,k+1]=1
        # w: C[i,k+1]=1 (from M6 in C12), C[i+1,k+1]=1 (from M6 in C22)
        
        # M7 = (A[i,j+1] + A[i+1,j+1]) * B[j,k]
        # u: A[i,j+1]=1, A[i+1,j+1]=1
        # v: B[j,k]=1
        # w: C[i,k]=1 (from M7 in C11)

        pass # Placeholder, I will implement the full logic below

    # Correct implementation of Strassen terms with signs
    def add_strassen_term(u_indices, v_indices, w_indices, coeff=1.0):
        nonlocal col
        for u in u_indices:
            U[u[0], col] += coeff * (1 if u[1] is None else 1) # Simplified: indices are (row, col)
            # Actually, U index is flat.
            # u_indices is list of flat indices.
        # This approach is getting messy. Let's stick to the explicit 44 terms generation.

    # I will generate the 44 terms explicitly using the block structure.
    
    # Block (i_block, k_block) where i_block in {0,1} (A split), k_block in {0,1} (B split)
    # A_split: 0 -> cols 0,1; 1 -> cols 2,3
    # B_split: 0 -> rows 0,1; 1 -> rows 2,3
    
    for i_block in range(2):
        for k_block in range(2):
            # A_block: rows 0..2, cols 2*i_block .. 2*i_block+1
            # B_block: rows 2*k_block .. 2*k_block+1, cols 0..3
            
            # Split B_block cols into 0,1 and 2,3
            for k_sub in range(2):
                # B_sub: rows 2*k_block .. 2*k_block+1, cols 2*k_sub .. 2*k_sub+1
                
                # A_sub is the full A_block (3x2)
                
                # We compute A_block * B_sub (3x2x2)
                
                # Part 1: Row 0 of A_block (1x2) * B_sub (2x2) -> Rank 4
                # A_row0: i=0, j=2*i_block, 2*i_block+1
                # B_sub: j=2*k_block, 2*k_block+1, k=2*k_sub, 2*k_sub+1
                # C: i=0, k=2*k_sub, 2*k_sub+1
                
                # T1
                add_term(0 * m + (2 * i_block), (2 * k_block) * p + (2 * k_sub), (2 * k_sub) * n + 0)
                # T2
                add_term(0 * m + (2 * i_block + 1), (2 * k_block + 1) * p + (2 * k_sub), (2 * k_sub) * n + 0)
                # T3
                add_term(0 * m + (2 * i_block), (2 * k_block) * p + (2 * k_sub + 1), (2 * k_sub + 1) * n + 0)
                # T4
                add_term(0 * m + (2 * i_block + 1), (2 * k_block + 1) * p + (2 * k_sub + 1), (2 * k_sub + 1) * n + 0)
                
                # Part 2: Rows 1,2 of A_block (2x2) * B_sub (2x2) -> Rank 7 (Strassen)
                # A_sub2: i=1,2; j=2*i_block, 2*i_block+1
                # B_sub: j=2*k_block, 2*k_block+1; k=2*k_sub, 2*k_sub+1
                # C_sub2: i=1,2; k=2*k_sub, 2*k_sub+1
                
                # M1: (A[1,j] + A[2,j]) * (B[j,k] + B[j+1,k])
                # u: A[1,j], A[2,j]
                # v: B[j,k], B[j+1,k]
                # w: C[1,k] (+1), C[2,k+1] (+1)
                add_term(1 * m + (2 * i_block), (2 * k_block) * p + (2 * k_sub), (2 * k_sub) * n + 1)
                add_term(1 * m + (2 * i_block), (2 * k_block + 1) * p + (2 * k_sub), (2 * k_sub + 1) * n + 2)
                add_term(2 * m + (2 * i_block), (2 * k_block) * p + (2 * k_sub), (2 * k_sub) * n + 1)
                add_term(2 * m + (2 * i_block), (2 * k_block + 1) * p + (2 * k_sub), (2 * k_sub + 1) * n + 2)
                # This adds 4 terms for M1? No, M1 is 1 term in Strassen, but in tensor rank it expands to 4 rank-1 tensors?
                # No. The rank of 2x2x2 is 7.
                # The 7 terms are the rank-1 tensors.
                # M1 is NOT a rank-1 tensor in the original basis.
                # The decomposition into 7 rank-1 tensors is:
                # 1. u1=(1,0,1,0), v1=(1,0,0,1), w1=(1,0,0,1) -> Corresponds to (A11+A21)(B11+B21)
                # This is a valid rank-1 tensor in the tensor space?
                # Yes, $u \otimes v \otimes w$ is rank 1.
                # So we add ONE term for M1.
                
                # Let's restart the term addition for Strassen.
                # We need to subtract the previous 4 terms added for M1.
                # I will rewrite the loop to be cleaner.

    # Reset and do it properly
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_rank1(u_flat, v_flat, w_flat, coeff=1.0):
        nonlocal col
        U[u_flat, col] += coeff
        V[v_flat, col] += coeff
        W[w_flat, col] += coeff
        col += 1

    for i_block in range(2):
        for k_block in range(2):
            for k_sub in range(2):
                # Standard 1x2x2 (4 terms)
                # A[0, 2*i_block], B[2*k_block, 2*k_sub] -> C[2*k_sub, 0]
                add_rank1(0 * m + (2 * i_block), (2 * k_block) * p + (2 * k_sub), (2 * k_sub) * n + 0)
                # A[0, 2*i_block+1], B[2*k_block+1, 2*k_sub] -> C[2*k_sub, 0]
                add_rank1(0 * m + (2 * i_block + 1), (2 * k_block + 1) * p + (2 * k_sub), (2 * k_sub) * n + 0)
                # A[0, 2*i_block], B[2*k_block, 2*k_sub+1] -> C[2*k_sub+1, 0]
                add_rank1(0 * m + (2 * i_block), (2 * k_block) * p + (2 * k_sub + 1), (2 * k_sub + 1) * n + 0)
                # A[0, 2*i_block+1], B[2*k_block+1, 2*k_sub+1] -> C[2*k_sub+1, 0]
                add_rank1(0 * m + (2 * i_block + 1), (2 * k_block + 1) * p + (2 * k_sub + 1), (2 * k_sub + 1) * n + 0)

                # Strassen 2x2x2 (7 terms)
                # A rows 1,2; B rows 2*k_block, 2*k_block+1; C rows 1,2; C cols 2*k_sub, 2*k_sub+1
                
                # M1: (A[1,j]+A[2,j])(B[j,k]+B[j+1,k])
                # u: A[1,j]=1, A[2,j]=1
                # v: B[j,k]=1, B[j+1,k]=1
                # w: C[1,k]=1, C[2,k+1]=1
                # This is a single rank-1 term in the decomposition?
                # No, the tensor is $T = \sum t_r$.
                # If I define $u = e_{A1j} + e_{A2j}$, $v = e_{Bjk} + e_{Bj+1k}$, $w = e_{C1k} + e_{C2k+1}$,
                # then $u \otimes v \otimes w$ contributes to 4 components of T.
                # This is valid.
                
                u1 = [1 * m + (2 * i_block), 2 * m + (2 * i_block)]
                v1 = [(2 * k_block) * p + (2 * k_sub), (2 * k_block + 1) * p + (2 * k_sub)]
                w1 = [(2 * k_sub) * n + 1, (2 * k_sub + 1) * n + 2]
                # Add term with u having 1s at u1, v having 1s at v1, w having 1s at w1
                for u in u1:
                    for v in v1:
                        for w in w1:
                            # This would be 4 rank 1 tensors.
                            # But Strassen uses 7 terms total for 2x2x2.
                            # The 7 terms are rank-1 tensors in the *extended* space?
                            # No.
                            # The rank of the tensor is 7.
                            # This means there are 7 vectors u, v, w such that sum(u_r v_r w_r) = T.
                            # The vectors u_r, v_r, w_r are sparse.
                            # For M1, u has 2 non-zeros, v has 2, w has 2.
                            # This single term $u_1 \otimes v_1 \otimes w_1$ is one of the 7 terms.
                            
                            pass # I cannot loop like that. I need to add ONE term to the matrices.
                            
                # To add a term with multiple non-zeros in u, v, w:
                # I must add a column to U, V, W.
                # U column has 1s at u1 indices.
                # V column has 1s at v1 indices.
                # W column has 1s at w1 indices.
                
                U[u1[0], col] = 1
                U[u1[1], col] = 1
                V[v1[0], col] = 1
                V[v1[1], col] = 1
                W[w1[0], col] = 1
                W[w1[1], col] = 1
                col += 1
                
                # M2: A[1,j] B[j,k+1]
                U[1 * m + (2 * i_block), col] = 1
                V[(2 * k_block) * p + (2 * k_sub + 1), col] = 1
                W[(2 * k_sub + 1) * n + 1, col] = 1
                col += 1
                
                # M3: A[2,j+1] B[j+1,k]
                U[2 * m + (2 * i_block + 1), col] = 1
                V[(2 * k_block + 1) * p + (2 * k_sub), col] = 1
                W[(2 * k_sub) * n + 2, col] = 1
                col += 1
                
                # M4: (A[2,j]+A[2,j+1])(B[j,k]+B[j,k+1])
                U[2 * m + (2 * i_block), col] = 1
                U[2 * m + (2 * i_block + 1), col] = 1
                V[(2 * k_block) * p + (2 * k_sub), col] = 1
                V[(2 * k_block) * p + (2 * k_sub + 1), col] = 1
                W[(2 * k_sub) * n + 1, col] = -1 # C[1,k] gets -M4
                col += 1
                
                # M5: A[1,j] B[j+1,k+1]
                U[1 * m + (2 * i_block), col] = 1
                V[(2 * k_block + 1) * p + (2 * k_sub + 1), col] = 1
                W[(2 * k_sub) * n + 2, col] = 1 # C[2,k] gets M5
                W[(2 * k_sub + 1) * n + 2, col] = -1 # C[2,k+1] gets -M5
                col += 1
                
                # M6: (A[1,j]+A[1,j+1])(B[j+1,k]+B[j+1,k+1])
                U[1 * m + (2 * i_block), col] = 1
                U[1 * m + (2 * i_block + 1), col] = 1
                V[(2 * k_block + 1) * p + (2 * k_sub), col] = 1
                V[(2 * k_block + 1) * p + (2 * k_sub + 1), col] = 1
                W[(2 * k_sub + 1) * n + 1, col] = 1 # C[1,k+1] gets M6
                W[(2 * k_sub + 1) * n + 2, col] = 1 # C[2,k+1] gets M6
                col += 1
                
                # M7: (A[1,j+1]+A[2,j+1]) B[j,k]
                U[1 * m + (2 * i_block + 1), col] = 1
                U[2 * m + (2 * i_block + 1), col] = 1
                V[(2 * k_block) * p + (2 * k_sub), col] = 1
                W[(2 * k_sub) * n + 1, col] = 1 # C[1,k] gets M7
                col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
