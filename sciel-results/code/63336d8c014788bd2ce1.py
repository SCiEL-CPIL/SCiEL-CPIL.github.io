import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Uses block-wise partitioning of the middle dimension m=5 into 2, 2, 1.
    Each block is further partitioned in the p dimension (7) into 2, 2, 2, 1.
    
    For 2x2x2 sub-blocks, we use the Strassen decomposition (rank 7) for the 2x2 part
    of A and a schoolbook decomposition (rank 4) for the 1x2 part of A (total rank 11).
    For other sub-blocks, we use schoolbook.
    
    Total rank calculation:
    Block 1 (m=2): 3 * (3x2x2 rank 11) + 1 * (3x2x1 rank 6) = 33 + 6 = 39
    Block 2 (m=2): 39
    Block 3 (m=1): 3 * (3x1x2 rank 6) + 1 * (3x1x1 rank 3) = 18 + 3 = 21
    Total: 39 + 39 + 21 = 99
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    
    # Define splits
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    # Calculate total rank
    total_rank = 0
    block_ranks = []
    for m_split in m_splits:
        block_rank = 0
        for p_split in p_splits:
            if m_split == 2 and p_split == 2:
                block_rank += 11 # Strassen(7) + Linear(4)
            elif m_split == 2 and p_split == 1:
                block_rank += 6 # Schoolbook 3x2x1
            elif m_split == 1 and p_split == 2:
                block_rank += 6 # Schoolbook 3x1x2
            elif m_split == 1 and p_split == 1:
                block_rank += 3 # Schoolbook 3x1x1
        block_ranks.append(block_rank)
        total_rank += block_rank
        
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    current_rank = 0
    
    # Helper to add terms
    def add_term(i, j, k, coeff):
        nonlocal current_rank
        U[i * m + j, current_rank] = coeff
        V[j * p + k, current_rank] = coeff
        W[k * n + i, current_rank] = coeff
        current_rank += 1

    # Helper for Strassen 2x2x2 (rows 0,1 of A, cols 0,1 of B)
    # i_offset: row offset in A
    # j_offset: col offset in A (and row offset in B)
    # k_offset: col offset in B
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # M1
        add_term(i_off + 0, j_off + 0, k_off + 0, 1) # A00 B00
        add_term(i_off + 1, j_off + 1, k_off + 0, 1) # A11 B00
        add_term(i_off + 0, j_off + 0, k_off + 1, 1) # A00 B11
        add_term(i_off + 1, j_off + 1, k_off + 1, 1) # A11 B11
        # This is not quite right. Strassen terms are products of sums.
        # The tensor decomposition requires linear terms in U, V, W.
        # U, V, W contain the coefficients of the factors.
        # For Strassen:
        # Term 1: (A00+A11) * (B00+B11) * C
        # U row for A00+A11: i0, j0 (coeff 1), i1, j1 (coeff 1)
        # V row for B00+B11: j0, k0 (coeff 1), j1, k1 (coeff 1)
        # W row for C: k0, i0 (coeff 1), k0, i1 (coeff 1), k1, i0 (coeff 1), k1, i1 (coeff 1)
        # This creates a rank 1 tensor for the product? No.
        # The decomposition is a sum of rank 1 tensors.
        # Each term in Strassen is a product of linear forms.
        # So each Strassen term corresponds to 1 rank-1 component in the tensor decomposition?
        # Yes, if we view the multiplication as $C = \sum T_r (U_r \cdot A) (V_r \cdot B)$.
        # The Strassen algorithm expresses $C$ as a linear combination of 7 products.
        # So we add 7 terms to the decomposition.
        
        # Let's implement the 7 Strassen terms properly.
        # The indices for U, V, W must correspond to the specific linear combinations.
        # However, our U, V, W matrices have rows indexed by single (i,j) or (j,k) or (k,i).
        # We cannot put a "sum" in a single row of U.
        # We must expand the linear combinations into multiple rank-1 terms?
        # No, that would increase the rank.
        # The standard tensor decomposition uses $U, V, W$ where each column is a vector.
        # If $U_r$ is a vector with non-zero entries at multiple positions, that's fine.
        # So for Term 1: $L_A = A_{00} + A_{11}$.
        # $U_r$ has 1 at (0,0) and 1 at (1,1).
        # $V_r$ has 1 at (0,0) and 1 at (1,1).
        # $W_r$ has 1 at (0,0), (0,1), (1,0), (1,1).
        # This is a valid rank-1 tensor component.
        
        # Let's define the 7 terms for 2x2x2.
        # i, j, k are local indices 0,1.
        
        # T1: (A00+A11)(B00+B11)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (0,1):1, (1,0):1, (1,1):1
        # Wait, W is $C_{ik}$.
        # $C = T1 * (B00+B11) * (A00+A11)^T$? No.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # The term contributes to all $C_{ik}$.
        # The W vector for this term should have 1s where the product contributes.
        # The product $(A_{00}+A_{11})(B_{00}+B_{11})$ contributes to $C_{00}, C_{01}, C_{10}, C_{11}$.
        # So W has 1 at (0,0), (1,0), (0,1), (1,1) in (k,i) ordering?
        # W row index is $k*n + i$.
        # So W[0*n+0]=1, W[0*n+1]=1, W[1*n+0]=1, W[1*n+1]=1.
        
        # Let's code a helper that adds a "block" of rank-1 terms defined by sparse vectors.
        
        pass # Replaced by direct implementation below

    # We will implement the terms directly in the loop for clarity and correctness.
    
    for m_idx, m_split in enumerate(m_splits):
        j_start = sum(m_splits[:m_idx])
        for p_idx, p_split in enumerate(p_splits):
            k_start = sum(p_splits[:p_idx])
            
            if m_split == 2 and p_split == 2:
                # 2x2x2 Strassen (Rank 7) for rows 0,1 of local A
                # Local i: 0,1. Local j: 0,1. Local k: 0,1.
                # Global i: i_start + i
                # Global j: j_start + j
                # Global k: k_start + k
                
                # T1: (A00+A11)(B00+B11)
                # U: i0,j0:1; i1,j1:1
                # V: j0,k0:1; j1,k1:1
                # W: k0,i0:1; k0,i1:1; k1,i0:1; k1,i1:1
                U[i_start* m + j_start, current_rank] = 1
                U[(i_start+1)* m + (j_start+1), current_rank] = 1
                V[j_start* p + k_start, current_rank] = 1
                V[(j_start+1)* p + (k_start+1), current_rank] = 1
                W[k_start* n + i_start, current_rank] = 1
                W[k_start* n + i_start+1, current_rank] = 1
                W[(k_start+1)* n + i_start, current_rank] = 1
                W[(k_start+1)* n + i_start+1, current_rank] = 1
                current_rank += 1
                
                # T2: (A10+A11)B00
                # U: i1,j0:1; i1,j1:1
                # V: j0,k0:1
                # W: k0,i1:1
                U[(i_start+1)* m + j_start, current_rank] = 1
                U[(i_start+1)* m + (j_start+1), current_rank] = 1
                V[j_start* p + k_start, current_rank] = 1
                W[k_start* n + i_start+1, current_rank] = 1
                current_rank += 1
                
                # T3: A00(B01-B11)
                # U: i0,j0:1
                # V: j0,k1:1; j1,k1:-1
                # W: k1,i0:1
                U[i_start* m + j_start, current_rank] = 1
                V[j_start* p + (k_start+1), current_rank] = 1
                V[(j_start+1)* p + (k_start+1), current_rank] = -1
                W[(k_start+1)* n + i_start, current_rank] = 1
                current_rank += 1
                
                # T4: A11(B10-B00)
                # U: i1,j1:1
                # V: j1,k0:1; j0,k0:-1
                # W: k0,i1:1
                U[(i_start+1)* m + (j_start+1), current_rank] = 1
                V[(j_start+1)* p + k_start, current_rank] = 1
                V[j_start* p + k_start, current_rank] = -1
                W[k_start* n + i_start+1, current_rank] = 1
                current_rank += 1
                
                # T5: (A00+A10)B11
                # U: i0,j0:1; i1,j0:1
                # V: j1,k1:1
                # W: k1,i0:1
                U[i_start* m + j_start, current_rank] = 1
                U[(i_start+1)* m + j_start, current_rank] = 1
                V[(j_start+1)* p + (k_start+1), current_rank] = 1
                W[(k_start+1)* n + i_start, current_rank] = 1
                current_rank += 1
                
                # T6: (A10-A00)(B00+B01)
                # U: i1,j0:1; i0,j0:-1
                # V: j0,k0:1; j0,k1:1
                # W: k0,i1:1
                U[(i_start+1)* m + j_start, current_rank] = 1
                U[i_start* m + j_start, current_rank] = -1
                V[j_start* p + k_start, current_rank] = 1
                V[j_start* p + (k_start+1), current_rank] = 1
                W[k_start* n + i_start+1, current_rank] = 1
                current_rank += 1
                
                # T7: (A01-A11)(B10+B11)
                # U: i0,j1:1; i1,j1:-1
                # V: j1,k0:1; j1,k1:1
                # W: k1,i0:1
                U[i_start* m + (j_start+1), current_rank] = 1
                U[(i_start+1)* m + (j_start+1), current_rank] = -1
                V[(j_start+1)* p + k_start, current_rank] = 1
                V[(j_start+1)* p + (k_start+1), current_rank] = 1
                W[(k_start+1)* n + i_start, current_rank] = 1
                current_rank += 1
                
                # Now add the 1x2x2 part for row 2 of A (Local i=2)
                # Schoolbook for 1x2x2 (Rank 4)
                # C2k = A20 B0k + A21 B1k
                
                # T8: A20 B00
                U[(i_start+2)* m + j_start, current_rank] = 1
                V[j_start* p + k_start, current_rank] = 1
                W[k_start* n + i_start+2, current_rank] = 1
                current_rank += 1
                
                # T9: A20 B01
                U[(i_start+2)* m + j_start, current_rank] = 1
                V[j_start* p + (k_start+1), current_rank] = 1
                W[(k_start+1)* n + i_start+2, current_rank] = 1
                current_rank += 1
                
                # T10: A21 B10
                U[(i_start+2)* m + (j_start+1), current_rank] = 1
                V[(j_start+1)* p + k_start, current_rank] = 1
                W[k_start* n + i_start+2, current_rank] = 1
                current_rank += 1
                
                # T11: A21 B11
                U[(i_start+2)* m + (j_start+1), current_rank] = 1
                V[(j_start+1)* p + (k_start+1), current_rank] = 1
                W[(k_start+1)* n + i_start+2, current_rank] = 1
                current_rank += 1
                
            elif m_split == 2 and p_split == 1:
                # 2x1x1 Schoolbook (Rank 6)
                # C_i0 = A_i0 B_00 + A_i1 B_10
                for i in range(3):
                    # T: A_i0 B_00
                    U[(i_start+i)* m + j_start, current_rank] = 1
                    V[j_start* p + k_start, current_rank] = 1
                    W[k_start* n + i_start+i, current_rank] = 1
                    current_rank += 1
                    
                    # T: A_i1 B_10
                    U[(i_start+i)* m + (j_start+1), current_rank] = 1
                    V[(j_start+1)* p + k_start, current_rank] = 1
                    W[k_start* n + i_start+i, current_rank] = 1
                    current_rank += 1

            elif m_split == 1 and p_split == 2:
                # 1x2x2 Schoolbook (Rank 6)
                # C_ik = A_0j B_jk
                for k in range(2):
                    for j in range(2):
                        for i in range(3):
                            U[(i_start+i)* m + j_start, current_rank] = 1
                            V[j_start* p + k_start+k, current_rank] = 1
                            W[(k_start+k)* n + i_start+i, current_rank] = 1
                            current_rank += 1

            elif m_split == 1 and p_split == 1:
                # 1x1x1 Schoolbook (Rank 3)
                for i in range(3):
                    U[(i_start+i)* m + j_start, current_rank] = 1
                    V[j_start* p + k_start, current_rank] = 1
                    W[k_start* n + i_start+i, current_rank] = 1
                    current_rank += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
