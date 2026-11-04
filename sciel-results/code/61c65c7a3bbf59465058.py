import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    deadline = start_time + 110  # Safety margin well before 120s limit
    
    # We will try to construct a rank-161 decomposition (3x3 rank 23 * 2x2 rank 7).
    # If verification fails, we fall back to rank-189 (3x3 rank 27 * 2x2 rank 7).
    
    block_size = 2
    num_blocks = 3
    
    # Strassen coefficients for 2x2 multiplication
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Rank-23 decomposition for 3x3 multiplication
    # This is the Coppersmith decomposition.
    # I will define the 23 terms.
    # Format: (U_list, V_list, W_list)
    # U_list: [(row, col, coeff), ...]
    
    # If I don't have the exact 23 terms, I will use schoolbook (27 terms) as fallback.
    # For now, I will assume I can retrieve them or use a placeholder that triggers fallback.
    # Actually, I will hardcode a known valid set if possible, or just use schoolbook for now to ensure stability?
    # No, the goal is to improve.
    
    # Let's try to use the rank-23 terms.
    # I will use a hardcoded list of 23 terms.
    # If the list is empty or None, use schoolbook.
    
    m3_terms = get_m3_rank23_terms()
    
    if not m3_terms:
        # Fallback to schoolbook 3x3 (27 terms)
        m3_terms = get_m3_schoolbook_terms()
        total_rank = 27 * 7
    else:
        total_rank = 23 * 7
        
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    col = 0
    bs = block_size
    dim = n  # 6
    
    for ib in range(num_blocks):
        for jb in range(num_blocks):
            for kb in range(num_blocks):
                # Outer 3x3 term index
                # We need to map the outer term to the inner Strassen terms.
                # The outer term provides coefficients for A[ib,jb], B[jb,kb], C[ib,kb].
                # Let outer_term be (u_outer, v_outer, w_outer).
                # u_outer is a vector of 9 coeffs for the 3x3 block A[ib,jb].
                # But Strassen operates on 2x2 blocks.
                # Wait, the block structure is:
                # A is 6x6. A[ib, jb] is a 2x2 block.
                # The 3x3 multiplication is on the *block* level.
                # So the "A" in the 3x3 multiplication is a 3x3 matrix of 2x2 blocks.
                # The rank-23 decomposition applies to the 3x3 matrix of blocks.
                # Each "entry" of the 3x3 block matrix is a 2x2 matrix.
                # So for each of the 23 outer terms, we have a linear combination of the 2x2 blocks.
                # Let the outer term be defined by matrices U3 (3x3), V3 (3x3), W3 (3x3).
                # The term is trace(U3 * A_blocks * V3 * B_blocks)? No.
                # The term is:
                # M_k = (sum_{i,j} u_{ij,k} A_{ij}) (sum_{i,j} v_{ij,k} B_{ij})
                # Here A_{ij} is a 2x2 matrix.
                # So M_k is a 2x2 matrix product?
                # No, the product of two 2x2 matrices is a 2x2 matrix.
                # So M_k is a 2x2 matrix.
                # Then we add W3_{ik} * M_k to C_{ik} (which is 2x2).
                
                # So for each outer term k (0..22):
                # We have u_coeffs (3x3 matrix of coeffs), v_coeffs (3x3 matrix of coeffs), w_coeffs (3x3 matrix of coeffs).
                # The term contributes to the global tensor.
                # U part: For each (i,j) in 3x3, u_{ij,k} * A_block[i,j].
                # V part: For each (j,k) in 3x3, v_{jk,k} * B_block[j,k].
                # W part: For each (i,k) in 3x3, w_{ik,k} * C_block[i,k].
                
                # But wait, the "product" in the rank decomposition is a scalar multiplication of the linear forms.
                # Here the linear forms are matrices (2x2).
                # The product of two 2x2 matrices is a 2x2 matrix.
                # So the "multiplication" in the tensor decomposition is the matrix multiplication of the 2x2 blocks.
                # This means we can't just use scalar Strassen.
                # We need to decompose the 2x2 matrix multiplication *inside* each term?
                
                # Let's re-evaluate the block decomposition.
                # Rank(M_{mn}) <= Rank(M_m) * Rank(M_n).
                # This is achieved by:
                # Rank(M_6) <= Rank(M_3) * Rank(M_2) = 23 * 7 = 161.
                # The construction is:
                # Take the rank-23 decomposition of M_3.
                # For each of the 23 terms, we have linear forms L_A (3x3 matrix of scalars? No, 3x3 matrix of 2x2 blocks? No.)
                
                # Let's clarify the tensor product structure.
                # M_6 = M_3 \otimes M_2.
                # The basis of M_3 is 3x3 matrices.
                # The basis of M_2 is 2x2 matrices.
                # The tensor product basis is 6x6 matrices.
                # A rank-1 term in M_3 is u v^T (outer product of vectors).
                # A rank-1 term in M_2 is a b^T.
                # A rank-1 term in M_6 is (u v^T) \otimes (a b^T) = (u \otimes a) (v \otimes b)^T.
                # This corresponds to a specific 6x6 matrix multiplication.
                
                # So, if we have a rank-23 decomposition of M_3:
                # C_3 = sum_{k=1}^{23} (U_k \circ A_3) (V_k \circ B_3) W_k ?
                # No, standard form: C_{ij} = sum_k u_{ik} v_{kj} m_k.
                # Here m_k = (sum a_{pq} u_{pq,k}) (sum b_{qr} v_{qr,k}).
                # For M_6, we replace scalars with 2x2 blocks.
                # m_k becomes a 2x2 matrix multiplication?
                # No. The "scalar" multiplication in the tensor decomposition is the field multiplication.
                # If we replace scalars with 2x2 matrices, the "multiplication" becomes matrix multiplication.
                # So m_k = (sum A_{pq} u_{pq,k}) (sum B_{qr} v_{qr,k}) is a 2x2 matrix product.
                # Then C_{ij} (2x2 block) = sum_k w_{ij,k} m_k.
                # This is NOT a valid tensor decomposition of M_6 using scalar multiplication.
                # Because the "product" m_k is a matrix product, not a scalar product.
                # The tensor decomposition requires the factors to be linear forms that are multiplied by scalar multiplication.
                
                # Therefore, the simple product Rank(M_3) * Rank(M_2) does NOT directly give a valid tensor decomposition for M_6 in the standard sense (where the intermediate step is scalar multiplication).
                # Wait. The rank of a tensor is the minimum number of rank-1 tensors.
                # A rank-1 tensor in M_6 is of the form U \otimes V \otimes W where U, V, W are vectors in the dual spaces.
                # This corresponds to $C_{ij} = \sum_k U_{ik,k} V_{jk,k} W_{ik,k}$? No.
                # $C_{ij} = \sum_k (U_k \cdot A) (V_k \cdot B) W_{ij,k}$.
                # Here $U_k \cdot A$ is a scalar.
                # So we must decompose the 2x2 multiplication into scalar multiplications.
                # That's what Strassen does.
                
                # So, for each of the 23 outer terms, we have a 2x2 matrix product $M_k$.
                # We must expand $M_k$ using Strassen (7 terms).
                # So for each outer term, we generate 7 inner terms.
                # Total rank = 23 * 7 = 161.
                
                # So the logic holds.
                # For each outer term k:
                # We have linear forms on the 3x3 block matrix A_blocks.
                # Let $A_{block} = [A_{ij}]$ where $A_{ij}$ are 2x2.
                # The outer term defines $X_k = \sum_{i,j} u_{ij,k} A_{ij}$ (a 2x2 matrix).
                # And $Y_k = \sum_{j,k} v_{jk,k} B_{jk}$ (a 2x2 matrix).
                # Then $M_k = X_k Y_k$ (2x2 matrix product).
                # We expand $M_k$ using Strassen:
                # $M_k = \sum_{s=1}^7 S_{ks}$.
                # Each $S_{ks}$ is a scalar product of linear forms in the entries of $X_k$ and $Y_k$.
                # But $X_k$ and $Y_k$ are themselves linear forms in the entries of $A$ and $B$.
                # So we can substitute.
                # The final result is a sum of 161 scalar products.
                
                # Implementation:
                # 1. Get 23 outer terms (u_3x3, v_3x3, w_3x3).
                # 2. For each outer term:
                #    a. Construct the 2x2 matrices X and Y symbolically? No, just coefficients.
                #    b. X is a 2x2 matrix where each entry is a linear combination of A's entries.
                #    c. Apply Strassen to the 2x2 multiplication X*Y.
                #    d. Strassen gives 7 terms. Each term is a linear form in X's entries and a linear form in Y's entries.
                #    e. Substitute the linear forms of X and Y with their definitions in terms of A and B.
                #    f. This gives 7 global terms.
                #    g. Add to U, V, W.
                
                pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
