import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Total rank estimate:
    # Part 1 (j=0): 6 terms
    # Part 2 (j=1..4): 3 k-groups * 14 terms/group = 42 terms
    # Total Rank = 48
    
    R = 48
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    # C[i, k] += A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {1, 2, 3, 4}
    # Split into two 2x2 blocks for A and B:
    # Block 1: j in {1, 2}
    # Block 2: j in {3, 4}
    # Split k into 3 groups of 2: {0,1}, {2,3}, {4,5}
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Each M contributes to C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add 14 terms for each k-group.
    # For each term, we define the coefficients for A (2x2) and B (2x2).
    # Then we map to global indices.
    
    strassen_terms = [
        # (coeffs for A: [A00, A01, A10, A11], coeffs for B: [B00, B01, B10, B11], coeffs for C: [C00, C01, C10, C11])
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 1, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0])
    ]
    
    for r in range(3):  # k groups
        for s in range(2):  # j blocks (0: j=1,2; 1: j=3,4)
            # Base global j offset for this block
            j_base = 1 + s * 2
            
            for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                # a_coeffs: [A00, A01, A10, A11]
                # b_coeffs: [B00, B01, B10, B11]
                # c_coeffs: [C00, C01, C10, C11]
                
                # Add to U, V, W
                # U indices: i*m + j
                # V indices: j*p + k
                # W indices: k*n + i
                
                # We need to distribute the coefficients.
                # This is a rank-1 term in the 2x2x2 subspace.
                # The term is sum_{i,j,k} U_{ij} V_{jk} W_{ki}
                # Here, the "tensor" is the Strassen product.
                # Actually, each Strassen term M is a product of a linear form in A and a linear form in B.
                # M = (sum a_i A_i) * (sum b_j B_j)
                # This contributes to C via the c_coeffs.
                # Wait, the standard decomposition expresses C_{ij} = sum_k U_{ik} V_{kj} W_k.
                # Here, for the 2x2x2 block, we have 7 rank-1 terms.
                # Each term corresponds to a specific linear combination of A and B, and a specific linear combination of C.
                # However, the standard tensor decomposition requires U, V, W to be vectors.
                # The Strassen algorithm provides a decomposition where:
                # C = sum_{t=1}^7 (u_t \otimes v_t) w_t^T ? No.
                # C_{ij} = sum_t U_{i,t} V_{j,t} W_{t} ? No, indices don't match.
                
                # Correct form:
                # C_{ij} = sum_t U_{i, t} V_{j, t} W_{t} is for square matrices?
                # No, for matrix mult C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
                # Tensor T_{ijk} = delta_{ij} delta_{jk} (with appropriate indices).
                # Decomposition: T = sum_r u_r \otimes v_r \otimes w_r.
                # u_r is vector in A-space (size n*m), v_r in B-space (size m*p), w_r in C-space (size p*n).
                
                # In Strassen:
                # Each M_t is a scalar.
                # M_t = (linear form in A) * (linear form in B).
                # C = sum_t M_t * (linear form in C? No, C is linear in M_t).
                # Actually, C_{ij} = sum_t coeff_{ij,t} M_t.
                # So, w_r (vector in C-space) has entries coeff_{ij,t}.
                # u_r (vector in A-space) has entries from the linear form in A.
                # v_r (vector in B-space) has entries from the linear form in B.
                
                # So for each Strassen term t:
                # u_t is the vector of coefficients for A.
                # v_t is the vector of coefficients for B.
                # w_t is the vector of coefficients for C.
                
                # We just need to populate U, V, W with these coefficients.
                
                # A coefficients: a_coeffs = [A00, A01, A10, A11]
                # These correspond to local indices (0,0), (0,1), (1,0), (1,1).
                # Global A indices:
                # (0,0) -> i=0, j=j_base+0
                # (0,1) -> i=0, j=j_base+1
                # (1,0) -> i=1, j=j_base+0
                # (1,1) -> i=1, j=j_base+1
                
                U_indices = [
                    0 * m + (j_base + 0),
                    0 * m + (j_base + 1),
                    1 * m + (j_base + 0),
                    1 * m + (j_base + 1)
                ]
                
                # B coefficients: b_coeffs = [B00, B01, B10, B11]
                # These correspond to local indices (0,0), (0,1), (1,0), (1,1) for B.
                # B_{local_j, local_k}.
                # Global B indices:
                # (0,0) -> j=j_base+0, k=2r+0
                # (0,1) -> j=j_base+0, k=2r+1
                # (1,0) -> j=j_base+1, k=2r+0
                # (1,1) -> j=j_base+1, k=2r+1
                
                V_indices = [
                    (j_base + 0) * p + (2 * r + 0),
                    (j_base + 0) * p + (2 * r + 1),
                    (j_base + 1) * p + (2 * r + 0),
                    (j_base + 1) * p + (2 * r + 1)
                ]
                
                # C coefficients: c_coeffs = [C00, C01, C10, C11]
                # These correspond to local indices (0,0), (0,1), (1,0), (1,1) for C.
                # C_{local_i, local_k}.
                # Global C indices:
                # (0,0) -> i=0, k=2r+0
                # (0,1) -> i=0, k=2r+1
                # (1,0) -> i=1, k=2r+0
                # (1,1) -> i=1, k=2r+1
                
                W_indices = [
                    (2 * r + 0) * n + 0,
                    (2 * r + 0) * n + 1,  # Wait, W index is k*n + i.
                    # C00: i=0, k=2r+0 -> (2r+0)*2 + 0
                    # C01: i=0, k=2r+1 -> (2r+1)*2 + 0
                    # C10: i=1, k=2r+0 -> (2r+0)*2 + 1
                    # C11: i=1, k=2r+1 -> (2r+1)*2 + 1
                ]
                # Let's re-verify W indices.
                # W[k*n + i]
                # c_coeffs[0] is C00 (i=0, k_local=0 -> k_global=2r+0)
                # c_coeffs[1] is C01 (i=0, k_local=1 -> k_global=2r+1)
                # c_coeffs[2] is C10 (i=1, k_local=0 -> k_global=2r+0)
                # c_coeffs[3] is C11 (i=1, k_local=1 -> k_global=2r+1)
                
                W_indices = [
                    (2 * r + 0) * n + 0,
                    (2 * r + 1) * n + 0,
                    (2 * r + 0) * n + 1,
                    (2 * r + 1) * n + 1
                ]
                
                # Populate U, V, W
                for idx, coeff in enumerate(a_coeffs):
                    if coeff != 0:
                        U[U_indices[idx]][col] = coeff
                for idx, coeff in enumerate(b_coeffs):
                    if coeff != 0:
                        V[V_indices[idx]][col] = coeff
                for idx, coeff in enumerate(c_coeffs):
                    if coeff != 0:
                        W[W_indices[idx]][col] = coeff
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
