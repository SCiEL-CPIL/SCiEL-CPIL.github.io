"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A is n x m (2x2), B is m x p (2x8), C is n x p (2x8)
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (16, rank)
    
    # The task specifies W[k*n+i, r] where i is row of C (0..n-1) and k is col of C (0..p-1).
    # Standard flattening for C[i, k] is i*p + k.
    # But the task says W[k*n+i, r]. This implies the index in W corresponds to (k, i) flattened as k*n + i.
    # So W index for C[i, k] is k*n + i.
    
    # Similarly for U and V, we need to determine the flattening.
    # Usually U is for A (n x m), V is for B (m x p).
    # A[i, j] index? Likely i*m + j.
    # B[j, k] index? Likely j*p + k.
    
    # Let's verify the tensor definition.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Tensor T_{i, j, k} = 1 if the multiplication contributes, 0 otherwise.
    # Actually, the tensor is T_{i, j, k} such that C[i, k] = sum_{j} A[i, j] B[j, k].
    # The rank-r decomposition is:
    # C[i, k] = sum_{s=1}^r (sum_j U[i, j, s] A[i, j]) * (sum_l V[j, l, s] B[j, l]) * W[i, k, s] ? No.
    # The standard form is:
    # C[i, k] = sum_{j, l} A[i, j] B[j, k] delta_{j,l}
    # Decomposition:
    # C[i, k] = sum_{s} (sum_{i', j'} U_{i, j', s} A[i', j']? No.
    
    # Let's use the explicit bilinear form.
    # C[i, k] = sum_{j} A[i, j] B[j, k]
    # We want to write this as sum_{s=1}^r u_s A \otimes v_s B \otimes w_s C?
    # Specifically:
    # C[i, k] = sum_{s=1}^r ( \sum_{j'} U_{i, j', s} A[i', j'] ) ( \sum_{l'} V_{j', l', s} B[j', l'] ) W_{i, k, s} ?? No.
    
    # Correct tensor decomposition definition:
    # The multiplication tensor T is a 3-way tensor with indices (i, j, k) where:
    # i: row of A (0..n-1)
    # j: col of A / row of B (0..m-1)
    # k: col of B (0..p-1)
    # T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # Actually, the tensor is defined such that:
    # C[i, k] = sum_{j} A[i, j] B[j, k]
    # This can be written as:
    # C[i, k] = sum_{j, l} A[i, j] B[l, k] T[i, j, l, k] ? No, it's a 3-tensor if we contract j.
    
    # The standard matrix multiplication tensor for sizes (n, m, p) has shape (n*m, m*p, n*p) ? No.
    # The tensor is of size n*m*p with entries T_{i, j, k} where:
    # i in [0, n*m) corresponds to (row_A, col_A)
    # j in [0, m*p) corresponds to (row_B, col_B)
    # k in [0, n*p) corresponds to (row_C, col_C)
    # T[i, j, k] = 1 if col_A == row_B AND the product contributes to C[row_C, col_C].
    
    # Let's stick to the algorithmic decomposition which is safer.
    # We are decomposing the operation C = A @ B.
    # A is 2x2. B is 2x8.
    # We can split B into four 2x2 blocks: B = [B_0, B_1, B_2, B_3] where B_b is 2x2.
    # C = A @ [B_0, B_1, B_2, B_3] = [A B_0, A B_1, A B_2, A B_3].
    # Each A B_b is a 2x2 matrix product.
    # Rank of 2x2 matrix product is 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Strassen decomposition for 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 + a00) * (b00 + b01)
    # M7 = (a11 + a01) * (b10 + b11)
    
    # c00 = M1 - M4 + M6 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 - M5
    
    # We need to construct U, V, W for the whole system.
    # U has shape (4, 28). Rows correspond to A elements (a00, a01, a10, a11).
    # V has shape (16, 28). Rows correspond to B elements.
    #   B is 2x8. Elements are B[j, k] for j in [0,1], k in [0..7].
    #   Index in V: j * 8 + k.
    # W has shape (16, 28). Rows correspond to C elements.
    #   C is 2x8. Elements are C[i, k] for i in [0,1], k in [0..7].
    #   Index in W: k * 2 + i (per task contract W[k*n+i, r]).
    
    rank = 28
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    # Strassen coefficients
    # For each step s (0..6), we have linear forms for A, B, and C.
    # alpha_s: coefficients for A (4 elements)
    # beta_s: coefficients for B (4 elements for a 2x2 block)
    # gamma_s: coefficients for C (4 elements for a 2x2 block)
    
    # A indices: 0:a00, 1:a01, 2:a10, 3:a11
    # B block indices (local): 0:b00, 1:b01, 2:b10, 3:b11
    # C block indices (local): 0:c00, 1:c01, 2:c10, 3:c11
    
    alphas = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, 1, 0],  # M6
        [0, 1, 0, 1]   # M7
    ]
    
    betas = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    gammas = [
        [1, 0, 0, 1],   # M1: c00=1, c11=1
        [0, 0, 1, -1],  # M2: c10=1, c11=-1
        [0, 1, 0, -1],  # M3: c01=1, c11=-1
        [-1, 0, 1, 0],  # M4: c00=-1, c10=1
        [0, 1, 0, -1],  # M5: c01=1, c11=-1
        [1, 0, 0, 0],   # M6: c00=1
        [-1, 0, 0, 0]   # M7: c00=-1
    ]
    
    # Now fill U, V, W for each of the 4 blocks.
    for b in range(4):
        # b is the block index (0..3)
        # B block b consists of B[0, 2*b], B[0, 2*b+1], B[1, 2*b], B[1, 2*b+1]
        # C block b consists of C[0, 2*b], C[0, 2*b+1], C[1, 2*b], C[1, 2*b+1]
        
        for s in range(7):
            r = 7 * b + s
            
            # Fill U (independent of block b, same A)
            for idx_A in range(4):
                U[idx_A][r] = alphas[s][idx_A]
            
            # Fill V
            # Local B index l (0..3) maps to global B[j, k]
            # l=0 -> j=0, k=2*b
            # l=1 -> j=0, k=2*b+1
            # l=2 -> j=1, k=2*b
            # l=3 -> j=1, k=2*b+1
            for l in range(4):
                j = l // 2
                k_local = l % 2
                k_global = 2 * b + k_local
                v_idx = j * 8 + k_global
                V[v_idx][r] = betas[s][l]
            
            # Fill W
            # Local C index l (0..3) maps to global C[i, k]
            # l=0 -> i=0, k=2*b
            # l=1 -> i=0, k=2*b+1
            # l=2 -> i=1, k=2*b
            # l=3 -> i=1, k=2*b+1
            for l in range(4):
                i = l // 2
                k_local = l % 2
                k_global = 2 * b + k_local
                w_idx = k_global * 2 + i
                W[w_idx][r] = gammas[s][l]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
