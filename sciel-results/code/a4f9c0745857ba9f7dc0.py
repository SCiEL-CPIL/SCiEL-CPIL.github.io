def _generate_strassen_terms(n, level=0):
    """
    Generate all scalar multiplication terms for n×n × n×n using Strassen.
    Returns a list of (u_dict, v_dict, w_dict) where each dict maps index to coefficient.
    u_dict: {A_index: coeff}, v_dict: {B_index: coeff}, w_dict: {C_index: coeff}
    """
    if n == 1:
        # Single scalar multiplication
        return [({0: 1}, {0: 1}, {0: 1})]
    
    n2 = n // 2
    # Strassen's 7 products for n×n × n×n:
    # P1 = (A11+A22) * (B11+B22)
    # P2 = (A21+A22) * B11
    # P3 = A11 * (B12-B22)
    # P4 = A22 * (B21-B11)
    # P5 = (A11+A12) * B22
    # P6 = (A21-A11) * (B11+B12)
    # P7 = (A12-A22) * (B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Define the 7 sub-problems with their A and B linear combination coefficients
    # and their output combination coefficients
    
    # For each sub-problem, we need:
    # - A_coeff: dict mapping (block_row, block_col, entry) to coefficient
    # - B_coeff: dict mapping (block_row, block_col, entry) to coefficient
    # - C_coeff: dict mapping (block_row, block_col, entry) to coefficient
    
    # Let me define the sub-problems:
    # Each sub-problem is a 2×2 block multiplication
    # A blocks: A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right)
    # B blocks: B11 (top-left), B12 (top-right), B21 (bottom-left), B22 (bottom-right)
    # C blocks: C11, C12, C21, C22
    
    sub_problems = [
        # (A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
        # P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # P3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # P4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    all_terms = []
    for A_blocks, B_blocks, C_blocks in sub_problems:
        # Get the sub-decomposition for n2×n2 × n2×n2
        sub_terms = _generate_strassen_terms(n2, level+1)
        
        for u_sub, v_sub, w_sub in sub_terms:
            # Compose: u = sum over A_blocks of coeff * u_sub shifted to block
            # v = sum over B_blocks of coeff * v_sub shifted to block
            # w = sum over C_blocks of coeff * w_sub shifted to block
            
            u_new = {}
            for (br, bc), coeff_A in A_blocks.items():
                for idx, coeff_sub in u_sub.items():
                    # Map sub-index to full index
                    # Sub-index idx corresponds to (sr, sc) in the n2×n2 block
                    sr, sc = divmod(idx, n2)
                    full_r = br * n2 + sr
                    full_c = bc * n2 + sc
                    full_idx = full_r * n + full_c
                    u_new[full_idx] = u_new.get(full_idx, 0) + coeff_A * coeff_sub
            
            v_new = {}
            for (br, bc), coeff_B in B_blocks.items():
                for idx, coeff_sub in v_sub.items():
                    sr, sc = divmod(idx, n2)
                    full_r = br * n2 + sr
                    full_c = bc * n2 + sc
                    full_idx = full_r * n + full_c
                    v_new[full_idx] = v_new.get(full_idx, 0) + coeff_B * coeff_sub
            
            w_new = {}
            for (br, bc), coeff_C in C_blocks.items():
                for idx, coeff_sub in w_sub.items():
                    # w_sub index: k*n2+i in the sub-problem
                    # Full index: k*n+i
                    sk, si = divmod(idx, n2)
                    full_k = bc * n2 + sk  # Wait, C block (br,bc) means row br, col bc
                    # C[i,k] where i is row, k is col
                    # In the sub-problem, w index = k_sub * n2 + i_sub
                    # Full: k_full * n + i_full
                    # i_full = br * n2 + i_sub
                    # k_full = bc * n2 + k_sub
                    full_idx = (bc * n2 + sk) * n + (br * n2 + si)
                    w_new[full_idx] = w_new.get(full_idx, 0) + coeff_C * coeff_sub
            
            all_terms.append((u_new, v_new, w_new))
    
    return all_terms
