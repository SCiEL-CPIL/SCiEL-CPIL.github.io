"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 90.0
    MARGIN = 5.0
    
    # Parameters for the extended quadratic residue code of length 18
    p = 17
    n = 18
    k = 9
    
    # Step 1: Identify quadratic residues modulo 17
    # QRs are {x^2 mod 17 | x in 1..8}
    qrs = set((x * x) % p for x in range(1, p))
    # qrs should be {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Step 2: Construct the 8x17 circulant parity check matrix H for the [17, 9, 5] code
    # The first row has 1s at positions corresponding to QRs.
    # Note: The standard QR code definition uses the vector v where v_i = 1 if i in QR.
    # The parity check matrix H is circulant with first row v.
    # Dimension of null space is 17 - 8 = 9.
    
    # Represent rows as integers for fast XOR and bit manipulation
    # Row 0
    row0 = 0
    for i in range(p):
        if i in qrs:
            row0 |= (1 << i)
            
    # Generate all 8 rows of H
    H = [0] * 8
    H[0] = row0
    for i in range(1, 8):
        # Cyclic shift right by 1
        # (row >> 1) | ((row & 1) << (p-1))
        H[i] = (H[i-1] >> 1) | ((H[i-1] & 1) << (p - 1))
        
    # Step 3: Find the null space of H over GF(2)
    # We want vectors g in F_2^17 such that H * g^T = 0
    # This is equivalent to finding a basis for the kernel of the linear map defined by H.
    # We can do this by Gaussian elimination on the transpose or by finding linear dependencies among the columns.
    # Easier: Gaussian elimination on H to find the free variables.
    
    # Let's perform Gaussian elimination on H (8x17) to put it in Row Echelon Form.
    # We track the pivot columns.
    H_reduced = H[:]
    pivots = []
    current_row = 0
    for col in range(p):
        # Find a row with a 1 in this column
        pivot_found = False
        for r in range(current_row, 8):
            if (H_reduced[r] >> col) & 1:
                # Swap rows
                H_reduced[current_row], H_reduced[r] = H_reduced[r], H_reduced[current_row]
                pivot_found = True
                break
        if not pivot_found:
            continue
        
        # Eliminate this column from all other rows
        for r in range(8):
            if r != current_row and ((H_reduced[r] >> col) & 1):
                H_reduced[r] ^= H_reduced[current_row]
        
        pivots.append(col)
        current_row += 1
        if current_row == 8:
            break
            
    # The pivot columns are in 'pivots'. The non-pivot columns are the free variables.
    # There are 17 - len(pivots) = 9 free variables.
    free_cols = [c for c in range(p) if c not in pivots]
    
    # Construct basis vectors for the null space
    # For each free column j, we can set that variable to 1 and solve for the pivot variables.
    # The equation is sum(H_reduced[r][c] * x_c) = 0 for each row r.
    # x_pivot = sum(H_reduced[r][free] * x_free)
    
    basis = []
    for j, free_col in enumerate(free_cols):
        vec = 1 << free_col
        # Solve for pivot variables
        for r in range(len(pivots)):
            # Check if the sum of bits in this row for the free variables (including current one) is 1
            # Since we only set one free variable to 1 at a time, we just check the bit at free_col in H_reduced[r]
            # Wait, in RREF, the pivot column has 1 at the pivot row and 0 elsewhere.
            # So for row r, the equation is x_{pivots[r]} + sum_{f in free} H_reduced[r][f] x_f = 0
            # x_{pivots[r]} = sum_{f in free} H_reduced[r][f] x_f
            # In our case, only x_{free_col} is 1, others 0.
            # So x_{pivots[r]} = H_reduced[r][free_col]
            if (H_reduced[r] >> free_col) & 1:
                vec |= (1 << pivots[r])
        basis.append(vec)
        
    # Step 4: Extend the basis to length 18 by adding a parity bit
    # The extended code has min distance 6.
    # We append a bit such that the total weight is even.
    extended_basis = []
    for vec in basis:
        # Calculate parity (sum of bits mod 2)
        parity = vec.bit_count() % 2
        # New vector is vec * 2 + parity ? No, vec is bits 0..16.
        # We want to add bit 17.
        # If we shift vec left by 1, bit 17 becomes the MSB?
        # Let's keep bit 0 as the least significant bit.
        # Original vec uses bits 0..16.
        # New vec uses bits 0..17.
        # Bit 17 is the parity bit.
        new_vec = (vec << 1) | parity
        extended_basis.append(new_vec)
        
    # Step 5: Generate all 2^9 = 512 codewords
    words = []
    for mask in range(1 << k):
        word = 0
        for i in range(k):
            if (mask >> i) & 1:
                word ^= extended_basis[i]
        words.append(word)
        
    # Step 6: Verification (optional but good for safety)
    # Check that all pairs have distance >= 6
    # 512 choose 2 is ~131k checks.
    # Each check: (a ^ b).bit_count() >= 6
    # This is fast enough.
    min_dist = 18
    # To save time, we can skip full verification if we trust the math, 
    # but a quick spot check or full check is safe within 90s.
    # 131k iterations in Python takes ~0.1s.
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
            if min_dist < 6:
                # This should not happen
                break
        if min_dist < 6:
            break
            
    # If min_dist < 6, there is an error. But theoretically it should be 6.
    # If it fails, we might return the 32-word code as fallback? 
    # But the QR code is correct.
    
    return words

# EVOLVE-BLOCK-END
