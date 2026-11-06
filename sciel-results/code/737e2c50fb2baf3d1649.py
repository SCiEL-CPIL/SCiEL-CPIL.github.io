def synthesize(matrix, depth_limit):
    import numpy as np
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    A = np.array(matrix, dtype=np.uint8) % 2
    
    # Compute RREF with row tracking
    # We'll maintain an identity matrix to track how rows combine
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)  # T[i] tells us how row i of M is a combination of original rows
    
    rank = 0
    pivot_cols = []
    
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap rows
        M[[rank, pivot]] = M[[pivot, rank]]
        T[[rank, pivot]] = T[[pivot, rank]]
        
        # Eliminate
        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                T[row] ^= T[rank]
        
        pivot_cols.append(col)
        rank += 1
        if rank == m:
            break
    
    # Now M is in RREF.
    # The first 'rank' rows of M are the basis.
    # T[i] for i in 0..rank-1 tells us which original rows combine to form basis row i.
    # But we want the opposite: express each original row as a combination of basis rows.
    
    # Actually, T is the transformation: M = T * A_orig
    # So A_orig = T^{-1} * M
    # Since T is invertible (product of swaps and XORs), T^{-1} exists.
    
    # Alternative: just use the RREF structure directly.
    # In RREF, row i (0 <= i < rank) has a 1 at pivot_cols[i] and 0s in other pivot columns.
    # For the original matrix, row j can be expressed as XOR of certain RREF rows.
    
    # Let me think differently. The RREF tells us:
    # - Which columns are pivot columns
    # - For each row, which pivot columns have 1s
    
    # For the circuit:
    # 1. Compute each basis vector (pivot row) as a gate
    # 2. Express each output as XOR of basis vectors
    
    # In RREF M, row i has 1s in positions where M[i,j]=1.
    # For i < rank: M[i, pivot_cols[i]] = 1, and M[i, pivot_cols[k]] = 0 for k != i.
    # But M[i,j] may be 1 for non-pivot columns j.
    
    # The key insight: the original row j equals XOR of RREF rows i where the "coefficient" is 1.
    # This coefficient is given by... hmm, I need to invert the transformation.
    
    # Simpler: just use the fact that in RREF, the non-pivot parts tell us the dependencies.
    
    # Let me just build the circuit directly from the RREF:
    
    gates = []
    
    # Step 1: For each pivot row i (0..rank-1), compute the wire
    # M[i] has 1s in certain columns. XOR those input wires.
    pivot_wires = []
    for i in range(rank):
        ones = [j for j in range(n) if M[i, j] == 1]
        if len(ones) == 0:
            pivot_wires.append(n)
        elif len(ones) == 1:
            pivot_wires.append(ones[0])
        else:
            current = ones[0]
            for j in range(1, len(ones)):
                g = len(gates)
                gates.append([current, ones[j]])
                current = n + 1 + g
            pivot_wires.append(current)
    
    # Step 2: For each output row, determine which pivot wires to XOR
    # We need to express original row j in terms of RREF rows.
    # Since M = T * A, and M is RREF, we have A = T^{-1} * M.
    # So original row j = XOR of RREF rows i where T^{-1}[j,i] = 1.
    
    # Compute T^{-1}
    # T is m x m, invertible. Invert over GF(2).
    # Augment [T | I] and do row reduction to get [I | T^{-1}]
    
    T_inv = np.zeros((m, m), dtype=np.uint8)
    aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
    for col in range(m):
        pivot = -1
        for row in range(col, m):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(m):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    T_inv = aug[:, m:]
    
    # Now original row j = XOR of RREF rows i where T_inv[j, i] = 1
    outputs = []
    for j in range(m):
        # Find which RREF rows contribute
        contributing = [i for i in range(rank) if T_inv[j, i] == 1]
        # Note: T_inv[j, i] for i >= rank should be 0 since M has 0 rows there
        
        if len(contributing) == 0:
            outputs.append(n)  # constant zero
        elif len(contributing) == 1:
            outputs.append(pivot_wires[contributing[0]])
        else:
            current = pivot_wires[contributing[0]]
            for i in range(1, len(contributing)):
                g = len(gates)
                gates.append([current, pivot_wires[contributing[i]]])
                current = n + 1 + g
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
