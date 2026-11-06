def synthesize(matrix, depth_limit):
    import numpy as np
    
    A = np.array(matrix, dtype=np.uint8)
    m, n = A.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Perform Gaussian elimination over GF(2) to get RREF
    # Track the transformation matrix T such that RREF = T @ A
    M = A.copy().astype(np.int64)
    T = np.eye(m, dtype=np.int64)
    
    pivot_cols = []
    row_idx = 0
    
    for col in range(n):
        # Find a pivot row with 1 in this column
        pivot_row = -1
        for r in range(row_idx, m):
            if M[r, col] % 2 == 1:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows if needed
        if pivot_row != row_idx:
            M[[row_idx, pivot_row]] = M[[pivot_row, row_idx]]
            T[[row_idx, pivot_row]] = T[[pivot_row, row_idx]]
        
        # Eliminate this column from all other rows
        for r in range(m):
            if r != row_idx and M[r, col] % 2 == 1:
                M[r] = (M[r] + M[row_idx]) % 2
                T[r] = (T[r] + T[row_idx]) % 2
        
        pivot_cols.append((row_idx, col))
        row_idx += 1
        
        if row_idx == m:
            break
    
    # Now M is in RREF form
    # T @ A = M, so A = T^{-1} @ M
    # We need to express each original row as a combination of RREF rows
    
    # Invert T over GF(2)
    T_inv = np.eye(m, dtype=np.int64)
    for col in range(m):
        # Find pivot
        pivot_row = -1
        for r in range(col, m):
            if T[r, col] % 2 == 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        if pivot_row != col:
            T[[col, pivot_row]] = T[[pivot_row, col]]
            T_inv[[col, pivot_row]] = T_inv[[pivot_row, col]]
        for r in range(m):
            if r != col and T[r, col] % 2 == 1:
                T[r] = (T[r] + T[col]) % 2
                T_inv[r] = (T_inv[r] + T_inv[col]) % 2
    
    # Now A = T_inv @ M
    # Each row i of A is sum over j where T_inv[i,j] == 1 of M[j]
    
    # Identify which rows of M are pivot rows (non-zero)
    pivot_row_indices = [r for r, c in pivot_cols]
    
    # Synthesize each pivot row of M (RREF)
    # A pivot row has 1 in its pivot column and possibly 1s in free columns
    gates = []
    outputs = []
    next_wire = n + 1
    
    # First, synthesize each pivot row of M using balanced XOR trees
    # to minimize depth
    pivot_row_wires = {}
    for r, c in pivot_cols:
        row = M[r]
        cols = np.where(row % 2 == 1)[0]
        
        if len(cols) == 0:
            pivot_row_wires[r] = n  # zero wire
        elif len(cols) == 1:
            pivot_row_wires[r] = int(cols[0])
        else:
            # Use balanced XOR tree for better depth
            cols_list = [int(x) for x in cols]
            # Build balanced tree
            level = cols_list[:]
            while len(level) > 1:
                next_level = []
                for i in range(0, len(level) - 1, 2):
                    gates.append([level[i], level[i+1]])
                    next_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    next_level.append(level[-1])
                level = next_level
            pivot_row_wires[r] = level[0]
    
    # Now synthesize each original row i as XOR of pivot rows where T_inv[i,j] == 1
    # Also use balanced XOR trees
    for i in range(m):
        # Find which pivot rows contribute to row i
        contributing = [r for r in pivot_row_indices if T_inv[i, r] % 2 == 1]
        
        if len(contributing) == 0:
            outputs.append(n)
        elif len(contributing) == 1:
            outputs.append(pivot_row_wires[contributing[0]])
        else:
            # Use balanced XOR tree
            level = [pivot_row_wires[r] for r in contributing]
            while len(level) > 1:
                next_level = []
                for i in range(0, len(level) - 1, 2):
                    gates.append([level[i], level[i+1]])
                    next_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    next_level.append(level[-1])
                level = next_level
            outputs.append(level[0])
    
    return {'gates': gates, 'outputs': outputs}
