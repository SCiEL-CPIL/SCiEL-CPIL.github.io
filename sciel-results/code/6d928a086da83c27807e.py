def synthesize(matrix, depth_limit):
    import numpy as np
    
    A = np.array(matrix, dtype=np.uint8)
    m, n = A.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert A to list of bitmasks
    # Bit j of row i corresponds to column j
    M = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        M.append(mask)
        
    # T is list of bitmasks, initially Identity
    T = [1 << i for i in range(m)]
    
    pivot_cols = []
    row_idx = 0
    
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for r in range(row_idx, m):
            if M[r] & (1 << col):
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap
        if pivot_row != row_idx:
            M[pivot_row], M[row_idx] = M[row_idx], M[pivot_row]
            T[pivot_row], T[row_idx] = T[row_idx], T[pivot_row]
        
        # Eliminate from all other rows
        for r in range(m):
            if r != row_idx and (M[r] & (1 << col)):
                M[r] ^= M[row_idx]
                T[r] ^= T[row_idx]
        
        pivot_cols.append((row_idx, col))
        row_idx += 1
        if row_idx == m:
            break
            
    # Invert T
    # T is m x m bitmask matrix.
    # We need T_inv such that T @ T_inv = I
    # We can use the same GE method on T augmented with I
    # Or just run GE on T to get RREF and track the inverse?
    # Let's write a helper to invert a bitmask matrix
    
    def invert_bitmask_matrix(mat):
        size = len(mat)
        # Augment: [mat | I]
        # Since we only have row bitmasks, we can't easily augment.
        # We will perform GE on mat and apply same ops to I.
        A_inv = [1 << i for i in range(size)]
        M_inv = mat[:]
        
        for col in range(size):
            pivot_row = -1
            for r in range(col, size):
                if M_inv[r] & (1 << col):
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
            if pivot_row != col:
                M_inv[pivot_row], M_inv[col] = M_inv[col], M_inv[pivot_row]
                A_inv[pivot_row], A_inv[col] = A_inv[col], A_inv[pivot_row]
            for r in range(size):
                if r != col and (M_inv[r] & (1 << col)):
                    M_inv[r] ^= M_inv[col]
                    A_inv[r] ^= A_inv[col]
        return A_inv

    T_inv = invert_bitmask_matrix(T)
    
    # Synthesize
    gates = []
    outputs = []
    next_wire = n + 1
    gate_cache = {}
    
    def get_xor(a, b):
        nonlocal next_wire
        if a == b:
            return n
        key = (min(a, b), max(a, b))
        if key in gate_cache:
            return gate_cache[key]
        wire = next_wire
        gates.append([a, b])
        gate_cache[key] = wire
        next_wire += 1
        return wire
        
    pivot_row_wires = {}
    
    # Synthesize pivot rows
    for r, c in pivot_cols:
        mask = M[r]
        if mask == 0:
            pivot_row_wires[r] = n
        else:
            inputs = []
            for j in range(n):
                if mask & (1 << j):
                    inputs.append(j)
            inputs.sort()
            
            if len(inputs) == 1:
                pivot_row_wires[r] = inputs[0]
            else:
                current = inputs[0]
                for j in range(1, len(inputs)):
                    current = get_xor(current, inputs[j])
                pivot_row_wires[r] = current
                
    # Synthesize outputs
    for i in range(m):
        mask = T_inv[i]
        if mask == 0:
            outputs.append(n)
        else:
            contributing = []
            for r, c in pivot_cols:
                if mask & (1 << r):
                    contributing.append(pivot_row_wires[r])
            
            if len(contributing) == 0:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(contributing[0])
            else:
                # Sort contributing wires? 
                # Wire IDs are arbitrary, but sorting might help cache if they were inputs.
                # But they are wires n+1.., so order doesn't matter for cache key (min,max).
                current = contributing[0]
                for j in range(1, len(contributing)):
                    current = get_xor(current, contributing[j])
                outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
