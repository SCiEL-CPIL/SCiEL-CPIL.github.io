def synthesize(matrix, depth_limit):
    import numpy as np
    
    A = np.array(matrix, dtype=np.uint8)
    m, n = A.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Perform Gaussian elimination over GF(2) to get RREF
    # and track row operations
    M = A.copy()
    row_ops = [list(range(m))]  # Each row of M is a combination of original rows
    # Actually, let's just compute RREF and use it directly
    
    # For synthesis, we just need to compute each row of A
    # The RREF approach: compute RREF, then for each original row, 
    # it's a linear combination of RREF rows
    
    # Simpler: just synthesize each row directly
    gates = []
    outputs = []
    next_wire = n + 1
    
    for i in range(m):
        row = A[i]
        cols = np.where(row == 1)[0]
        
        if len(cols) == 0:
            outputs.append(n)
        elif len(cols) == 1:
            outputs.append(int(cols[0]))
        else:
            current = int(cols[0])
            for j in range(1, len(cols)):
                gates.append([current, int(cols[j])])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
