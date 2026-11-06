def synthesize(matrix, depth_limit):
    import numpy as np
    
    mat = np.array(matrix, dtype=int) % 2
    m, n = mat.shape
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # RREF over GF(2)
    rref = mat.copy()
    rows, cols = rref.shape
    rank = 0
    pivot_cols = []
    
    for col in range(cols):
        pivot_row = None
        for row in range(rank, rows):
            if rref[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        rref[[rank, pivot_row]] = rref[[pivot_row, rank]]
        pivot_cols.append(col)
        
        for row in range(rows):
            if row != rank and rref[row, col] == 1:
                rref[row] = (rref[row] + rref[rank]) % 2
        
        rank += 1
        if rank == rows:
            break
    
    gates = []
    outputs = []
    
    for i in range(rank):
        ones = np.where(rref[i] == 1)[0]
        if len(ones) == 0:
            outputs.append(n)
            continue
        elif len(ones) == 1:
            outputs.append(int(ones[0]))
            continue
        else:
            wires_to_xor = [int(w) for w in ones]
            current = wires_to_xor[0]
            for j in range(1, len(wires_to_xor)):
                gate_idx = len(gates)
                new_wire = n + 1 + gate_idx
                gates.append([current, wires_to_xor[j]])
                current = new_wire
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
