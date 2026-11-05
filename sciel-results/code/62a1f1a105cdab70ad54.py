def synthesize(matrix, depth_limit):
    import numpy as np
    
    m, n = matrix.shape
    # Work with integer representation for GF(2)
    mat = matrix.astype(np.uint8)
    
    # We'll track: each row is represented as a set of input wire indices
    # But for circuit synthesis, we need to build gates
    
    # Approach: Forward Gaussian elimination
    # Each row starts as a set of input indices
    # When we do row_i ^= row_j, we create a new gate wire
    
    # Let's track rows as lists of (wire_index) where the row is XOR of those wires
    # Initially, row i is just the set of inputs where mat[i,j]=1
    
    # Actually, let's think of it as: each "row state" is a list of wire IDs
    # that XOR together to give that row
    
    # Start: row i has wires [j for j in range(n) if mat[i,j]==1]
    # But that's not a circuit yet - we need gates
    
    # Better: build a circuit where each intermediate is a gate
    # 
    # Let's use a different approach:
    # 1. Find a basis using Gaussian elimination
    # 2. Express each original row as XOR of basis vectors
    # 3. Build gates for the basis vectors, then gates to combine them
    
    # Simpler direct approach:
    # For each output row, we need to compute XOR of certain inputs
    # We can share subexpressions
    
    # Let me use a practical heuristic:
    # 1. Do forward elimination, each step creates a gate
    # 2. The final nonzero rows are our outputs (or we need to map back)
    
    # Actually, the cleanest approach:
    # - We have m rows to compute
    # - Use Gaussian elimination to get row echelon form
    # - Each elimination step is a gate
    # - The RREF rows tell us which original inputs to XOR
    
    # Let me implement a direct circuit builder:
    
    # Each "expression" is a list of wire IDs that XOR to give the value
    # Initially, input wire j has expression [j]
    # A gate combining expr_a and expr_b creates wire w with expr_a + expr_b (symmetric diff)
    
    # We want to compute each row of the matrix
    
    # Greedy approach: for each row, compute it from inputs, sharing when possible
    
    # Let's use a dictionary: wire_id -> frozenset of input indices
    # This tells us which inputs a wire depends on
    
    wire_to_inputs = {}
    for j in range(n):
        wire_to_inputs[j] = frozenset([j])
    wire_to_inputs[n] = frozenset()  # constant zero
    
    gates = []  # list of [w1, w2]
    next_wire = n + 1
    
    def xor_gates(w1, w2):
        nonlocal next_wire
        # Check if we already have a wire that is XOR of these
        # For simplicity, just create a new gate
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        return w
    
    # For each row, compute the XOR of the input wires where it has a 1
    # To minimize gates, we can chain XORs
    # Row with k ones needs k-1 gates if computed independently
    # But we can share
    
    # Better: process rows in order, and for each row, find the cheapest way
    # to compute it given existing wires
    
    outputs = []
    
    for i in range(m):
        ones = [j for j in range(n) if mat[i, j] == 1]
        
        if len(ones) == 0:
            outputs.append(n)  # constant zero
            continue
        
        if len(ones) == 1:
            outputs.append(ones[0])
            continue
        
        # Compute XOR of all these input wires
        # Chain them: ((a^b)^c)^...
        # But first, check if any existing wire matches a subset
        
        # Simple: just chain
        w = ones[0]
        for j in range(1, len(ones)):
            w = xor_gates(w, ones[j])
        
        outputs.append(w)
    
    return {'gates': gates, 'outputs': outputs}
