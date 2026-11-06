"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like of 0s and 1s (n x m)
        depth_limit: maximum circuit depth (int)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # Convert to numpy array
    mat = np.array(matrix, dtype=np.int8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    n, m = mat.shape
    
    # Input wires: 0..n-1, constant-zero wire: n
    # Gate k produces wire n+1+k
    
    # Strategy: For each output column, we want to express it as an XOR of inputs.
    # We use Gaussian elimination to find a sparse representation.
    # 
    # Key insight: We can share intermediate XOR results across multiple outputs.
    # We'll use a greedy approach:
    # 1. Compute all column vectors
    # 2. Use a set-cover-like greedy to find shared sub-expressions
    # 3. Build the circuit
    
    # First, let's try a direct approach: express each column as XOR of input rows
    # using sparse Gaussian elimination
    
    # Work in GF(2). Each column is a vector in GF(2)^n.
    # We want to express each column vector as a linear combination of the standard basis vectors
    # (which correspond to input wires 0..n-1).
    
    # Actually, the matrix IS the set of output expressions. mat[i,j] = 1 means output j uses input i.
    # So the "circuit" is trivially: output j = XOR of inputs i where mat[i,j]=1.
    # But we can do better by sharing intermediate XORs.
    
    # Let's think about this differently:
    # We have m output expressions, each being an XOR of some subset of the n inputs.
    # We can introduce intermediate gates to share common sub-expressions.
    
    # Approach: 
    # 1. Find frequent sub-expressions (pairs of inputs that appear together in many outputs)
    # 2. Create gates for those sub-expressions
    # 3. Express each output using the gates and remaining inputs
    
    # Count how often each pair of inputs co-occurs across all output columns
    # This is a heuristic for finding useful shared sub-expressions
    
    # For efficiency, work with column vectors
    cols = mat.T  # shape (m, n), each row is an output expression
    
    # Find all pairs (i, j) with i < j that appear together in at least 2 outputs
    # and where replacing them saves at least 1 gate per occurrence
    # 
    # Actually, let's use a simpler but effective approach:
    # Greedy set cover on "useful sub-expressions"
    
    # A sub-expression is a vector v in GF(2)^n with weight 2 (or 1, but those are just inputs).
    # Creating a gate for v costs 1 gate. It helps any output that contains both bits of v.
    # For an output with weight w that contains v, it becomes weight w-1 (using the gate + the other bits... wait no).
    # If output = v XOR u (where u is the rest), then output = gate(v) XOR u, so weight becomes 1 + weight(u) = 1 + (w - 2) = w - 1.
    # So we save 1 gate per output that uses this sub-expression.
    
    # But we also need to consider that using a gate might prevent using another gate.
    
    # Let's use a different, more principled approach:
    # Gaussian elimination to find a basis, then express everything in terms of that basis.
    # But the basis vectors are the inputs themselves, so that doesn't help directly.
    
    # Better approach: Use the structure of the matrix.
    # 
    # Let's try: for each output, find its sparsest representation.
    # The sparsest representation of a vector v in GF(2)^n using the standard basis is just v itself (weight = popcount(v)).
    # To do better, we need to introduce new basis vectors (gates).
    
    # Greedy algorithm:
    # 1. Start with no gates. Each output j has expression = set of inputs where mat[:,j]=1.
    # 2. Repeatedly find the pair (i,j) of inputs that appears in the most outputs (where both i and j are in the expression).
    #    If creating a gate for (i,j) reduces the total gate count, do it.
    # 3. Stop when no more improvement is possible or time limit.
    
    # Let's implement this:
    
    # Current expressions: for each output j, a set of wire indices
    # Initially, expression[j] = {i : mat[i,j] == 1}
    
    # Represent expressions as bitmasks for efficiency (n <= 16 in our test cases, so fits in int)
    # But n could be larger. Let's use Python sets or numpy arrays.
    
    # For n up to 16, we can use Python ints as bitmasks.
    if n <= 60:
        # Use integer bitmasks
        exprs = [0] * m
        for j in range(m):
            mask = 0
            for i in range(n):
                if mat[i, j]:
                    mask |= (1 << i)
            exprs[j] = mask
        
        # Gates: list of (wire1, wire2, gate_wire) where gate_wire = n + 1 + k
        gates = []
        gate_masks = []  # bitmask of each gate's value
        
        def popcount(x):
            return bin(x).count('1')
        
        # Current total cost: sum of popcounts of all expressions (inputs used) + number of gates
        # Wait, the cost is the number of gates. Inputs are free.
        # Actually, re-reading the task: "Minimize mean gate-count / independent-row baseline"
        # So we want to minimize the number of gates.
        
        # Current number of gates: len(gates)
        # But we also need to account for the fact that using a gate in an expression doesn't add to the gate count.
        # The gate count is just the number of XOR gates we create.
        
        # Let me re-think. The circuit has:
        # - Input wires 0..n-1 (free)
        # - Constant zero wire n (free)
        # - Gate wires n+1, n+2, ... (each costs 1 gate)
        # Each gate is an XOR of two wires.
        # Outputs are wires.
        
        # So the total gate count is the number of gates we create.
        # We want to minimize this.
        
        # The "independent-row baseline" is presumably the number of gates if we don't share anything.
        # If we express each output directly as XOR of inputs, we might need multiple gates per output
        # if the output has weight > 2 (since each gate only XORs two wires).
        
        # Wait, I think I'm overcomplicating this. Let me re-read.
        
        # "gate k (zero-based) produces wire n+1+k"
        # "gates": [[wire, wire], ...]
        # So each gate takes two wires and produces a new wire.
        # To compute an XOR of k inputs, you need k-1 gates (a tree of XORs).
        
        # So the baseline for an output with weight w is w-1 gates (if w>0), or 0 if w=0.
        # But if we share sub-expressions, we can reduce the total.
        
        # Let me reconsider the problem.
        
        # Actually, I think the "independent-row baseline" refers to computing each output independently
        # without sharing. For an output with weight w, you need w-1 gates.
        # Total baseline = sum over outputs of max(0, weight(output) - 1).
        
        # We want to minimize the actual gate count by sharing.
        
        # Let's implement a greedy approach:
        # 1. Start with each output expressed as a set of input wires.
        # 2. Find pairs of wires that appear together in multiple outputs.
        # 3. Create a gate for such a pair if it reduces the total gate count.
        
        # Total gate count calculation:
        # If we have a set of gates, and each output is expressed as an XOR of some gates and inputs,
        # the total gate count is the number of gates + the number of "extra" XORs needed to combine
        # the terms in each output expression.
        
        # Hmm, this is getting complex. Let me try a different approach.
        
        # Simpler approach: Just compute each output as a tree of XORs, and try to share
        # common subtrees.
        
        # Actually, let me just implement a straightforward solution first and see if it beats the baseline.
        
        # For each output j:
        #   Let S_j = {i : mat[i,j] = 1}
        #   If |S_j| = 0: output j = wire n (zero)
        #   If |S_j| = 1: output j = the single input
        #   If |S_j| > 1: build a tree of XORs
        #
        # To share: if two outputs share a common subset of inputs, we can compute that subset once.
        
        # Let's use a dictionary to track computed sub-expressions.
        
        # Map from bitmask to wire index
        wire_map = {}
        # Input wires
        for i in range(n):
            wire_map[1 << i] = i
        wire_map[0] = n  # zero wire
        
        next_wire = n + 1
        gates = []
        
        def get_wire(mask):
            """Get the wire for the given bitmask, creating gates if needed."""
            nonlocal next_wire
            if mask in wire_map:
                return wire_map[mask]
            if mask == 0:
                return n
            
            # Find the lowest set bit
            low_bit = mask & (-mask)
            rest = mask ^ low_bit
            
            w1 = get_wire(low_bit)
            w2 = get_wire(rest)
            
            gate_idx = len(gates)
            gates.append([w1, w2])
            wire = n + 1 + gate_idx
            wire_map[mask] = wire
            next_wire = n + 1 + gate_idx + 1
            return wire
        
        # Compute wires for all outputs
        outputs = []
        for j in range(m):
            mask = 0
            for i in range(n):
                if mat[i, j]:
                    mask |= (1 << i)
            outputs.append(get_wire(mask))
        
        # Check time
        if time.time() > deadline:
            return {'gates': gates, 'outputs': outputs}
        
        # Now try to improve by sharing.
        # The above already shares common sub-expressions via the wire_map.
        # But the order of decomposition matters. Let's try different orderings.
        
        # Actually, the recursive decomposition always splits off the lowest bit, which is fine.
        # The sharing happens automatically if two outputs have common sub-masks.
        
        # Let's verify: if output 1 has mask {a,b,c} and output 2 has mask {a,b,d},
        # then get_wire({a,b,c}) will compute get_wire({a,b}) and get_wire({c}),
        # and get_wire({a,b,d}) will compute get_wire({a,b}) and get_wire({d}).
        # So {a,b} is computed once and shared. 
        
        # This should be a good solution. Let's also try to optimize by
        # choosing the decomposition order to maximize sharing.
        
        # For now, let's return this solution.
        
        return {'gates': gates, 'outputs': outputs}
    else:
        # For large n, use a simpler approach
        gates = []
        outputs = []
        
        # For each output, build a tree of XORs
        # No sharing for now
        for j in range(m):
            inputs_for_j = [i for i in range(n) if mat[i, j]]
            if len(inputs_for_j) == 0:
                outputs.append(n)
            elif len(inputs_for_j) == 1:
                outputs.append(inputs_for_j[0])
            else:
                # Build a tree
                current = inputs_for_j[0]
                for i in range(1, len(inputs_for_j)):
                    gates.append([current, inputs_for_j[i]])
                    current = n + 1 + (len(gates) - 1)
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
