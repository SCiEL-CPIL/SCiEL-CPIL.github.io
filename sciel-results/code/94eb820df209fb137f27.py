"""Rewritten program for XOR-circuit synthesis."""

def synthesize(matrix, depth_limit):
    import time
    import numpy as np

    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin

    # 1. Parse input matrix to list of integers (bitmasks)
    # Input wires are 0..n-1.
    n = len(matrix)
    if n == 0:
        return {'gates': [], 'outputs': []}
        
    targets = []
    for row in matrix:
        val = 0
        for j, bit in enumerate(row):
            if bit:
                val |= (1 << j)
        targets.append(val)
    
    # 2. Greedy Basis Extension
    # Initial basis: standard unit vectors
    basis = [1 << i for i in range(n)]
    wire_map = {1 << i: i for i in range(n)}
    gates = []  # List of (wire_u, wire_v) for each new gate
    
    # Helper to calculate cost: sum of (weight of representation - 1) for all targets
    def calculate_cost(basis, targets):
        total_cost = 0
        # We need to find the representation of each target in terms of the basis.
        # Since the basis might not be in RREF, we perform Gaussian elimination.
        # For efficiency, we can convert the basis to RREF once per cost calculation?
        # Or just do elimination for each target.
        
        # Optimization: If basis size is small, just do elimination.
        # N is small (<=16).
        
        # We'll create a copy of the basis vectors to use for elimination
        # and keep track of the coefficients.
        
        for t in targets:
            if t == 0:
                continue # Cost 0
            
            # We want to express t as XOR of a subset of basis vectors.
            # This is equivalent to solving A x = t where A is the matrix of basis vectors.
            # Since N is small, we can use a simple iterative elimination.
            
            # Let's maintain the basis in a form that allows quick lookup?
            # For now, simple elimination.
            
            rem = t
            coeffs = [0] * len(basis)
            
            # We need to eliminate bits.
            # We'll sort basis vectors by highest set bit?
            # Actually, we just need to find ANY representation that minimizes weight?
            # No, the "cost" in the greedy step assumes we use the *unique* representation 
            # if the basis is independent, or a *greedy* representation if not.
            # To be consistent, let's assume we use a greedy elimination:
            # Iterate through basis vectors, if the highest bit of rem matches the highest bit of b, 
            # XOR b and mark coeff.
            
            # This greedy elimination does NOT guarantee minimum weight.
            # But it is fast and consistent.
            
            # To make it better, we can sort basis vectors by weight or index.
            # Let's sort by index of highest set bit (pivot).
            
            # Pre-compute pivots for current basis?
            # This is getting complex.
            
            # Alternative: 
            # The cost function for the greedy step can be an estimate.
            # Estimate: For each target, count how many basis vectors are "useful".
            
            # Let's stick to a simpler cost:
            # Cost = Sum of Hamming weights of targets in the standard basis? No, that's constant.
            
            # Let's use the number of basis vectors needed to span the targets?
            
            # Actually, let's just use the Hamming weight of the target 
            # reduced by the "coverage" of the new gate.
            
            pass
            
        return total_cost

    # Given the complexity of exact cost calculation in a loop, 
    # and the small N, let's use a different strategy:
    # **Greedy Common Subexpression Elimination (CSE) on the Output Rows directly.**
    
    # Strategy:
    # 1. Start with each output row as a set of input indices.
    # 2. Repeatedly find two rows (or intermediates) that share a large intersection?
    #    No, for XOR, we look for $A \oplus B = C$ where C is useful.
    
    # Let's try a simple heuristic that is known to be good:
    # **Berkowitz-like**: 
    # 1. Compute RREF.
    # 2. For each pivot, if the row has weight > 1, try to find a gate that reduces the weight of multiple rows.
    
    # I will implement a **Beam Search** with a small beam width (e.g., 5) 
    # over the set of "active" gates.
    
    # State: (set_of_gates, cost)
    # But state space is huge.
    
    # Let's go back to the **Greedy Basis Extension** but with a **fast cost estimate**.
    # Cost Estimate: 
    # For a candidate gate $g = u \oplus v$, 
    # for each target $t$, check if $t \oplus g$ has a lower weight than $t$?
    # This is not quite right.
    
    # Let's use this:
    # 1. Maintain a list of "expressions" for the targets.
    # 2. Initially, expression for target i is the set of input indices.
    # 3. For each candidate gate $g$ (defined by wires $w_1, w_2$):
    #    - For each target expression $E_i$:
    #      - If $g$ can be used to simplify $E_i$, update $E_i$.
    #      - How? If $E_i$ contains $w_1$ and $w_2$, replace them with $g$? 
    #      - Or if $E_i \oplus g$ has lower weight?
    
    # This is getting too vague. I will use a **fixed-time greedy search** 
    # that tries to add gates that reduce the total Hamming weight of the targets 
    # when expressed in the current basis.
    
    # To make it fast, I will precompute the min-weight of every vector in GF(2)^N 
    # using the *current* basis? No, that's $2^N \times N$ per step.
    # For N=16, $65536 \times 16 \approx 1M$ ops. 
    # If we do this for 100 steps, that's 100M ops. Might be too slow in Python.
    
    # I will use a **heuristic pair selection**:
    # 1. For each pair of existing wires $u, v$, compute $w = u \oplus v$.
    # 2. Score the pair by: 
    #    $Score = \sum_{t \in targets} (\text{weight}(t) - \text{weight}(t \oplus w))$
    #    This measures how much $w$ "helps" to reduce the weight of $t$ if we were to XOR it.
    #    This is an upper bound on the benefit.
    # 3. Pick the pair with the highest Score.
    # 4. If Score > 1 (cost of new gate), add it.
    
    basis = [1 << i for i in range(n)]
    wire_map = {1 << i: i for i in range(n)}
    gates = []
    
    # We will run this loop for a fixed number of iterations or until no improvement
    max_iterations = 50
    current_best_score = float('inf')
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        best_pair = None
        best_score = 1.0 # Must be > 1 to be worth it (cost of gate is 1)
        
        # Generate all pairs
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                u = basis[i]
                v = basis[j]
                w = u ^ v
                
                if w in wire_map:
                    continue # Already exists
                
                # Calculate Score
                score = 0
                for t in targets:
                    # Benefit is reduction in weight if we use w
                    # This is a heuristic
                    w_t = bin(t).count('1')
                    w_tw = bin(t ^ w).count('1')
                    score += (w_t - w_tw)
                
                # Normalize? 
                # If score > 1, it's potentially good.
                if score > best_score:
                    best_score = score
                    best_pair = (i, j, w)
                    
        if best_pair is None:
            break
            
        i, j, w = best_pair
        # Add gate
        wire_u = wire_map[basis[i]]
        wire_v = wire_map[basis[j]]
        gates.append((wire_u, wire_v))
        
        new_wire_idx = n + len(gates)
        basis.append(w)
        wire_map[w] = new_wire_idx
        
    # 3. Construct Outputs
    # Now we have a set of gates. We need to express each target using the available wires.
    # Available wires: Inputs (0..n-1) and Gates (n..n+len(gates)-1).
    # We need to find the minimum weight representation of each target using these wires.
    # Since N is small, we can use a BFS or DP to find the min weight representation?
    # Or just use the greedy elimination with the final basis.
    
    # Let's use the final basis (which includes all gates) to find representations.
    # We'll perform Gaussian elimination for each target against the final basis.
    
    outputs = []
    
    # To make elimination fast, let's order the basis by pivot (highest set bit)
    # and create a RREF-like structure.
    
    # Actually, we can just use the `basis` list we built.
    # We'll do a simple greedy elimination for each target.
    
    # Sort basis by weight? Or by index?
    # Let's sort by the index of the highest set bit (pivot) to ensure we eliminate high bits first.
    def get_pivot(vec):
        if vec == 0: return -1
        return vec.bit_length() - 1
        
    sorted_basis_indices = sorted(range(len(basis)), key=lambda idx: get_pivot(basis[idx]), reverse=True)
    sorted_basis = [basis[idx] for idx in sorted_basis_indices]
    sorted_wires = [wire_map[basis[idx]] for idx in sorted_basis_indices]
    
    for t in targets:
        rem = t
        used_wires = []
        
        # Greedy elimination
        for vec, wire in zip(sorted_basis, sorted_wires):
            if rem == 0:
                break
            pivot = get_pivot(vec)
            if pivot == -1:
                continue
            if (rem >> pivot) & 1:
                rem ^= vec
                used_wires.append(wire)
        
        # If rem != 0, it means t is not in the span? 
        # But the span of inputs is the whole space, so it must be.
        
        # The output is the XOR of all used_wires.
        # If len(used_wires) == 0, output is 0 (wire n? No, constant 0 is wire n).
        # Wait, the constant zero wire is n.
        # If the output is 0, we should return wire n.
        
        if len(used_wires) == 0:
            outputs.append(n) # Constant zero
        elif len(used_wires) == 1:
            outputs.append(used_wires[0])
        else:
            # We need to create gates to combine these wires?
            # The problem asks for 'gates' and 'outputs'.
            # The 'gates' list contains the intermediate gates we already defined.
            # Do we need to add gates for the final combination?
            # Yes, if an output is a XOR of >1 wires, we need gates to produce it.
            
            # However, the 'gates' list is built during the synthesis.
            # The final combination is part of the circuit.
            # We need to append these combination gates to the 'gates' list 
            # and set the output to the new wire.
            
            # But wait, if we add these gates now, they might be reusable?
            # No, they are specific to this output.
            
            # We'll add the gates for the final combination.
            # Let's combine them one by one.
            current_wire = used_wires[0]
            for next_wire in used_wires[1:]:
                # Create a new gate
                gates.append((current_wire, next_wire))
                current_wire = n + len(gates) - 1
            
            outputs.append(current_wire)
            
    return {'gates': gates, 'outputs': outputs}

