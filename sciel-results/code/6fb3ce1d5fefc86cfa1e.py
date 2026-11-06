"""XOR circuit synthesis with gate minimization."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    A = np.array(matrix, dtype=np.uint8) % 2
    m, n = A.shape
    
    # Wire numbering:
    # 0..n-1: inputs
    # n: constant zero
    # n+1, n+2, ...: gate outputs (gate k -> wire n+1+k)
    
    # Strategy: Build a circuit that computes each output row.
    # Use a dictionary to track which wires compute which linear combinations.
    
    gates = []  # list of [wire1, wire2]
    wire_to_vec = {}  # wire index -> vector (tuple) representing what it computes
    
    # Initialize input wires and zero wire
    for i in range(n):
        vec = [0] * n
        vec[i] = 1
        wire_to_vec[i] = tuple(vec)
    
    zero_vec = tuple([0] * n)
    wire_to_vec[n] = zero_vec
    
    # We'll build gates as needed
    # For each output, we need to compute A[i]
    
    output_wires = []
    
    for i in range(m):
        target = tuple(A[i].tolist())
        
        # Check if we already have a wire computing this
        found = False
        for wire_idx, vec in wire_to_vec.items():
            if vec == target:
                output_wires.append(wire_idx)
                found = True
                break
        
        if found:
            continue
        
        # Need to build this combination
        # Find which inputs are needed
        needed = [j for j in range(n) if target[j] == 1]
        
        if len(needed) == 0:
            # Zero vector
            output_wires.append(n)
            continue
        
        if len(needed) == 1:
            # Single input
            output_wires.append(needed[0])
            continue
        
        # Build XOR chain: start with first needed input, XOR with others
        # This uses len(needed) - 1 gates
        current_wire = needed[0]
        for j in needed[1:]:
            # Create new gate: current_wire XOR j
            new_wire = n + 1 + len(gates)
            gates.append([current_wire, j])
            # Compute the new vector
            vec1 = wire_to_vec[current_wire]
            vec2 = wire_to_vec[j]
            new_vec = tuple((a ^ b) for a, b in zip(vec1, vec2))
            wire_to_vec[new_wire] = new_vec
            current_wire = new_wire
        
        output_wires.append(current_wire)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return what we have
        pass
    
    return {'gates': gates, 'outputs': output_wires}
