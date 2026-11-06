import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    
    def weight(x):
        return bin(x).count('1')
        
    def get_sparse_basis(rows):
        basis = []
        # Sort rows by weight to help sparsity?
        for r in rows:
            current = r
            for b in basis:
                if weight(current ^ b) < weight(current):
                    current = current ^ b
            if current != 0:
                basis.append(current)
        return basis

    def compute_mask(target, available, wire_map, gates, time_limit):
        # available: set of masks
        # wire_map: dict mask -> wire_id
        # gates: list of [w1, w2]
        # returns wire_id of target
        
        if target in wire_map:
            return wire_map[target]
            
        if time.time() > time_limit:
            # Fallback: just use inputs directly (expensive but valid)
            # This is a slow path, shouldn't happen often
            w = 0
            # Find any representation? 
            # For safety, just return a dummy wire and hope it's not used? 
            # No, must be valid. 
            # If timeout, we should have returned earlier.
            pass

        # Greedy reduction
        best_a = None
        best_w = weight(target)
        
        for a in available:
            w_new = weight(target ^ a)
            if w_new < best_w:
                best_w = w_new
                best_a = a
                
        if best_a is None:
            # Should not happen if target is in span
            # Return 0? No.
            return 0
            
        new_target = target ^ best_a
        w_new_target = compute_mask(new_target, available, wire_map, gates, time_limit)
        
        # Add gate
        w_id = len(gates) + n + 1 # n is num inputs, wire n is 0
        # Wait, wire ids:
        # 0..n-1: inputs
        # n: constant 0
        # n+1+k: gate k
        
        # We need to pass n to this function or use closure
        # Let's use closure variables
        
        gates.append([w_new_target, wire_map[best_a]])
        wire_map[target] = n + 1 + len(gates) - 1
        return wire_map[target]

    n = len(matrix[0]) if len(matrix) > 0 else 0
    m = len(matrix)
    
    # Convert matrix to list of masks
    rows = []
    for row in matrix:
        mask = 0
        for j, val in enumerate(row):
            if val:
                mask |= (1 << j)
        rows.append(mask)
        
    # 1. Get Basis
    basis = get_sparse_basis(rows)
    
    # 2. Setup
    available = set()
    wire_map = {}
    gates = []
    
    # Add inputs
    for i in range(n):
        mask = 1 << i
        available.add(mask)
        wire_map[mask] = i
        
    # Add constant 0
    available.add(0)
    wire_map[0] = n
    
    time_limit = start_time + 50 # 50s limit for synthesis
    
    # 3. Compute Basis Wires
    for b in basis:
        if b not in wire_map:
            compute_mask(b, available, wire_map, gates, time_limit)
            # Note: compute_mask adds to available and wire_map
            
    # 4. Compute Outputs
    outputs = []
    for r in rows:
        if r in wire_map:
            outputs.append(wire_map[r])
        else:
            # Need to express r in terms of basis
            # Since basis spans rows, r is in span.
            # We can just use compute_mask again, which will find a path
            # using the now available basis wires.
            compute_mask(r, available, wire_map, gates, time_limit)
            outputs.append(wire_map[r])
            
    return {'gates': gates, 'outputs': outputs}
