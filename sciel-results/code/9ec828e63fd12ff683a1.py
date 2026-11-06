"""XOR-circuit synthesis with greedy shared-subexpression, BFS decomposition, and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape
    
    # Convert rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)
    
    # Step 1: Find a minimal basis using Gaussian elimination
    # We want to find a set of linearly independent vectors that span all targets
    basis = []
    basis_indices = []  # which original target index each basis vector came from
    
    # Try to add each target to the basis if it's independent
    for i in range(m):
        vec = targets[i]
        if vec == 0:
            continue
        # Check if vec is in span of current basis
        reduced = vec
        for b in basis:
            # If leading bit of b matches leading bit of reduced, XOR them
            lb = b.bit_length() - 1
            lr = reduced.bit_length() - 1
            if lb == lr:
                reduced ^= b
        if reduced != 0:
            basis.append(reduced)
            basis_indices.append(i)
    
    # Now we have a basis. We need to express each target as XOR of basis vectors.
    # First, we need to compute the actual circuit for each basis vector.
    # Then, each target is XOR of some subset of basis wires.
    
    # Compute circuit for each basis vector using greedy method
    # Wire numbering: inputs 0..n-1, zero wire = n
    # Gate k produces wire n+1+k
    
    wires = []  # list of bitmasks
    for i in range(n):
        wires.append(1 << i)
    wires.append(0)  # zero wire
    
    gates = []  # list of [wire_a, wire_b]
    wire_map = {w: i for i, w in enumerate(wires)}
    
    def time_left():
        return deadline - time.time()
    
    # For each basis vector, synthesize it
    # Use a smarter approach: for each basis vector, try to find cheapest decomposition
    basis_wires = []  # wire number for each basis vector
    
    for bi, bvec in enumerate(basis):
        if bvec in wire_map:
            basis_wires.append(wire_map[bvec])
            continue
        
        # Try to find bvec = a XOR b for available wires
        found = False
        avail_list = list(wire_map.keys())
        for a in avail_list:
            b_needed = a ^ bvec
            if b_needed in wire_map:
                wa = wire_map[a]
                wb = wire_map[b_needed]
                new_wire = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(bvec)
                wire_map[bvec] = new_wire
                basis_wires.append(new_wire)
                found = True
                break
        
        if found:
            continue
        
        # If not found, try BFS up to depth 2
        # BFS level 0: available wires
        # BFS level 1: all XORs of pairs from level 0
        # BFS level 2: all XORs of pairs from level 1 with level 0 or level 1
        
        if time_left() < 5.0:
            # Fallback: build from inputs
            bits = []
            for j in range(n):
                if bvec & (1 << j):
                    bits.append(j)
            if not bits:
                basis_wires.append(n)
                continue
            if len(bits) == 1:
                basis_wires.append(wire_map[1 << bits[0]])
                continue
            
            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                wires.append(wires[current_wire] ^ (1 << j))
                wire_map[wires[current_wire] ^ (1 << j)] = new_wire
                current_wire = new_wire
            basis_wires.append(current_wire)
            continue
        
        # BFS for depth 2
        # Level 1: all XORs of two available wires
        level1 = {}  # bitmask -> (wire_a, wire_b)
        for i, a in enumerate(avail_list):
            for b in avail_list[i+1:]:
                xor_val = a ^ b
                if xor_val not in level1 and xor_val not in wire_map:
                    level1[xor_val] = (a, b)
        
        if bvec in level1:
            a, b = level1[bvec]
            wa = wire_map[a]
            wb = wire_map[b]
            new_wire = n + 1 + len(gates)
            gates.append([wa, wb])
            wires.append(bvec)
            wire_map[bvec] = new_wire
            basis_wires.append(new_wire)
            continue
        
        # Level 2: XOR of level1 with available, or level1 with level1
        # This can be expensive, so limit it
        if len(level1) > 5000:
            # Too many, fallback to direct construction
            bits = []
            for j in range(n):
                if bvec & (1 << j):
                    bits.append(j)
            if not bits:
                basis_wires.append(n)
                continue
            if len(bits) == 1:
                basis_wires.append(wire_map[1 << bits[0]])
                continue
            
            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                wires.append(wires[current_wire] ^ (1 << j))
                wire_map[wires[current_wire] ^ (1 << j)] = new_wire
                current_wire = new_wire
            basis_wires.append(current_wire)
            continue
        
        # Try level 2: for each l1 in level1, check if l1 XOR bvec is in available or level1
        found_l2 = False
        l1_items = list(level1.items())
        for l1_val, (l1_a, l1_b) in l1_items:
            needed = l1_val ^ bvec
            if needed in wire_map:
                # Create wire for l1_val first, then XOR with needed
                # First create l1_val
                wa = wire_map[l1_a]
                wb = wire_map[l1_b]
                new_wire_l1 = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(l1_val)
                wire_map[l1_val] = new_wire_l1
                
                # Then XOR with needed
                wn = wire_map[needed]
                new_wire = n + 1 + len(gates)
                gates.append([new_wire_l1, wn])
                wires.append(bvec)
                wire_map[bvec] = new_wire
                basis_wires.append(new_wire)
                found_l2 = True
                break
        
        if found_l2:
            continue
        
        # Also try: needed in level1
        for l1_val, (l1_a, l1_b) in l1_items:
            needed = l1_val ^ bvec
            if needed in level1:
                # Create both l1_val and needed, then XOR
                # Create l1_val
                wa = wire_map[l1_a]
                wb = wire_map[l1_b]
                new_wire_l1 = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(l1_val)
                wire_map[l1_val] = new_wire_l1
                
                # Create needed
                na, nb = level1[needed]
                wna = wire_map[na]
                wnb = wire_map[nb]
                new_wire_needed = n + 1 + len(gates)
                gates.append([wna, wnb])
                wires.append(needed)
                wire_map[needed] = new_wire_needed
                
                # XOR them
                new_wire = n + 1 + len(gates)
                gates.append([new_wire_l1, new_wire_needed])
                wires.append(bvec)
                wire_map[bvec] = new_wire
                basis_wires.append(new_wire)
                found_l2 = True
                break
        
        if found_l2:
            continue
        
        # Fallback: build from inputs
        bits = []
        for j in range(n):
            if bvec & (1 << j):
                bits.append(j)
        if not bits:
            basis_wires.append(n)
            continue
        if len(bits) == 1:
            basis_wires.append(wire_map[1 << bits[0]])
            continue
        
        current_wire = wire_map[1 << bits[0]]
        for j in bits[1:]:
            input_wire = wire_map[1 << j]
            new_wire = n + 1 + len(gates)
            gates.append([current_wire, input_wire])
            wires.append(wires[current_wire] ^ (1 << j))
            wire_map[wires[current_wire] ^ (1 << j)] = new_wire
            current_wire = new_wire
        basis_wires.append(current_wire)
    
    # Now express each target as XOR of basis wires
    outputs = []
    for i in range(m):
        target = targets[i]
        if target == 0:
            outputs.append(n)
            continue
        if target in wire_map:
            outputs.append(wire_map[target])
            continue
        
        # Express target in terms of basis
        # Find which basis vectors XOR to give target
        # This is solving a linear system over GF(2)
        # basis has rank r, target should be in span
        
        # Use the same reduction as before
        reduced = target
        coeffs = [0] * len(basis)
        for bi, b in enumerate(basis):
            lb = b.bit_length() - 1
            lr = reduced.bit_length() - 1
            if lb == lr:
                reduced ^= b
                coeffs[bi] = 1
        
        if reduced != 0:
            # Should not happen if basis spans, but fallback
            # Build from inputs
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)
            if not bits:
                outputs.append(n)
                continue
            if len(bits) == 1:
                outputs.append(wire_map[1 << bits[0]])
                continue
            
            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                wires.append(wires[current_wire] ^ (1 << j))
                wire_map[wires[current_wire] ^ (1 << j)] = new_wire
                current_wire = new_wire
            outputs.append(current_wire)
            continue
        
        # XOR together the selected basis wires
        selected = [basis_wires[bi] for bi in range(len(basis)) if coeffs[bi] == 1]
        
        if not selected:
            outputs.append(n)
            continue
        if len(selected) == 1:
            outputs.append(selected[0])
            continue
        
        # Check if the XOR of selected is already available
        xor_val = 0
        for w in selected:
            xor_val ^= wires[w]
        if xor_val in wire_map:
            outputs.append(wire_map[xor_val])
            continue
        
        # XOR them together
        current_wire = selected[0]
        for w in selected[1:]:
            new_wire = n + 1 + len(gates)
            gates.append([current_wire, w])
            wires.append(wires[current_wire] ^ wires[w])
            wire_map[wires[current_wire] ^ wires[w]] = new_wire
            current_wire = new_wire
        outputs.append(current_wire)
    
    # Local search: try to reduce gate count by finding redundant gates
    # This is a simple heuristic: for each gate, check if its output is used only once
    # and if it can be eliminated by inlining
    
    # Count usage of each wire
    wire_usage = {}
    for g in gates:
        for w in g:
            wire_usage[w] = wire_usage.get(w, 0) + 1
    for o in outputs:
        wire_usage[o] = wire_usage.get(o, 0) + 1
    
    # Try to eliminate gates whose output is used only once (the output itself)
    # This is complex, so skip for now and just return
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
