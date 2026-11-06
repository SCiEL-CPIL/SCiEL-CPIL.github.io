"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def synthesize(matrix, depth_limit):
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape
    
    # Wire numbering: inputs 0..n-1, zero wire = n
    # Gate k produces wire n+1+k
    
    # We'll build up wires. Start with input wires + zero wire.
    # available[wire] = the actual value (as a set of input indices or vector)
    # For efficiency, represent each wire as a frozenset of input indices (symmetric difference = XOR)
    
    # Actually, for small n (up to 16), we can use bitmasks
    # Each wire is represented as an n-bit integer (bitmask of which inputs are XORed)
    
    # Initialize: wire i for i in 0..n-1 has bitmask (1 << i)
    # wire n (zero) has bitmask 0
    
    wires = []  # list of bitmasks
    for i in range(n):
        wires.append(1 << i)
    wires.append(0)  # zero wire
    
    # Target outputs: each row of A as a bitmask
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)
    
    # Greedy: for each target, find the cheapest way to compute it from available wires
    # Cost of XORing wire a and wire b = 1 gate
    # We can also use 0-cost if target already exists
    
    # Strategy: process targets, maintaining a set of available wire bitmasks
    # For each target, find if it's already available, or find two wires whose XOR gives it
    
    # To minimize gates, we want to maximize reuse. 
    # A good heuristic: process rows in order of increasing Hamming weight (easier first)
    # Or: use a BFS/shortest path in the space of achievable bitmasks
    
    # For n <= 16, the space has 2^n states which is too large for n=16 (65536 is actually OK)
    # But let's use a simpler greedy with row reordering
    
    # Sort targets by popcount (ascending) - easier to build first
    target_indices = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    
    gates = []  # list of [wire_a, wire_b]
    available = set(wires)  # set of bitmasks we can access
    # Map from bitmask to wire number
    wire_map = {w: i for i, w in enumerate(wires)}
    
    # For each target, try to find cheapest combination
    for idx in target_indices:
        target = targets[idx]
        
        # Check if already available
        if target in wire_map:
            continue
        
        # Try: is target = a XOR b for some available a, b?
        found = False
        # Get list of available wires
        avail_list = list(available)
        
        # For each pair, check if XOR gives target
        # This is O(k^2) where k is number of available wires
        # For better performance, we can be smarter:
        # For each a in available, check if (a XOR target) is in available
        for a in avail_list:
            b_needed = a ^ target
            if b_needed in wire_map:
                # Use wires a and b_needed
                wa = wire_map[a]
                wb = wire_map[b_needed]
                new_wire = n + 1 + len(gates)
                gates.append([wa, wb])
                wires.append(target)
                wire_map[target] = new_wire
                available.add(target)
                found = True
                break
        
        if found:
            continue
        
        # If not found in one step, we may need multi-step
        # For now, build up from scratch using inputs
        # Decompose target into individual bits
        # This is suboptimal but correct
        
        # Actually, let's try a BFS from available wires to see if we can reach target in <=2 steps
        # Step 0: available
        # Step 1: all pairs XOR
        # Step 2: all pairs from step 1 with step 0 or step 1
        
        # For simplicity and correctness, let's just build from inputs if needed
        # Build target from scratch: XOR together the input bits that are set
        
        bits = []
        for j in range(n):
            if target & (1 << j):
                bits.append(j)
        
        if not bits:
            # Zero, already available
            continue
        
        if len(bits) == 1:
            # Single input, already available
            continue
        
        # XOR them together one by one
        # Start with first bit
        current_wire = wire_map[1 << bits[0]]
        
        for j in bits[1:]:
            input_wire = wire_map[1 << j]
            new_wire = n + 1 + len(gates)
            gates.append([current_wire, input_wire])
            wires.append(wires[current_wire] ^ (1 << j))
            wire_map[wires[current_wire] ^ (1 << j)] = new_wire
            available.add(wires[current_wire] ^ (1 << j))
            current_wire = new_wire
        
        # Now current_wire should have the target
        # But we may have added intermediate wires. That's fine.
    
    # Now map each original target to its wire
    outputs = []
    for i in range(m):
        target = targets[i]
        if target in wire_map:
            outputs.append(wire_map[target])
        else:
            # Should not happen, but fallback
            outputs.append(n)  # zero wire
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
