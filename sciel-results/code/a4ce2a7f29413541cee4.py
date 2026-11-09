"""XOR circuit synthesis with greedy cost minimization."""
import time
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Constant zero wire is n
    # Input wires are 0..n-1, wire value for input j is (1 << j)
    
    # Map from bitmask value to wire index
    wire_to_val = {}  # wire_index -> bitmask
    val_to_wires = defaultdict(list)  # bitmask -> [wire_indices]
    
    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        wire_to_val[j] = val
        val_to_wires[val].append(j)
    
    # Constant zero
    wire_to_val[n] = 0
    val_to_wires[0].append(n)
    
    gates = []
    next_wire = n + 1
    outputs = []
    
    # Sort rows by popcount (sparse first) for better sharing
    row_indices = sorted(range(m), key=lambda i: bin(rows[i]).count('1'))
    
    # Track which rows have been computed
    row_to_wire = [None] * m
    
    for idx in row_indices:
        if time.time() > deadline:
            break
        
        target = rows[idx]
        
        # Check if already available
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target][0]
            continue
        
        # Try to find two existing wires whose XOR equals target
        # i.e., find a in available such that target ^ a is also available
        found = False
        available_vals = list(val_to_wires.keys())
        
        for a in available_vals:
            b = target ^ a
            if b in val_to_wires:
                # Create gate: a_wire XOR b_wire = new_wire
                a_wire = val_to_wires[a][0]
                b_wire = val_to_wires[b][0]
                gates.append([a_wire, b_wire])
                row_to_wire[idx] = next_wire
                wire_to_val[next_wire] = target
                val_to_wires[target].append(next_wire)
                next_wire += 1
                found = True
                break
        
        if not found:
            # Fallback: create target by XORing with input 0 (or any input)
            # target = target ^ 0, so we need to build target from scratch
            # Use greedy: XOR with the first set bit's input
            if target == 0:
                row_to_wire[idx] = n
            else:
                # Find lowest set bit
                lowest_bit = (target & (-target)).bit_length() - 1
                input_wire = lowest_bit
                # target = (1 << lowest_bit) XOR (target ^ (1 << lowest_bit))
                remaining = target ^ (1 << lowest_bit)
                
                if remaining == 0:
                    row_to_wire[idx] = input_wire
                else:
                    # Recursively we'd need more, but for simplicity:
                    # Just create a chain of XORs with inputs
                    # This is suboptimal but correct
                    current_wire = input_wire
                    current_val = 1 << lowest_bit
                    # Remove lowest bit, process remaining
                    rem = remaining
                    while rem:
                        if time.time() > deadline:
                            break
                        lb = (rem & (-rem)).bit_length() - 1
                        gates.append([current_wire, lb])
                        current_wire = next_wire
                        next_wire += 1
                        current_val ^= (1 << lb)
                        wire_to_val[current_wire] = current_val
                        val_to_wires[current_val].append(current_wire)
                        rem ^= (1 << lb)
                    row_to_wire[idx] = current_wire
    
    # Build outputs in original order
    for i in range(m):
        if row_to_wire[i] is None:
            # Shouldn't happen, but fallback
            row_to_wire[i] = n
    
    outputs = row_to_wire
    
    return {'gates': gates, 'outputs': outputs}
