"""XOR circuit synthesis with basis-aware greedy cost minimization and recursive fallback."""
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Identify basis rows using Gaussian Elimination
    # We want to process basis rows first to establish a strong set of available wires
    basis_indices = []
    non_basis_indices = []
    
    # Simple Gaussian elimination to find pivot rows
    # We work on the rows themselves
    processed_rows = []
    # Store (original_row_value, pivot_bit)
    
    for i in range(m):
        r = rows[i]
        if r == 0:
            non_basis_indices.append(i)
            continue
            
        # Check if r is independent of processed_rows
        # Reduce r by processed_rows
        reduced_r = r
        for pr, pivot_bit in processed_rows:
            if (reduced_r >> pivot_bit) & 1:
                reduced_r ^= pr
        
        if reduced_r != 0:
            # Find pivot (lowest set bit)
            pivot_bit = (reduced_r & (-reduced_r)).bit_length() - 1
            # Store the original row r, but we identified pivot based on reduced_r
            # Actually, for dependency checking, we just need the reduced form to be non-zero.
            # We add the original r to the list of available basis vectors for future reduction?
            # No, standard GE adds the reduced vector. But here we are just ordering.
            # The key is that 'r' is independent of previous 'r's.
            processed_rows.append((r, pivot_bit))
            basis_indices.append(i)
        else:
            non_basis_indices.append(i)
            
    # Sort basis indices by popcount (sparse first) to build simple wires early
    basis_indices.sort(key=lambda i: bin(rows[i]).count('1'))
    # Sort non-basis indices by popcount
    non_basis_indices.sort(key=lambda i: bin(rows[i]).count('1'))
    
    # Combined processing order: Basis first, then non-basis
    processing_order = basis_indices + non_basis_indices
    
    # Map from bitmask value to wire index
    val_to_wires = {}
    
    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        val_to_wires[val] = j
    
    # Constant zero
    val_to_wires[0] = n
    
    gates = []
    next_wire = n + 1
    
    # Track which rows have been computed
    row_to_wire = [None] * m
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_greedy(target):
        """
        Recursively build 'target' using existing wires.
        Returns the wire index for 'target'.
        """
        nonlocal next_wire, gates, val_to_wires
        
        # If already available, return it
        if target in val_to_wires:
            return val_to_wires[target]
            
        if target == 0:
            return n
            
        # Find 'a' in val_to_wires that minimizes popcount(target ^ a)
        # This is the greedy step: pick the helper that makes the remaining part simplest.
        best_a = None
        best_a_wire = None
        min_pc = float('inf')
        
        # Iterate over all available wires
        # Optimization: If we have many wires, this is O(N). 
        # For N<=16 (16x16 case), N is small.
        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            pc = popcount(diff)
            if pc < min_pc:
                min_pc = pc
                best_a = a
                best_a_wire = a_wire
                if pc == 1: # Can't do better than 1 (which would mean diff is an input, handled by recursion base case or val_to_wires check)
                    # Actually if pc==1, diff is an input, which is in val_to_wires.
                    # So the next recursive call will just return the input wire.
                    # We can break early? No, we want the 'a' that leads to the simplest diff.
                    # If pc==1, diff is a single bit, so it's an input. 
                    # Any 'a' that results in a single bit diff is equally good in terms of depth (1 more gate).
                    # But we might prefer 'a' that is already a "strong" wire?
                    # For now, first found min is fine.
                    pass
        
        if best_a is None:
            # Should not happen if n>0, but fallback to input 0
            best_a = 0
            best_a_wire = n
            
        diff = target ^ best_a
        
        # Recursively build 'diff'
        diff_wire = build_greedy(diff)
        
        # If diff is 0, then target == best_a, so we just return best_a_wire
        if diff == 0:
             return best_a_wire
             
        # Create gate: best_a_wire XOR diff_wire
        # Check if this combination was already created? 
        # val_to_wires[target] would have been set if it existed.
        # So we create a new gate.
        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire
        
        return new_wire

    for idx in processing_order:
        if time.time() > deadline:
            break
        
        target = rows[idx]
        
        # Check if already available
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            continue
        
        # Try to find two existing wires whose XOR equals target
        # This is the "1-gate" optimization. If found, it's cheaper than the recursive fallback (which takes >= 1 gate, usually more).
        found = False
        
        # Optimization: Iterate over a list of keys to avoid mutation issues if we were to add inside, 
        # but we don't add inside this loop.
        # We can also prioritize 'a' values that are "close" to target? 
        # Actually, any 1-gate solution is optimal for this row. 
        # So we just need to find *any* pair.
        
        # To speed up, we can iterate over the smaller set of keys? 
        # val_to_wires grows. 
        # We can check if target ^ a is in val_to_wires.
        
        # Heuristic: Check inputs first? They are the most "fundamental".
        # Then check other wires.
        
        # Simple iteration
        # To avoid O(N^2) blowup if N is large, we rely on N being small (<=16 inputs + generated).
        # Max wires ~ 16 + 16*16 = 272. 272^2 = 73k checks per row. 16 rows -> 1M checks. Fast.
        
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                # Avoid using the same wire twice (unless it's zero, but target!=0 here)
                if a_wire == b_wire and target != 0:
                    continue
                # Create gate: a_wire XOR b_wire = new_wire
                gates.append([a_wire, b_wire])
                row_to_wire[idx] = next_wire
                val_to_wires[target] = next_wire
                next_wire += 1
                found = True
                break
        
        if not found:
            # Use the greedy recursive builder
            # This will add intermediate wires to val_to_wires as well, aiding future rows
            row_to_wire[idx] = build_greedy(target)

    # Build outputs in original order
    for i in range(m):
        if row_to_wire[i] is None:
            # Shouldn't happen, but fallback
            row_to_wire[i] = n
            
    outputs = row_to_wire
    
    return {'gates': gates, 'outputs': outputs}
