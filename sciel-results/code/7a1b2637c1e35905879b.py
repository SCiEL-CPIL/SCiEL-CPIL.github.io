"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using Gaussian Elimination to find a minimal basis."""
    start_time = time.time()
    DEADLINE = 85.0
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    # Convert rows to integer masks
    output_masks = []
    for i in range(n_outputs):
        m = 0
        for j in range(n_inputs):
            if mat[i][j]:
                m |= (1 << j)
        output_masks.append(m)
        
    # Gaussian Elimination to find basis
    # We will maintain a list of basis rows (masks) and their "coefficients" 
    # which represent which original output rows they are formed from.
    # However, we actually need to express the ORIGINAL outputs in terms of the basis.
    # So we perform GE on the matrix of output masks.
    
    # Let's maintain the state of the rows and a transformation matrix.
    # current_rows[i] is the mask of the i-th row in the current elimination state.
    # transform[i] is a bitmask of size n_outputs, indicating which original rows were XORed to get current_rows[i].
    
    current_rows = list(output_masks)
    # transform[i] has bit k set if original row k is part of the XOR sum for current_rows[i]
    transform = [1 << i for i in range(n_outputs)]
    
    pivot_cols = []
    basis_indices = []
    
    for col in range(n_inputs):
        # Find a row with 1 in this column and 0 in all previous pivot columns
        pivot_row = -1
        for i in range(n_outputs):
            if current_rows[i] == 0:
                continue
            # Check if this row has 0 in all previous pivot columns
            is_pivot_candidate = True
            for pc in pivot_cols:
                if (current_rows[i] >> pc) & 1:
                    is_pivot_candidate = False
                    break
            if is_pivot_candidate and ((current_rows[i] >> col) & 1):
                pivot_row = i
                break
        
        if pivot_row == -1:
            continue
            
        # Swap pivot_row to the top of the "active" set? 
        # We don't strictly need to swap, but we need to record that this row is a basis vector.
        # For efficiency, we can just process it.
        pivot_cols.append(col)
        basis_indices.append(pivot_row)
        
        # Eliminate this column from all other rows
        for i in range(n_outputs):
            if i == pivot_row:
                continue
            if (current_rows[i] >> col) & 1:
                current_rows[i] ^= current_rows[pivot_row]
                transform[i] ^= transform[pivot_row]
                
    # Now, current_rows[i] for i in basis_indices are the basis vectors.
    # transform[i] for i in basis_indices tells us how these basis vectors relate to original rows.
    # But we need to express original rows in terms of basis vectors.
    # The matrix formed by transform[i] for basis rows is invertible (it's the transformation to RREF).
    # Actually, it's easier to just solve for each original output mask using the basis masks.
    
    basis_masks = [current_rows[i] for i in basis_indices]
    basis_wires = []
    next_wire = n_inputs + 1
    gates = []
    
    def popcount(m):
        return bin(m).count('1')
        
    def bits_from_mask(m):
        bits = []
        j = 0
        while m:
            if m & 1:
                bits.append(j)
            m >>= 1
            j += 1
        return bits

    # Create wires for basis vectors
    # To minimize depth and gate count, we compute each basis vector from inputs.
    # Since n_inputs is small (<=16), a simple XOR tree is fine.
    # We can optimize by reusing sub-expressions, but for now, direct computation is safe.
    # Actually, we can share sub-expressions between basis vectors if we use a global wire map.
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    def create_wire_for_mask(mask):
        nonlocal next_wire
        if mask in wire_map:
            return wire_map[mask]
        
        # Find a bit to split on
        bit = None
        for i in range(n_inputs):
            if mask & (1 << i):
                bit = 1 << i
                break
        if bit is None:
            return n_inputs # Should not happen for non-zero mask
            
        rest = mask ^ bit
        w1 = wire_map[bit]
        w2 = create_wire_for_mask(rest)
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        wire_map[mask] = w
        return w

    for b_mask in basis_masks:
        if b_mask != 0:
            w = create_wire_for_mask(b_mask)
            basis_wires.append(w)
        else:
            basis_wires.append(n_inputs)
            
    # Now express each original output in terms of basis wires
    # We need to find coefficients c such that output_mask = XOR(c_i * basis_masks[i])
    # This is solving a system of linear equations.
    # Since basis_masks are linearly independent, we can do this greedily.
    
    output_wires = []
    for k in range(n_outputs):
        target_mask = output_masks[k]
        if target_mask == 0:
            output_wires.append(n_inputs)
            continue
            
        # Find combination of basis vectors
        # We'll use a simple backtracking or greedy method since rank is small.
        # Greedy: iterate basis vectors, if target_mask has the highest bit of basis_vec, XOR it.
        # This works if basis vectors are in RREF (pivot bits are distinct and highest).
        # Our GE produced RREF-like structure.
        
        # Let's verify RREF property: 
        # For each basis vector, its pivot column has 0 in all other basis vectors.
        # So we can just check the pivot bit.
        
        # However, we didn't store the pivot columns for each basis vector explicitly in an easy way.
        # Let's re-derive the pivot for each basis mask.
        # Actually, we can just solve the system.
        
        # Given small size, we can just try all combinations? No, 2^rank.
        # Rank <= 16. 2^16 = 65536. This is fast.
        
        # But we can do better.
        # Let's use the fact that basis_masks are linearly independent.
        # We can build a lookup table for the basis?
        # Or just use Gaussian elimination on the system B * x = target.
        
        # Let's use a simple method:
        # Iterate through basis vectors. If the target mask contains the "signature" bit of the basis vector, include it.
        # What is the signature bit? The highest set bit?
        # In RREF, the pivot bit is the highest bit? No, GE processes columns 0 to n-1.
        # So pivot bits are increasing.
        # Let's just use a generic solver for the small system.
        
        # System:
        # b_0 * x_0 + b_1 * x_1 + ... + b_{r-1} * x_{r-1} = target
        # This is vector addition over GF(2).
        
        # We can just use the standard algorithm:
        # For each basis vector, if target has the pivot bit set, XOR that basis vector into target.
        # We need to know the pivot bit for each basis vector.
        
        # Let's re-run GE just to get pivot bits? Or store them?
        # We stored pivot_cols in order.
        # basis_indices[i] corresponds to pivot_cols[i].
        
        # Let's map basis_index -> pivot_col
        basis_pivot_map = {}
        for idx, b_idx in enumerate(basis_indices):
            basis_pivot_map[b_idx] = pivot_cols[idx]
            
        # Now, for each output, we can determine the coefficients.
        # We need to know which basis vector corresponds to which pivot.
        # Let's create a list of (pivot_col, basis_wire)
        pivot_to_wire = []
        for idx, b_idx in enumerate(basis_indices):
            pc = pivot_cols[idx]
            w = basis_wires[idx]
            pivot_to_wire.append((pc, w))
            
        # To be efficient, we can just check all basis vectors.
        # Since r is small, O(r * n_outputs) is fine.
        
        # Actually, we can just use the fact that the basis vectors form a basis.
        # We can solve for each output by "decoding".
        # Let's just use a simple loop.
        
        # Wait, the basis vectors might not be in "triangular" order relative to bits.
        # But they are linearly independent.
        # Let's just use a generic linear solver for the small dimension.
        # Or, simpler:
        # We know that the set of basis masks spans the space.
        # We can pre-compute the inverse of the basis matrix?
        
        # Let's stick to the simplest robust method:
        # For each output, find the set of basis vectors that XOR to it.
        # We can do this by iterating through basis vectors and checking if XORing them reduces the "distance" to target?
        # No, that's heuristic.
        
        # Correct method:
        # We have basis vectors B_0...B_{r-1}.
        # We want x such that XOR(x_i B_i) = T.
        # This is equivalent to solving B^T x = T (where B is matrix of basis vectors as columns? No, rows).
        # Let's just use the pivot property.
        # If we processed columns 0..n-1, the pivot for basis vector i is the first column where it has a 1 and all previous basis vectors have 0.
        # This implies that if we sort basis vectors by their pivot column, we can decode greedily from highest pivot to lowest.
        
        # Let's sort basis wires by pivot column descending.
        sorted_basis = sorted(pivot_to_wire, key=lambda x: x[0], reverse=True)
        
        remaining_mask = target_mask
        selected_wires = []
        
        for pc, w in sorted_basis:
            if (remaining_mask >> pc) & 1:
                # We need to XOR this basis vector
                # But we don't have the mask of the basis vector here?
                # We have the wire. We need the mask to update remaining_mask.
                # We have basis_masks[idx].
                # Let's restructure to keep masks and wires together.
                
                pass # Logic needs adjustment
                
    # Let's restart the output assembly part with correct data structures.
    
    # Re-initialize output assembly
    output_wires = []
    
    # Create a list of (mask, wire, pivot_col)
    basis_info = []
    for idx, b_idx in enumerate(basis_indices):
        m = basis_masks[idx]
        w = basis_wires[idx]
        pc = pivot_cols[idx]
        basis_info.append((m, w, pc))
        
    # Sort by pivot column descending to allow greedy decoding
    basis_info.sort(key=lambda x: x[2], reverse=True)
    
    for k in range(n_outputs):
        target_mask = output_masks[k]
        if target_mask == 0:
            output_wires.append(n_inputs)
            continue
            
        remaining_mask = target_mask
        selected_wires = []
        
        for m, w, pc in basis_info:
            if (remaining_mask >> pc) & 1:
                # XOR this basis vector
                remaining_mask ^= m
                selected_wires.append(w)
                
        # Now remaining_mask should be 0
        # Combine selected_wires
        if not selected_wires:
            output_wires.append(n_inputs)
        else:
            # XOR all selected wires
            final_wire = selected_wires[0]
            for w in selected_wires[1:]:
                gates.append([final_wire, w])
                final_wire = next_wire
                next_wire += 1
            output_wires.append(final_wire)
            
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
