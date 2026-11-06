"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def synthesize(matrix, depth_limit):
    try:
        A = np.array(matrix, dtype=int)
    except Exception:
        A = np.array(matrix)
        A = (A != 0).astype(int)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    m, n = A.shape
    
    # Convert rows to bitmasks (integers)
    # n <= 16, so int is sufficient
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i][j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # 1. Find a basis for the row space using Gaussian Elimination
    # basis: list of masks
    # We also want to track the "cost" or structure if possible, 
    # but for now, we just find the basis vectors.
    # To minimize gate count, we prefer basis vectors with lower weight.
    # Standard Gaussian elimination (left-to-right) is a good heuristic.
    
    basis = []
    # We'll process rows and add them to basis if they are independent.
    # We can sort rows by weight first to potentially get a "sparser" basis?
    # Or just process in order.
    
    # Let's try to find a basis that minimizes total weight.
    # Greedy: pick the row with min weight, add to basis, eliminate from others?
    # This is similar to finding a minimum weight basis, which is hard.
    # Let's stick to standard Gaussian elimination but on sorted rows (by weight).
    
    sorted_row_indices = sorted(range(m), key=lambda i: bin(row_masks[i]).count('1'))
    
    for idx in sorted_row_indices:
        r = row_masks[idx]
        if r == 0:
            continue
            
        # Check if r is in the span of existing basis
        # Perform elimination
        current = r
        for b in basis:
            # Find highest set bit in current
            if current == 0:
                break
            # Find highest set bit in b
            # We want to eliminate the highest bit of current using b
            # This requires b to have that bit set.
            # Standard GE: for each bit from high to low, if current has it, XOR with basis vector that has it as pivot.
            # Our basis is not necessarily in pivot form yet.
            # Let's maintain basis in row echelon form.
            pass
            
    # Let's do proper Row Echelon Form construction
    # basis[i] will have pivot at bit i (or some unique bit)
    # We'll use a dictionary: pivot_bit -> basis_vector
    basis_by_pivot = {}
    
    # Sort rows by weight to potentially find simpler basis vectors first?
    # Actually, for GE, we usually go by column (bit position).
    # Let's iterate through bit positions from high to low.
    
    # We need to keep track of the remaining rows to eliminate.
    remaining = row_masks[:]
    
    # We will build a list of basis vectors.
    # Each basis vector is a mask.
    # We also need to express each original row in terms of the basis.
    
    # Let's do standard GE on the matrix A (conceptually).
    # We will produce a list of basis masks.
    # And for each original row, its coordinates in the basis.
    
    # To do this efficiently, we can augment the matrix with an identity matrix (coordinates).
    # But n is small.
    
    # Augmented matrix: [A | I]
    # We perform GE on the left part.
    # The right part will give the coordinates of the basis vectors in terms of the original rows?
    # No, we want coordinates of original rows in terms of the basis.
    # If we compute RREF of A, the non-zero rows are the basis.
    # The original rows are linear combinations of the RREF rows.
    
    # Let's use numpy for the linear algebra part to find the RREF and coordinates.
    # Since we need exact integer arithmetic, we can do it manually or use numpy int.
    
    # Create augmented matrix: m x (n + m)
    # Left: A (m x n)
    # Right: I (m x m)
    
    aug = np.zeros((m, n + m), dtype=int)
    aug[:, :n] = A
    aug[:, n:] = np.eye(m, dtype=int)
    
    # Perform Gaussian Elimination to get RREF
    # We want to find the pivot rows.
    # We'll iterate through columns 0 to n-1.
    # Find a row with 1 in this column among the remaining rows.
    # Swap to top.
    # Eliminate other rows.
    
    pivot_rows = []
    current_row = 0
    
    for col in range(n):
        # Find pivot
        pivot = -1
        for r in range(current_row, m):
            if aug[r, col] == 1:
                pivot = r
                break
        
        if pivot == -1:
            continue
            
        # Swap current_row and pivot
        aug[[current_row, pivot]] = aug[[pivot, current_row]]
        
        # Eliminate other rows
        for r in range(m):
            if r != current_row and aug[r, col] == 1:
                aug[r] ^= aug[current_row]
        
        pivot_rows.append(current_row)
        current_row += 1
        
    # Now, the rows in pivot_rows are the basis vectors (in RREF form).
    # The right part of these rows gives the coordinates of the basis vectors in terms of the *original* rows?
    # Wait. The operation `aug[r] ^= aug[current_row]` updates both sides.
    # The left side is the RREF of A.
    # The right side is the transformation matrix T such that T * A = RREF(A).
    # So, RREF(A) = T * A.
    # This means each basis vector (row of RREF) is a linear combination of the original rows.
    # We want the reverse: each original row is a linear combination of the basis vectors.
    # If R = T A, then A = T^{-1} R.
    # So we need to invert T? Or just perform the elimination in a way that tracks the inverse.
    
    # Alternative:
    # We have the basis vectors (masks) from the left part of pivot_rows.
    # We need to express each original row (mask) as a sum of these basis vectors.
    # We can do this by solving a system of linear equations for each original row,
    # or by performing the elimination on the original rows using the basis.
    
    # Let's extract the basis masks.
    basis_masks = []
    for r in pivot_rows:
        mask = 0
        for j in range(n):
            if aug[r, j] == 1:
                mask |= (1 << j)
        basis_masks.append(mask)
        
    # Now, for each original row, find its coordinates in the basis.
    # We can do this by taking the original row and eliminating it using the basis vectors.
    # Since the basis is in RREF, this is straightforward.
    # For each original row mask `om`:
    #   coords = []
    #   for each basis vector `bm` in basis_masks:
    #       if `om` has the pivot bit of `bm` set, then coord is 1, and `om` ^= `bm`.
    #       else coord is 0.
    #   The remaining `om` should be 0.
    
    # We need to know the pivot bit for each basis vector.
    # In RREF, the pivot is the first 1 in the row.
    
    basis_pivots = []
    for bm in basis_masks:
        # Find lowest set bit? Or highest?
        # In our GE, we iterated col 0 to n-1.
        # So the pivot is the lowest set bit?
        # Let's check: `for col in range(n): ... if aug[r, col] == 1`.
        # Yes, we found the first 1 from left (index 0).
        # So pivot is the index of the lowest set bit.
        pivot = 0
        while bm > 0:
            if bm & 1:
                break
            bm >>= 1
            pivot += 1
        # Wait, the `bm` in the loop above is modified. I need the original.
        # Let's re-calculate.
        pass
        
    # Let's just store the pivot index during extraction.
    basis_info = [] # (mask, pivot_index)
    for r in pivot_rows:
        mask = 0
        pivot_idx = -1
        for j in range(n):
            if aug[r, j] == 1:
                mask |= (1 << j)
                if pivot_idx == -1:
                    pivot_idx = j
        basis_info.append((mask, pivot_idx))
        
    # Now express each original row.
    outputs = []
    gates = []
    next_wire = [n + 1]
    
    # First, compute the basis vectors.
    # We have `basis_info`.
    # We need to create wires for these basis vectors.
    # To minimize gates, we can compute them in an order that allows reuse?
    # Since they are independent, we can just compute each one from its constituent inputs.
    # Or, we can use the fact that they are in RREF, so they are "sparse" in a sense?
    # Not necessarily.
    
    # Let's compute each basis vector by XORing its inputs.
    # This is safe and simple.
    basis_wires = []
    for mask, pivot_idx in basis_info:
        # Get indices of set bits
        bits = []
        temp = mask
        while temp:
            if temp & 1:
                bits.append(0) # This is wrong, need to track position
            temp >>= 1
            
        # Better way:
        bits = [j for j in range(n) if (mask >> j) & 1]
        
        if not bits:
            # This shouldn't happen for basis vectors
            basis_wires.append(n) # Zero wire
            continue
            
        if len(bits) == 1:
            basis_wires.append(bits[0])
            continue
            
        # Compute XOR of bits
        # We can do this in a tree or chain. Chain is fine.
        # w = bits[0]
        # for b in bits[1:]:
        #    new_w = next_wire
        #    gates.append([w, b])
        #    w = new_w
        #    next_wire++
        
        # This creates len(bits)-1 gates.
        w = bits[0]
        for b in bits[1:]:
            new_w = next_wire[0]
            next_wire[0] += 1
            gates.append([w, b])
            w = new_w
            
        basis_wires.append(w)
        
    # Now, for each original row, express it as a sum of basis vectors.
    for i in range(m):
        om = row_masks[i]
        
        # Find coordinates
        coords = [0] * len(basis_info)
        temp_om = om
        
        for k, (bm, pivot_idx) in enumerate(basis_info):
            # If temp_om has the pivot bit set, then this basis vector is part of the sum
            if (temp_om >> pivot_idx) & 1:
                coords[k] = 1
                temp_om ^= bm
                
        # temp_om should be 0 now.
        
        # Compute the output wire by XORing the corresponding basis_wires
        # If no coords are 1, output is 0 (wire n)
        # If one coord is 1, output is that basis wire.
        # If multiple, XOR them.
        
        active_wires = [basis_wires[k] for k in range(len(basis_info)) if coords[k] == 1]
        
        if not active_wires:
            outputs.append(n)
        elif len(active_wires) == 1:
            outputs.append(active_wires[0])
        else:
            w = active_wires[0]
            for b in active_wires[1:]:
                new_w = next_wire[0]
                next_wire[0] += 1
                gates.append([w, b])
                w = new_w
            outputs.append(w)
            
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
