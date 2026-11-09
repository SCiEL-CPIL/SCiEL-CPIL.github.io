"""Exact XOR-circuit synthesis using Greedy Minimum-Weight Basis."""

def synthesize(matrix, depth_limit):
    import numpy as np
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Wire numbering: 0..n-1 are inputs, n is constant zero
    gates = []
    
    def new_gate(a, b):
        gates.append([int(a), int(b)])
        return n + 1 + (len(gates) - 1)
    
    # Step 1: Find a basis with low weights
    # We use a greedy approach: sort rows by weight, add to basis if independent.
    # We also need to track the coefficients to express original rows in terms of the basis.
    
    # To express original rows, we can just re-reduce them against the found basis.
    # But we need the basis vectors themselves to synthesize.
    
    # Let's collect candidate rows sorted by weight.
    # We need to handle duplicates? No, basis handles independence.
    
    # We will store the basis vectors as numpy arrays.
    basis = []
    basis_indices = [] # indices in original matrix? Not strictly needed for synthesis, just for logic.
    
    # Sort row indices by weight (number of 1s)
    weights = np.sum(matrix, axis=1)
    sorted_indices = np.argsort(weights)
    
    # Perform Gaussian elimination to find basis
    # We work in GF(2)
    current_basis = []
    
    for idx in sorted_indices:
        row = matrix[idx].copy()
        # Reduce row by existing basis
        for b in current_basis:
            if np.any(row):
                # Find pivot in b
                # b is in RREF? Not necessarily. 
                # Let's maintain basis in a form where we can reduce easily.
                # Standard: for each pivot column, there is a basis vector.
                pass
        
    # This simple loop is hard to manage without a proper structure.
    # Let's use a standard RREF approach but with column pivoting to keep vectors sparse?
    # Or just standard RREF and hope the weights are okay.
    
    # Actually, let's just use the RREF basis but synthesize it smartly.
    # The issue with RREF was dense basis vectors.
    # Let's try to find a basis by selecting the **sparsest** independent rows.
    
    # Algorithm for Sparse Basis:
    # 1. Iterate through rows sorted by weight.
    # 2. Maintain a list of basis vectors.
    # 3. For each row, check if it's in the span of current basis.
    #    - If yes, skip.
    #    - If no, add to basis.
    # 4. To check span membership and get coefficients, we can maintain the basis in RREF form
    #    internally, but store the **original** sparse rows for synthesis?
    #    NO. The synthesis must be of the actual vectors.
    #    If we add a row $r$ to the basis, we synthesize $r$.
    #    Later, we express other rows $m$ as $c \cdot r_{basis}$.
    #    The $r_{basis}$ are the original rows we picked.
    
    # So:
    # 1. Pick a set of independent rows $R_{basis}$ from $M$ with low weights.
    # 2. Synthesize wires for each $r \in R_{basis}$.
    # 3. For each $m \in M$, find coeffs $c$ such that $m = \sum c_i r_i$.
    # 4. Synthesize output as XOR of basis wires.
    
    # How to find $R_{basis}$ and coeffs?
    # We can do Gaussian elimination on $M^T$? No.
    # We can do Gaussian elimination on $M$ but swap rows to prefer low weight.
    
    # Let's do this:
    # Create a copy of M.
    # Keep track of the "original" row that each current row represents?
    # No, that's for expressing original in terms of reduced.
    # We want the opposite: Express original in terms of a subset of original.
    
    # This is equivalent to finding a subset of columns? No, rows.
    
    # Let's use the property:
    # If we perform Gaussian elimination on M, we get RREF R.
    # R = E M.
    # M = E^{-1} R.
    # The non-zero rows of R are a basis.
    # But they are dense.
    
    # Alternative:
    # Perform elimination on M, but **don't** normalize.
    # Just find pivot rows.
    # The pivot rows (from the original M, before modification? No, modified) span the space.
    # If we modify rows, the basis vectors change.
    
    # Let's stick to the **Greedy Selection of Original Rows**.
    # 1. Sort indices by weight.
    # 2. Maintain a basis matrix $B_{curr}$ (initially empty).
    # 3. For each index $i$:
    #    a. Check if $M[i]$ is in the span of $B_{curr}$.
    #    b. If not, add $M[i]$ to $B_{curr}$.
    # 4. At the end, $B_{curr}$ is a basis of original rows.
    # 5. To get coefficients for any $M[j]$, we just solve $M[j] = C B_{curr}$.
    #    Since $B_{curr}$ is a set of rows, we can form the matrix $B$ (stacked).
    #    $M[j] = C B \implies C = M[j] B^+ $ (pseudo-inverse? No, GF(2) inverse of pivot submatrix).
    
    # Implementation of Step 3 (Span Check):
    # We can maintain $B_{curr}$ in RREF form for fast span checking,
    # but we must remember the **original** vectors for synthesis.
    
    basis_original = []
    basis_rref = [] # RREF versions of the basis vectors for checking
    
    for idx in sorted_indices:
        row = matrix[idx].copy()
        # Reduce row by basis_rref
        # If it becomes 0, it's in the span.
        # If not, it's independent.
        
        # To reduce:
        # For each pivot in basis_rref, if row has 1 at pivot, XOR with that basis vector.
        
        # We need to know the pivots of basis_rref.
        # Let's store (pivot_col, rref_vector)
        
        is_in_span = True
        temp_row = row.copy()
        for pivot_col, b_vec in basis_rref:
            if temp_row[pivot_col] == 1:
                temp_row = (temp_row + b_vec) % 2
        
        if np.any(temp_row):
            # Independent. Add to basis.
            # We need to put this row into RREF form for future checks.
            # Find pivot for this row (first 1).
            pivot_col = np.argmax(temp_row) # First 1
            # Normalize? In GF(2), pivot is always 1.
            # Eliminate this pivot from other basis vectors?
            # Yes, to keep RREF.
            
            new_rref = temp_row.copy()
            
            # Eliminate this pivot from existing basis vectors
            new_basis_rref = []
            for p_col, b_vec in basis_rref:
                if b_vec[pivot_col] == 1:
                    b_vec = (b_vec + new_rref) % 2
                new_basis_rref.append((p_col, b_vec))
            
            # Add new vector
            new_basis_rref.append((pivot_col, new_rref))
            
            # Sort by pivot col? Helpful for reduction order.
            new_basis_rref.sort(key=lambda x: x[0])
            
            basis_rref = new_basis_rref
            basis_original.append(matrix[idx]) # Store the ORIGINAL row for synthesis
            
    # Now we have basis_original.
    # Synthesize wires for each basis vector.
    
    basis_wires = []
    for b in basis_original:
        indices = np.where(b > 0)[0]
        if len(indices) == 0:
            w = n # zero wire
        elif len(indices) == 1:
            w = indices[0]
        else:
            w = indices[0]
            for i in indices[1:]:
                w = new_gate(w, i)
        basis_wires.append(w)
        
    # Now express each original row in terms of basis_original.
    # We can use the basis_rref to find coefficients.
    # For a row m, reduce it by basis_rref.
    # The coefficients are 1 for each basis vector used in reduction.
    
    outputs = []
    for row in matrix:
        temp_row = row.copy()
        coeffs = [0] * len(basis_rref)
        
        # Reduce in reverse order of pivots? Or forward?
        # RREF allows reduction in any order if we use the pivot property.
        # Standard: iterate through basis vectors.
        # If temp_row has 1 at pivot of b_i, then coeff[i] = 1, and temp_row ^= b_i.
        
        for i, (pivot_col, b_vec) in enumerate(basis_rref):
            if temp_row[pivot_col] == 1:
                coeffs[i] = 1
                temp_row = (temp_row + b_vec) % 2
                
        # If temp_row is not zero, something is wrong (should be zero if basis is complete)
        
        # Synthesize output from basis_wires using coeffs
        if sum(coeffs) == 0:
            outputs.append(n)
        elif sum(coeffs) == 1:
            outputs.append(basis_wires[coeffs.index(1)])
        else:
            w = basis_wires[coeffs.index(1)]
            for i in range(1, len(coeffs)):
                if coeffs[i] == 1:
                    w = new_gate(w, basis_wires[i])
            outputs.append(w)
            
    return {'gates': gates, 'outputs': outputs}
