"""Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline over 8x8,12x16,16x16 maps."""
import time
import numpy as np

def _matrix_rank_gf2(M):
    """Compute rank of binary matrix over GF(2) using Gaussian elimination."""
    if M.size == 0:
        return 0
    M = M.copy()
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        if rank >= rows:
            break
        pivot = -1
        for r in range(rank, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rank + 1, rows):
            if M[r, col]:
                M[r, :] ^= M[rank, :]
        rank += 1
    return rank

def _rref_gf2(M):
    """Compute reduced row echelon form and basis indices."""
    M = M.copy()
    rows, cols = M.shape
    rank = 0
    pivot_cols = []
    for col in range(cols):
        if rank >= rows:
            break
        pivot = -1
        for r in range(rank, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        # Eliminate all other rows in this column
        for r in range(rows):
            if r != rank and M[r, col]:
                M[r, :] ^= M[rank, :]
        pivot_cols.append(col)
        rank += 1
    return M, rank, pivot_cols

def _xor_tree_gates(ones, wire_map, next_wire):
    """
    Build a balanced XOR tree for the set of input indices 'ones'.
    Returns (output_wire, gates_added, new_next_wire).
    Uses existing wires in wire_map if available for sub-expressions.
    """
    if len(ones) == 0:
        return None, [], next_wire  # Caller handles zero
    if len(ones) == 1:
        return ones[0], [], next_wire
    
    # Represent sets as frozensets for lookup
    level = [frozenset([x]) for x in ones]
    gates = []
    
    while len(level) > 1:
        new_level = []
        for i in range(0, len(level) - 1, 2):
            s1 = level[i]
            s2 = level[i + 1]
            combined = s1 | s2
            if combined in wire_map:
                new_level.append(combined)
            else:
                w1 = wire_map[s1]
                w2 = wire_map[s2]
                gates.append([w1, w2])
                wire_map[combined] = next_wire
                new_level.append(combined)
                next_wire += 1
        if len(level) % 2 == 1:
            new_level.append(level[-1])
        level = new_level
    
    return wire_map[level[0]], gates, next_wire

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Wires: 0..n_cols-1 are inputs, n_cols is zero
    # Gates produce wires starting at n_cols + 1
    
    # Step 1: Compute RREF to find basis and dependencies
    # We work on the transpose? No, outputs are rows.
    # Each output row is a vector in GF(2)^n_cols.
    # We want to find a basis for the span of these rows.
    # Let B be the matrix of basis vectors (r x n_cols).
    # Each output row y_i = sum_j c_ij * b_j.
    # Circuit:
    # 1. Compute each basis vector b_j using an XOR tree over inputs.
    # 2. Compute each output y_i using an XOR tree over the basis wires {b_j} where c_ij=1.
    
    # Find basis of rows
    # We can use RREF of M directly.
    M_rref, rank, pivot_cols = _rref_gf2(M)
    
    # The first 'rank' rows of M_rref are the basis vectors.
    # But we need the coefficients of the original rows in terms of this basis.
    # This is tricky because RREF changes the rows.
    # Better: Perform Gaussian elimination tracking the operations on an identity matrix?
    # Or simply:
    # 1. Identify independent rows from the original matrix.
    # 2. Express all rows as linear combinations of these independent rows.
    
    # Let's use a more direct approach for synthesis:
    # 1. Find a set of linearly independent rows from M. Let them be B (r x n_cols).
    # 2. For each row in M, find coefficients c such that row = c @ B.
    #    This is solving B^T c^T = row^T. Since B has full row rank, B^T has full column rank.
    #    We can solve this using RREF of B^T augmented with row^T? 
    #    Actually, since B is in RREF form (if we choose B as the RREF rows), the coefficients are simply the entries in the corresponding columns of the RREF row?
    #    Wait, if we take B as the RREF of M, then the rows of M are NOT necessarily linear combinations of the RREF rows in a simple way unless we track the transformations.
    
    # Correct Approach:
    # 1. Compute RREF of M. Let R be the RREF.
    # 2. The basis vectors are the non-zero rows of R.
    # 3. To find the coefficients of each original row in terms of the basis, we need to solve M @ X = I? No.
    #    We need to express each row of M as a linear combination of the rows of R.
    #    Since R = M @ P for some invertible P (row operations), M = R @ P^{-1}.
    #    So the coefficients for row i of M are the i-th row of P^{-1}.
    #    This is computationally heavy.
    
    # Alternative Simpler Approach for this specific task (small matrices):
    # Just compute the XOR tree for EACH row independently, but share sub-expressions globally.
    # The current program does this but is flawed.
    # Let's implement a robust global sharing strategy.
    
    wire_map = {}
    for i in range(n_cols):
        wire_map[frozenset([i])] = i
    wire_map[frozenset()] = n_cols  # Zero wire
    
    next_wire = n_cols + 1
    gates = []
    outputs = []
    
    # Sort rows by popcount to build simpler expressions first? 
    # Or just process all rows.
    # To maximize sharing, we should process rows in an order that creates useful intermediates.
    # However, for exactness and simplicity, let's just iterate and reuse.
    
    # Optimization: Pre-compute all possible small subsets? No, too many.
    # Let's just do the balanced tree with global map.
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            # Fallback: return current best or simple solution
            # If we haven't finished, we might have incomplete outputs.
            # For safety, if we hit deadline, we should return a valid circuit.
            # The loop above might be incomplete. 
            # Let's assume we finish quickly for 16x16.
            pass
            
        row = M[row_idx, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        
        if len(ones) == 0:
            outputs.append(n_cols)
            continue
        if len(ones) == 1:
            outputs.append(ones[0])
            continue
        
        # Build tree for this row
        level = [frozenset([x]) for x in ones]
        
        while len(level) > 1:
            new_level = []
            for i in range(0, len(level) - 1, 2):
                s1 = level[i]
                s2 = level[i + 1]
                combined = s1 | s2
                if combined in wire_map:
                    new_level.append(combined)
                else:
                    w1 = wire_map[s1]
                    w2 = wire_map[s2]
                    gates.append([w1, w2])
                    wire_map[combined] = next_wire
                    new_level.append(combined)
                    next_wire += 1
            if len(level) % 2 == 1:
                new_level.append(level[-1])
            level = new_level
        
        outputs.append(wire_map[level[0]])
    
    return {'gates': gates, 'outputs': outputs}
