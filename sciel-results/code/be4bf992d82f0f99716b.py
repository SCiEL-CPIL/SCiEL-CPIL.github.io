"""
Exact XOR-circuit synthesis using global algebraic decomposition.

Strategy:
1. Compute the Row Reduced Echelon Form (RREF) of the input matrix over GF(2) to find a basis for the row space.
2. Identify the pivot columns corresponding to independent rows.
3. Synthesize each basis vector (row of RREF) into a wire using a balanced XOR tree with a global cache to share sub-expressions.
4. Express the original input matrix rows as linear combinations of the basis rows (via the transformation matrix from Gaussian elimination).
5. Synthesize the final outputs by XORing the appropriate basis wires, again using the global cache to share sub-expressions.

This approach minimizes gate count by:
- Reducing the problem to synthesizing only 'rank' independent vectors.
- Using balanced trees for O(log k) depth and minimal gates for each vector.
- Caching all intermediate XOR results to reuse them across both basis synthesis and output combination.
"""
import time
import numpy as np

def _rref_gf2_with_basis(M):
    """
    Compute RREF of a binary matrix over GF(2).
    Returns:
    - rref: The RREF matrix
    - pivot_cols: List of column indices that have pivots
    - basis_indices: List of original row indices that form the basis (in order of pivot columns)
    - transform: The matrix T such that M = T @ RREF (over GF(2)). 
                 Specifically, if we track the row operations, we can reconstruct T.
                 Actually, it's easier to just track which original rows were swapped/added.
                 Let's return the coefficients of the original rows that make up each RREF row.
    """
    M = np.array(M, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # We will perform Gaussian elimination.
    # We also keep track of the transformation.
    # Let's maintain a matrix 'basis_coeffs' where basis_coeffs[i] is the row in the original space 
    # corresponding to the current row i of the working matrix.
    # Initially, basis_coeffs[i] = e_i (identity row).
    
    if n_rows == 0:
        return M, [], [], np.eye(0, 0, dtype=np.uint8)
        
    work = M.copy()
    basis_coeffs = np.eye(n_rows, dtype=np.uint8)
    
    rank = 0
    pivot_cols = []
    basis_indices = []
    
    for col in range(n_cols):
        if rank >= n_rows:
            break
            
        # Find pivot
        pivot = -1
        for row in range(rank, n_rows):
            if work[row, col] == 1:
                pivot = row
                break
                
        if pivot == -1:
            continue
            
        # Swap pivot row with current rank row
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
            basis_coeffs[[rank, pivot]] = basis_coeffs[[pivot, rank]]
            
        pivot_cols.append(col)
        basis_indices.append(rank) # This is the index in the current permutation, not original. 
                                   # Actually, we need the original index. 
                                   # Let's track original indices separately.
        
        # Eliminate all other rows in this column
        for row in range(n_rows):
            if row != rank and work[row, col] == 1:
                work[row, :] ^= work[rank, :]
                basis_coeffs[row, :] ^= basis_coeffs[rank, :]
                
        rank += 1
        
    # The first 'rank' rows of work are the RREF.
    # The corresponding rows in basis_coeffs are the linear combinations of original rows.
    rref = work[:rank, :]
    # basis_coeffs[:rank, :] gives the coefficients of the original rows for each basis vector.
    # However, the "basis_indices" logic above was slightly off. 
    # The rows 0..rank-1 of basis_coeffs represent the basis vectors in terms of the ORIGINAL rows.
    
    return rref, pivot_cols, list(range(rank)), basis_coeffs[:rank, :]

def _build_xor_tree(indices, wire_map, next_wire, gates):
    """
    Build a balanced XOR tree for a set of input wire indices.
    Returns the wire index representing the XOR of these inputs.
    Uses global caching via wire_map.
    """
    if not indices:
        return None # Should not happen for non-zero vectors
        
    if len(indices) == 1:
        return indices[0]
        
    # Sort indices for consistent caching (though order doesn't matter for XOR, 
    # it helps if we ever use sets, but here we use tuples for wire_map keys? 
    # Actually, the current code uses frozenset in the seed, but that's wrong for XOR if duplicates exist.
    # However, indices are unique input wires. So a frozenset of input indices is a valid key for the XOR result.
    # BUT, if we are XORing intermediate wires, the key must be the tuple of wire IDs.
    # For input wires, the key is the tuple of input IDs.
    # For intermediate wires, the key is the tuple of wire IDs.
    
    # To be safe and general, let's use a tuple of sorted wire IDs as the key.
    key = tuple(sorted(indices))
    if key in wire_map:
        return wire_map[key]
        
    # Balanced split
    mid = len(indices) // 2
    left_indices = indices[:mid]
    right_indices = indices[mid:]
    
    left_wire = _build_xor_tree(left_indices, wire_map, next_wire, gates)
    right_wire = _build_xor_tree(right_indices, wire_map, next_wire, gates)
    
    # Combine
    gates.append([left_wire, right_wire])
    wire_id = next_wire[0]
    next_wire[0] += 1
    
    wire_map[key] = wire_id
    return wire_id

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix.
    
    Args:
        matrix: list of lists or 2D array, n x m binary matrix
        depth_limit: maximum circuit depth (must be <= 16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Input wires are 0..n_cols-1
    # Constant-zero wire is n_cols
    next_wire = [n_cols + 1]
    gates = []
    wire_map = {}
    
    # 1. Compute RREF and basis
    rref, pivot_cols, _, basis_coeffs = _rref_gf2_with_basis(M)
    rank = rref.shape[0]
    
    if rank == 0:
        return {'gates': [], 'outputs': [n_cols] * n_rows}
        
    # 2. Synthesize each basis vector (row of RREF)
    # The basis vectors are linear combinations of the PIVOT columns? 
    # No, RREF rows have 1s in pivot columns and 0s in other pivot columns, but may have 1s in non-pivot columns.
    # The value of the basis vector i is the XOR of the input wires corresponding to the 1s in rref[i, :].
    
    basis_wires = []
    for i in range(rank):
        row = rref[i, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        if not ones:
            basis_wires.append(n_cols) # Zero wire
        else:
            w = _build_xor_tree(ones, wire_map, next_wire, gates)
            basis_wires.append(w)
            
    # 3. Express original rows as combinations of basis vectors
    # The matrix 'basis_coeffs' (returned from _rref_gf2_with_basis) has shape (rank, n_rows).
    # Wait, the previous function returned basis_coeffs[:rank, :] which is (rank, n_rows).
    # Let's re-verify the meaning.
    # In _rref_gf2_with_basis, basis_coeffs starts as Identity (n_rows x n_rows).
    # We swap and add rows.
    # After elimination, work[:rank, :] is RREF.
    # basis_coeffs[:rank, :] tells us: RREF_row_i = sum_j (basis_coeffs[i, j] * Original_Row_j).
    # So, Original_Row_j is NOT directly given.
    # We want to compute Original_Row_j.
    # We know RREF = T * M. So M = T^-1 * RREF.
    # The rows of T are the basis_coeffs? 
    # If RREF_row_i = sum_j T[i,j] M_row_j, then the matrix T has rows basis_coeffs[i].
    # So RREF = T @ M.
    # We want M = T^-1 @ RREF.
    # Since T is invertible (it's a product of elementary row ops on I), we can compute T^-1.
    # Or, simpler: The columns of T^-1 give the coefficients of the basis vectors to form each original row.
    # Let's compute the inverse of the transformation matrix T.
    
    # T is the matrix where T[i, j] is the coefficient of Original Row j in RREF Row i.
    # T = basis_coeffs[:rank, :]. This is rank x n_rows.
    # This is not square if rank < n_rows.
    # Wait, the elimination was performed on all n_rows.
    # The pivot rows are 0..rank-1.
    # The rows rank..n_rows-1 of work are zero.
    # The rows rank..n_rows-1 of basis_coeffs are the null space vectors?
    # Actually, we only care about expressing the original rows M.
    # M has n_rows rows.
    # We have rank basis vectors B_0 ... B_{rank-1} (wires).
    # Each original row M_j is a linear combination of the basis vectors.
    # M_j = sum_{i=0}^{rank-1} c_{i,j} B_i.
    # How to find c_{i,j}?
    # We know RREF = T @ M.
    # Let T_pivot be the rank x rank submatrix of T corresponding to the pivot rows?
    # No, T is rank x n_rows.
    # Let's use the property that the RREF has pivots at pivot_cols.
    # For any original row M_j, its value on the pivot columns determines its coefficients in the basis.
    # Specifically, if we look at the submatrix M[:, pivot_cols], it is full rank (rank x rank).
    # And RREF[:, pivot_cols] is the Identity matrix (if we order pivot cols correctly? No, RREF has 1s in pivot cols, but they are in standard positions).
    # Actually, RREF[:, pivot_cols] is NOT identity unless pivot_cols are 0..rank-1.
    # But RREF has the property that the pivot in row i is at column pivot_cols[i], and it's the only non-zero in that column.
    # So, RREF[:, pivot_cols] is a matrix where entry (i, k) is 1 if pivot_cols[i] == pivot_cols[k]. 
    # Since pivot_cols are distinct, RREF[:, pivot_cols] is the Identity matrix of size rank x rank.
    
    # Therefore, M[:, pivot_cols] = (RREF[:, pivot_cols])^-1 * (M[:, pivot_cols])? No.
    # RREF = T @ M.
    # Let M_p = M[:, pivot_cols] (n_rows x rank).
    # Let R_p = RREF[:, pivot_cols] (rank x rank). We established R_p = I_rank.
    # So I_rank = T @ M_p.
    # This implies M_p = T^-1.
    # So the coefficients of the basis vectors for original row j are the entries of the j-th column of T^-1.
    # Wait, T is rank x n_rows. T^-1 is not defined in the usual sense.
    # But T @ M_p = I_rank.
    # This means the rows of T form a basis for the dual space?
    # Actually, we want c such that M_j = sum_i c_i B_i.
    # In terms of wires: Wire(M_j) = XOR_{i: c_i=1} Wire(B_i).
    # We know Wire(B_i) computes RREF_row_i.
    # So we need to find c such that M_j = sum_i c_i RREF_row_i.
    # This is equivalent to solving sum_i c_i RREF_row_i = M_j.
    # Since RREF_row_i are linearly independent, this system has a unique solution for the coefficients c_i.
    # We can find c by looking at the values on the pivot columns.
    # (sum_i c_i RREF_row_i)[:, pivot_cols] = M_j[:, pivot_cols].
    # Since RREF_row_i has a 1 at pivot_cols[i] and 0 at other pivot_cols, 
    # the i-th entry of the vector M_j[:, pivot_cols] (ordered by pivot_cols) is exactly c_i.
    
    # So, for each original row j, the coefficients c_i are simply M_j[pivot_cols[i]] for i=0..rank-1.
    
    outputs = []
    for j in range(n_rows):
        # Get coefficients
        coeffs = M[j, pivot_cols] # This is a vector of length rank, dtype uint8
        # The wires to XOR are basis_wires[i] where coeffs[i] == 1
        active_wires = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        
        if not active_wires:
            outputs.append(n_cols)
        elif len(active_wires) == 1:
            outputs.append(active_wires[0])
        else:
            # Build XOR tree for these basis wires
            # Note: These are intermediate wires, not input wires.
            # The caching key should be the tuple of these wire IDs.
            w = _build_xor_tree(active_wires, wire_map, next_wire, gates)
            outputs.append(w)
            
    return {'gates': gates, 'outputs': outputs}
