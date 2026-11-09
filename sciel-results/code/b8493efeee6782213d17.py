    def synthesize(matrix, depth_limit):
        import numpy as np
        
        matrix = np.array(matrix, dtype=np.uint8)
        m, n = matrix.shape
        
        if m == 0 or n == 0:
            return {'gates': [], 'outputs': [n] * m}
            
        # 1. Greedy Basis Selection
        # Represent rows as integers for fast XOR and bit_count
        rows = []
        for i in range(m):
            val = 0
            for j in range(n):
                if matrix[i, j]:
                    val |= (1 << j)
            rows.append(val)
            
        # We need to track which original rows sum to the current row
        # mask[i] is a bitmask of original row indices
        masks = [1 << i for i in range(m)]
        
        basis_vals = []
        basis_masks = []
        
        # To allow efficient removal/selection, we can just scan
        # m is small (<=16), n is small (<=16)
        
        active = list(range(m))
        
        while active:
            # Find index in active with min bit_count
            min_idx = -1
            min_weight = n + 1
            for idx in active:
                w = rows[idx].bit_count()
                if w < min_weight:
                    min_weight = w
                    min_idx = idx
            if min_idx == -1:
                break
            if rows[min_idx] == 0:
                active.remove(min_idx)
                continue
                
            # Select min_idx as basis vector
            basis_vals.append(rows[min_idx])
            basis_masks.append(masks[min_idx])
            
            # Remove from active
            active.remove(min_idx)
            
            # Eliminate this row from all other active rows
            b_val = rows[min_idx]
            b_mask = masks[min_idx]
            
            for idx in active:
                rows[idx] ^= b_val
                masks[idx] ^= b_mask
                
        r = len(basis_vals)
        
        # 2. Synthesize Basis Wires
        gates = []
        def new_gate(a, b):
            gates.append([int(a), int(b)])
            return n + 1 + (len(gates) - 1)
            
        basis_wires = []
        for bv in basis_vals:
            # bv is an integer representing the row
            # Find indices of 1s
            ones = [j for j in range(n) if (bv >> j) & 1]
            if not ones:
                # Zero vector? Should not happen if we picked non-zero
                basis_wires.append(n)
            elif len(ones) == 1:
                basis_wires.append(ones[0])
            else:
                w = ones[0]
                for j in ones[1:]:
                    w = new_gate(w, j)
                basis_wires.append(w)
                
        # 3. Compute Coordinates C such that A = C B
        # B is r x n. We want C (m x r).
        # We know basis_masks: b_k = sum_{i in mask_k} A_i.
        # This means B = D A, where D is r x m.
        # D_{k, i} = 1 if i in basis_masks[k].
        
        # We want A = C B = C D A.
        # So C D is the projection.
        # Actually, we can just solve for C directly using the basis vectors.
        # For each row i of A, find c_i such that A_i = sum c_{ij} b_j.
        
        # Let's build D (r x m)
        D = np.zeros((r, m), dtype=np.uint8)
        for k in range(r):
            mask = basis_masks[k]
            for i in range(m):
                if (mask >> i) & 1:
                    D[k, i] = 1
        
        # We have B = D A.
        # We want C such that A = C B.
        # This is equivalent to finding C such that C D A = A.
        # Since B is a basis, the mapping is unique on the row space.
        # We can pick any r independent columns of B to solve.
        # Let P be the indices of pivot columns in B.
        
        B_mat = np.array(basis_vals, dtype=np.uint8).reshape(r, -1) # Wait, basis_vals are ints.
        # Reconstruct B_mat from basis_vals
        B_mat = np.zeros((r, n), dtype=np.uint8)
        for k in range(r):
            bv = basis_vals[k]
            for j in range(n):
                if (bv >> j) & 1:
                    B_mat[k, j] = 1
                    
        # Find pivot columns in B_mat
        # Perform GE on B_mat to find pivots
        B_copy = B_mat.copy()
        pivots = []
        for j in range(n):
            # Find row with 1 in col j
            pivot_row = -1
            for i in range(r):
                if B_copy[i, j] == 1:
                    pivot_row = i
                    break
            if pivot_row == -1:
                continue
            # Swap to top? We just need to record pivot
            # Eliminate other rows
            for i in range(r):
                if i != pivot_row and B_copy[i, j] == 1:
                    B_copy[i] ^= B_copy[pivot_row]
            pivots.append(j)
            
        if len(pivots) < r:
            # Should not happen if basis is correct
            pass
            
        # Solve C[:, pivots] = A[:, pivots] * B[:, pivots]^-1
        # Let B_p = B_mat[:, pivots] (r x r)
        # Let A_p = matrix[:, pivots] (m x r)
        # C_p = C[:, pivots] (m x r)
        # C_p B_p = A_p  => C_p = A_p B_p^-1
        
        B_p = B_mat[:, pivots]
        A_p = matrix[:, pivots]
        
        # Compute inverse of B_p over GF(2)
        # Use numpy? Or manual.
        # Since r is small, manual is fine.
        
        def inv_gf2(mat):
            # mat is r x r
            n_r = mat.shape[0]
            # Augment with identity
            aug = np.hstack([mat, np.eye(n_r, dtype=np.uint8)])
            for i in range(n_r):
                # Find pivot
                pivot = -1
                for k in range(i, n_r):
                    if aug[k, i] == 1:
                        pivot = k
                        break
                if pivot == -1:
                    continue
                aug[[i, pivot]] = aug[[pivot, i]]
                # Eliminate other rows
                for k in range(n_r):
                    if k != i and aug[k, i] == 1:
                        aug[k] ^= aug[i]
            return aug[:, n_r:]
            
        B_p_inv = inv_gf2(B_p)
        
        # C_p = A_p @ B_p_inv
        C_p = (A_p.astype(np.int8) @ B_p_inv.astype(np.int8)) % 2
        
        # Now we have C_p (m x r).
        # We need the full C?
        # We only need to synthesize outputs.
        # Output i is sum of basis_wires[j] where C[i, j] == 1.
        # But we only have C_p (columns corresponding to pivots).
        # Do we have the other columns of C?
        # Yes, because C is unique.
        # But wait, C is m x r. We solved for C[:, pivots].
        # We need C[:, all_cols].
        # Is C[:, non_pivots] determined?
        # Yes, because B[:, non_pivots] is a linear combination of B[:, pivots].
        # And A[:, non_pivots] is a linear combination of A[:, pivots].
        # So C[:, non_pivots] is determined by C[:, pivots] and the relations in B.
        # Actually, we can just use the basis vectors directly to compute the outputs?
        # No, we need the coefficients.
        
        # Alternative:
        # We have B = D A.
        # We want C such that A = C B.
        # This implies C D is a projection.
        # Actually, we can just compute the coordinates of each A_i in the basis B.
        # We can do this by solving B^T x = A_i^T? No.
        
        # Let's just compute the full C by solving the system for each row?
        # Or, note that C_p is m x r.
        # We can verify if C_p B_p = A_p.
        # To get the full C, we can use the fact that C B = A.
        # C [B_p, B_n] = [A_p, A_n].
        # C B_p + C B_n = A_p + A_n.
        # We know C B_p = A_p.
        # So C B_n = A_n.
        # B_n is r x (n-r). A_n is m x (n-r).
        # This is a system of equations for C[:, non_pivots].
        # But B_n is dependent on B_p.
        
        # Simpler:
        # We have the basis vectors b_1...b_r.
        # We have the wires w_1...w_r.
        # For each output i, we need to find which w_j to XOR.
        # We can just solve the linear system for each i using the pivot structure.
        
        # Let's construct the full C matrix.
        # We have C_p.
        # We need C_full.
        # We can compute C_full by solving C_full B = A.
        # We already have C_p = A_p B_p^{-1}.
        # We can compute the rest?
        
        # Actually, we can just use the `basis_masks` and `rows`?
        # No, `basis_masks` tells us how b's are made from A's.
        
        # Let's just solve for C_full directly using numpy lstsq? No, GF(2).
        
        # We can extend the pivot solution.
        # We have C_p.
        # We need C_n such that C_n B_n = A_n.
        # Since B_n columns are in span of B_p, let B_n = B_p K.
        # Then A_n = C B_n = C B_p K = A_p K.
        # So A_n = A_p K.
        # This is consistent.
        # It doesn't give C_n uniquely?
        # Wait, C is unique.
        # C = C_p [I, ...]?
        
        # Let's just calculate the coefficients for each output i by checking
        # which combination of basis wires produces A_i.
        # Since r is small, we can iterate? No, 2^r is too big.
        
        # We can compute C_full by:
        # C_full = A B^+ (Pseudoinverse)?
        
        # Let's use the property:
        # We have D (r x m) such that B = D A.
        # We want C (m x r) such that A = C B.
        # This means C D A = A.
        # Let P be the projection matrix.
        
        # Actually, we can just compute C by solving the linear system for each column of C?
        # No, for each row of A.
        
        # Let's compute C_full using the pivot columns.
        # We have C_p.
        # We can compute the transformation matrix from pivots to non-pivots.
        # Let B = [B_p, B_n].
        # We found pivots by GE.
        # During GE, we can track how non-pivot columns are formed.
        
        # Let's redo GE on B_mat and track the coefficients.
        # This is getting complex.
        
        # Simplest Fallback:
        # Since r is small, and we have the wires w_1...w_r.
        # We can just try to match A_i? No.
        
        # Let's use the `basis_masks` to our advantage.
        # We know b_k = sum_{i in mask_k} A_i.
        # We want A_i = sum_k c_{ik} b_k.
        # This is a linear system in c.
        # Let's construct the matrix D (r x m).
        # We want C (m x r).
        # We know B = D A.
        # We want A = C B.
        # Substitute B: A = C D A.
        # So C D is the identity on the row space.
        
        # We can compute C = A (A^T A)^{-1} A^T? No, GF(2).
        
        # Let's just solve C B = A using the pivot columns we found.
        # We have C_p = A_p B_p^{-1}.
        # We need C_n.
        # We know B_n = B_p M for some M (r x (n-r)).
        # And A_n = A_p M.
        # We want C_n B_n = A_n.
        # C_n B_p M = A_p M.
        # (C_n B_p - A_p) M = 0.
        # This doesn't fix C_n.
        
        # Wait, C is unique.
        # C B = A.
        # C [B_p, B_n] = [A_p, A_n].
        # C B_p = A_p.
        # C B_n = A_n.
        # Since B_n is in the column space of B_p, the second equation is dependent on the first?
        # No, C is the same matrix.
        # If we fix C on the pivot columns, is C fixed?
        # Yes, because the rows of C are determined by the equation C B = A.
        # If B has full row rank, the solution C is unique.
        # The values C[:, pivots] determine C entirely?
        # Yes, because any other column of B is a linear combination of the pivot columns.
        # So the constraint C b_j = a_j for non-pivot j is automatically satisfied if it's satisfied for pivots?
        # No.
        # But C is a matrix.
        # If we know C[:, pivots], do we know C[:, non_pivots]?
        # Yes, because C is the unique solution.
        # We can compute C[:, non_pivots] by solving?
        
        # Actually, we can just use C_p as the coefficients if we reorder the basis?
        # No.
        
        # Let's just compute the full C by solving the system for all columns.
        # We can use the fact that C = A_p B_p^{-1} is only for the pivot part.
        
        # Let's try a different angle:
        # We have the wires w_1...w_r.
        # We can just compute the output by XORing the wires that are "active" in the basis decomposition of A_i.
        # How to find the active wires?
        # We can solve the system B^T x = A_i^T?
        
        # Let's just use numpy to solve the linear system for each row.
        # For each i, solve sum c_j b_j = A_i.
        # This is B^T c = A_i^T.
        
        B_T = B_mat.T
        # For each i:
        outputs = []
        for i in range(m):
            # Solve B_T c = A_i
            # Use the pivot structure
            # We have B_p_inv.
            # We can solve for c_p (coordinates in pivot basis)?
            # No, c is length r.
            
            # Let's just use the C_p we computed.
            # C_p is m x r.
            # Is C_p the full C?
            # Let's check: C_p B_p = A_p.
            # Does C_p B_n = A_n?
            # If yes, then C_p is the full C.
            
            # Let's verify for one row.
            # If it holds, we use C_p.
            
            # It should hold because C is unique.
            # If C_p B_p = A_p, and C is the unique solution to C B = A, then C_p must be C.
            # Why? Because the columns of B_p span the column space of B.
            # So the constraints C B_p = A_p imply C B = A.
            
            # So C_p IS the full C.
            
            c_row = C_p[i]
            # c_row is a vector of length r.
            # Find non-zero indices
            active_wires = [j for j in range(r) if c_row[j] == 1]
            
            if not active_wires:
                outputs.append(n)
            elif len(active_wires) == 1:
                outputs.append(basis_wires[active_wires[0]])
            else:
                w = basis_wires[active_wires[0]]
                for j in active_wires[1:]:
                    w = new_gate(w, basis_wires[j])
                outputs.append(w)
                
        return {'gates': gates, 'outputs': outputs}
