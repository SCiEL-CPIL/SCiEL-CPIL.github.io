"""Rewritten program for matrix multiplication 2x5x6 tensor decomposition."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 130  # Safety margin before 150s kill

    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col

    def schoolbook_block(U, V, W, col, i_list, j_list, k_list):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col

    def build_strassen_split(j_pairs, k_pairs, remaining_j):
        """Build decomposition using Strassen on 2x2x2 blocks + schoolbook for rest."""
        rank = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                rank += 7
        for j in remaining_j:
            rank += 2 * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        return U, V, W, rank

    def try_rank_reduction_linear(U, V, W, rank):
        """
        Try to reduce rank by finding if any column is redundant.
        For each column c, check if removing it and solving a linear system
        for the remaining columns can still satisfy all constraints.
        """
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        
        # Build constraint system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear in general, but if we fix U and V and solve for W (or vice versa),
        # it becomes linear.
        
        # Strategy: Try to find a column c such that we can express its contribution
        # as a linear combination of others by adjusting the remaining columns.
        # This is equivalent to checking if the rank-1 tensor for column c is in the
        # span of the others (in the tensor space).
        
        # More practical: use linear algebra on the "flattened" constraint matrix.
        # For fixed U, V: the constraint is linear in W.
        # For fixed U, W: the constraint is linear in V.
        # For fixed V, W: the constraint is linear in U.
        
        # Let's try: fix U and V, see if we can solve for W with one fewer column.
        # For each (i,j,k): sum_{c != c_remove} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a linear system in W.
        
        num_constraints = n * m * p
        num_w_vars = p * n
        
        for c_remove in range(rank):
            if time.time() > deadline - 20:
                break
            
            # Build linear system: A * W_vec = b
            # where A[row, w_idx] = U[i*m+j, c] * V[j*p+k, c] for c != c_remove
            # and b[row] = 1
            
            # Actually, for each (i,j,k) and each w_idx = k'*n+i', the coefficient is:
            # sum_{c != c_remove} U[i*m+j,c] * V[j*p+k,c] * delta_{k',k} * delta_{i',i}
            # = U[i*m+j,c] * V[j*p+k,c] if k'=k and i'=i, else 0
            # Wait, that's not right. Let me re-think.
            #
            # The constraint is: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # If we fix U and V, then for each (i,j,k), we have:
            # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            #
            # This is a linear equation in the variables W[0], W[1], ..., W[p*n-1].
            # The coefficient of W[w_idx] in the equation for (i,j,k) is:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * delta_{w_idx, k*n+i}
            # = U[i*m+j,c] * V[j*p+k,c] if w_idx == k*n+i, else 0
            #
            # Wait, that means each equation only involves one W variable!
            # So the system is: for each (i,j,k): (sum_c U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] ... 
            # No wait, W[k*n+i,c] depends on c too. Let me be more careful.
            #
            # W is a (p*n) x rank matrix. W[w_idx, c] is the entry.
            # The constraint for (i,j,k) is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i, c] = 1
            #
            # If we vectorize W as a vector of length p*n*rank, then this is linear.
            # But if we want to reduce rank, we're reducing the number of columns.
            
            # Alternative approach: check if the set of rank-1 tensors {u_c ⊗ v_c ⊗ w_c}
            # is linearly dependent. If so, we might be able to remove one.
            
            # Practical check: for each c_remove, compute the "residual" after removing column c_remove,
            # and check if it can be expressed as a combination of the remaining rank-1 tensors.
            # This is a tensor completion problem, which is NP-hard in general.
            
            # Simpler heuristic: check if the contribution of column c_remove is small
            # relative to the total, and if removing it leaves a valid decomposition
            # after rescaling the remaining columns.
            
            # For now, just check if removing the column and not adjusting still gives
            # a valid decomposition (unlikely but cheap to check).
            r_new = rank - 1
            U_new = np.delete(U, c_remove, axis=1)
            V_new = np.delete(V, c_remove, axis=1)
            W_new = np.delete(W, c_remove, axis=1)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
                break
        
        return best_U, best_V, best_W, best_rank

    def try_gaussian_elimination_on_constraints(U, V, W, rank):
        """
        More sophisticated rank reduction: use the structure of the constraints.
        The tensor T[i*m+j, j*p+k, k*n+i] = 1 has a specific structure.
        We can try to find a lower-rank decomposition by solving a system of
        polynomial equations, but that's hard. Instead, we use a heuristic:
        try to find linear combinations of the rank-1 terms that cancel out.
        """
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        
        # Try to find two columns that can be combined.
        # If u_a ⊗ v_a ⊗ w_a + u_b ⊗ v_b ⊗ w_b can be replaced by
        # a single rank-1 term (or fewer), we can reduce rank.
        # This is generally not possible, but we can check specific patterns.
        
        # For Strassen, the 7 terms have specific structure. Check if any pair
        # of Strassen terms can be merged.
        
        # This is complex, so we skip for now and rely on other strategies.
        return best_U, best_V, best_W, best_rank

    def build_combined_strassen(j_pairs, k_pairs, remaining_j, use_shared=False):
        """Build decomposition with optional shared intermediate computations."""
        if use_shared:
            # Use shared computations between adjacent Strassen blocks
            # This can reduce rank by exploiting overlapping subproblems
            rank = 0
            for j0, j1 in j_pairs:
                for k0, k1 in k_pairs:
                    rank += 7
            for j in remaining_j:
                rank += 2 * p
            
            # Try to find savings from shared computations
            # When two Strassen blocks share a j index, we can sometimes merge terms
            # For now, use the standard count but with potential for optimization
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for j0, j1 in j_pairs:
                for k0, k1 in k_pairs:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            for j in remaining_j:
                for k in range(p):
                    U[0*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+0, col] += 1
                    col += 1
                    U[1*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+1, col] += 1
                    col += 1
            return U, V, W, rank
        else:
            return build_strassen_split(j_pairs, k_pairs, remaining_j)

    # Strategy 1: Standard Strassen split - pair up j indices and k indices
    def try_strassen_split():
        # m=5: pair (0,1), (2,3), remaining {4}
        # p=6: pair (0,1), (2,3), (4,5)
        j_pairs = [(0,1), (2,3)]
        k_pairs = [(0,1), (2,3), (4,5)]
        remaining_j = [4]
        return build_strassen_split(j_pairs, k_pairs, remaining_j)

    U, V, W, rank = try_strassen_split()
    if not validate(U, V, W, rank):
        # Fallback to full schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))

    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Strategy 2: Try all possible j pairings with different k pairings
    if time.time() < deadline - 30:
        import itertools
        
        # All ways to partition 5 j indices into pairs + singles
        j_options = [
            ([(0,1), (2,3)], [4]),
            ([(0,1), (3,4)], [2]),
            ([(1,2), (3,4)], [0]),
            ([(0,2), (1,3)], [4]),
            ([(0,2), (3,4)], [1]),
            ([(0,3), (1,4)], [2]),
            ([(0,4), (1,2)], [3]),
            ([(1,3), (2,4)], [0]),
            ([(1,4), (2,3)], [0]),
            ([(2,4), (0,3)], [1]),
        ]
        
        k_options = [
            [(0,1), (2,3), (4,5)],
            [(0,1), (2,4), (3,5)],
            [(0,2), (1,3), (4,5)],
            [(0,2), (1,4), (3,5)],
            [(0,3), (1,2), (4,5)],
            [(0,3), (1,5), (2,4)],
            [(0,4), (1,2), (3,5)],
            [(0,4), (1,3), (2,5)],
            [(0,5), (1,2), (3,4)],
            [(0,5), (1,3), (2,4)],
        ]
        
        for j_pairs, rem_j in j_options:
            for k_pairs in k_options:
                if time.time() > deadline - 20:
                    break
                U2, V2, W2, rank2 = build_strassen_split(j_pairs, k_pairs, rem_j)
                if validate(U2, V2, W2, rank2) and rank2 < best_rank:
                    best_U, best_V, best_W, best_rank = U2, V2, W2, rank2

    # Strategy 3: Try rank reduction with linear algebra
    if time.time() < deadline - 15:
        U5, V5, W5, rank5 = try_rank_reduction_linear(best_U, best_V, best_W, best_rank)
        if rank5 < best_rank:
            best_U, best_V, best_W, best_rank = U5, V5, W5, rank5

    # Convert to lists for the return interface
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
