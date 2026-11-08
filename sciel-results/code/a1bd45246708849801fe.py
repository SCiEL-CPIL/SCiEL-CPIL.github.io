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

    # Strategy 2: Different j pairings
    if time.time() < deadline - 30:
        # Try j_pairs = [(0,1), (3,4)], remaining_j = [2]
        def try_split_j01_34():
            j_pairs = [(0,1), (3,4)]
            k_pairs = [(0,1), (2,3), (4,5)]
            remaining_j = [2]
            return build_strassen_split(j_pairs, k_pairs, remaining_j)
        
        U2, V2, W2, rank2 = try_split_j01_34()
        if validate(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2

    # Strategy 3: Try different k pairings
    if time.time() < deadline - 30:
        # Try k_pairs = [(0,2), (1,3), (4,5)] - non-adjacent pairs
        def try_split_k02_13_45():
            j_pairs = [(0,1), (2,3)]
            k_pairs = [(0,2), (1,3), (4,5)]
            remaining_j = [4]
            return build_strassen_split(j_pairs, k_pairs, remaining_j)
        
        U3, V3, W3, rank3 = try_split_k02_13_45()
        if validate(U3, V3, W3, rank3) and rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3

    # Strategy 4: Use 2x2x3 blocks with a better algorithm
    if time.time() < deadline - 25:
        # For 2x2x3, we can use R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11
        # Or try to find a better bound.
        # R(2,2,3) is known to be at most 11, possibly lower.
        # Let's use the simple 7+4 approach.
        
        def build_2x2x3_block(U, V, W, col, i0, i1, j0, j1, k0, k1, k2):
            """Build 2x2x3 block: Strassen on (j0,j1,k0,k1) + schoolbook on k2."""
            col = strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1)
            for i in [i0, i1]:
                for j in [j0, j1]:
                    U[i*m+j, col] += 1
                    V[j*p+k2, col] += 1
                    W[k2*n+i, col] += 1
                    col += 1
            return col
        
        def try_2x2x3_split():
            # Split p=6 into three 2x2x3 blocks: (k0,k1,k2), (k3,k4,k5)
            # For each j pair: 2x2x3 + 2x2x3 = 11 + 11 = 22
            # For remaining j: 2x6 schoolbook = 12
            # Total: 22 + 22 + 12 = 56 (worse than 54)
            # So this doesn't help.
            return None
        
        # Try m=5 as 2+3: use 2x3x6 for one part and 2x2x6 for another
        # R(2,3,6) <= R(2,2,6) + R(2,1,6)
        # R(2,2,6) <= 3*R(2,2,2) = 21
        # R(2,1,6) = 12
        # So R(2,3,6) <= 33
        # R(2,2,6) <= 21
        # Total: 33 + 21 = 54 (same)
        
        # Try to find a better R(2,3,6)
        # R(2,3,6) <= R(2,3,3) + R(2,3,3)
        # R(2,3,3) <= R(2,2,3) + R(2,1,3) = 11 + 6 = 17
        # So R(2,3,6) <= 34 (worse)
        
        # The best we can do with these splits is 54.
        pass

    # Strategy 5: Try rank reduction on the best decomposition
    if time.time() < deadline - 15:
        U5, V5, W5, rank5 = try_rank_reduction_linear(best_U, best_V, best_W, best_rank)
        if rank5 < best_rank:
            best_U, best_V, best_W, best_rank = U5, V5, W5, rank5

    # Strategy 6: Try a completely different approach - use the fact that
    # the tensor is a direct sum structure and try to exploit it.
    if time.time() < deadline - 10:
        # The 2x5x6 tensor can be seen as:
        # For i=0: sum over j,k of A[0,j]*B[j,k]*W[k,0]
        # For i=1: sum over j,k of A[1,j]*B[j,k]*W[k,1]
        # These are coupled through the B matrix.
        
        # Try to find a decomposition where the W matrix has a special structure
        # that allows for rank reduction.
        
        # This is a research-level problem, so we'll just try a few more splits.
        
        # Try j_pairs = [(1,2), (3,4)], remaining_j = [0]
        def try_split_j12_34():
            j_pairs = [(1,2), (3,4)]
            k_pairs = [(0,1), (2,3), (4,5)]
            remaining_j = [0]
            return build_strassen_split(j_pairs, k_pairs, remaining_j)
        
        U6, V6, W6, rank6 = try_split_j12_34()
        if validate(U6, V6, W6, rank6) and rank6 < best_rank:
            best_U, best_V, best_W, best_rank = U6, V6, W6, rank6

    # Convert to lists for the return interface
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
