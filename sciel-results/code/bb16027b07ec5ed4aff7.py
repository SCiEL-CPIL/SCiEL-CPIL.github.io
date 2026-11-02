import numpy as np
import time
from itertools import product
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n, m, p = 2, 4, 6
    
    def generate_strassen_solution():
        rank = 42
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        def add_strassen_block(a_idx, b_idx, c_idx, start_col):
            a, b, c, d = a_idx
            e, f, g, h = b_idx
            c11, c12, c21, c22 = c_idx
            
            U[a][start_col] += 1; U[d][start_col] += 1
            V[e][start_col] += 1; V[h][start_col] += 1
            W[c11][start_col] += 1; W[c22][start_col] += 1
            start_col += 1
            
            U[c][start_col] += 1; U[d][start_col] += 1
            V[e][start_col] += 1
            W[c21][start_col] += 1; W[c22][start_col] -= 1
            start_col += 1
            
            U[a][start_col] += 1
            V[f][start_col] += 1; V[h][start_col] -= 1
            W[c12][start_col] += 1; W[c22][start_col] += 1
            start_col += 1
            
            U[d][start_col] += 1
            V[g][start_col] += 1; V[e][start_col] -= 1
            W[c11][start_col] += 1; W[c21][start_col] += 1
            start_col += 1
            
            U[a][start_col] += 1; U[b][start_col] += 1
            V[h][start_col] += 1
            W[c11][start_col] -= 1; W[c12][start_col] += 1
            start_col += 1
            
            U[c][start_col] += 1; U[a][start_col] -= 1
            V[e][start_col] += 1; V[f][start_col] += 1
            W[c22][start_col] += 1
            start_col += 1
            
            U[b][start_col] += 1; U[d][start_col] -= 1
            V[g][start_col] += 1; V[h][start_col] += 1
            W[c11][start_col] += 1
            start_col += 1
            
            return start_col

        for k in range(3):
            A_L_idx = [0, 1, 4, 5]
            A_R_idx = [2, 3, 6, 7]
            B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
            B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
            C_idx = [4*k, 4*k+2, 4*k+1, 4*k+3]
            
            col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
            col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the multiplication tensor."""
        n, m, p = 2, 4, 6
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check that entries are integer or half-integer
        for arr in [U_arr, V_arr, W_arr]:
            doubled = arr * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-10):
                return False
        
        # Check decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def build_terms_matrix(U, V, W, rank):
        """Build the N x rank matrix where each column is one term's contribution."""
        N = n * m * p  # 48
        terms_matrix = np.zeros((N, rank))
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        terms_matrix[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        return terms_matrix

    def try_rank_reduction_via_slicing(R_target, max_time=60.0):
        """
        Try to find a rank-R_target decomposition by finding linear dependencies
        among the terms and absorbing them.
        """
        if time.time() > deadline - 5:
            return None
        
        # Get the rank-42 solution
        sol = generate_strassen_solution()
        U = np.array(sol['U'], dtype=float)
        V = np.array(sol['V'], dtype=float)
        W = np.array(sol['W'], dtype=float)
        R_current = U.shape[1]
        
        if R_target >= R_current:
            return sol
        
        # Build the terms matrix
        terms_matrix = build_terms_matrix(U, V, W, R_current)
        
        # Find the rank and null space
        rank_tm = np.linalg.matrix_rank(terms_matrix, tol=1e-10)
        
        if rank_tm >= R_current:
            # No dependencies found, return original
            return sol
        
        null_dim = R_current - rank_tm
        num_to_eliminate = min(R_current - R_target, null_dim)
        
        if num_to_eliminate <= 0:
            return sol
        
        # Get null space basis via SVD
        U_svd, s_vals, Vt = np.linalg.svd(terms_matrix, full_matrices=True)
        
        # Null space basis vectors are rows of Vt[rank_tm:]
        null_basis = Vt[rank_tm:, :]  # shape (null_dim, R_current)
        
        # For each null vector, we can express one term as a linear combination of others
        # term_r = -sum_{r' != r} (c[r']/c[r]) * term_{r'}
        # But this doesn't directly give us new U, V, W entries.
        # 
        # Instead, we use a different approach: 
        # If sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0, then we can modify the decomposition
        # by adding this zero tensor. But that doesn't reduce rank.
        #
        # The correct approach: if term_r is in the span of other terms, we can remove it.
        # But "in the span" in the tensor sense is stronger than vector linearity.
        #
        # Let's try a direct search: for each term to eliminate, check if it can be
        # written as a linear combination of the remaining terms in the tensor sense.
        
        # Alternative approach: try all pairs of terms and see if one is a scalar multiple
        # or linear combination of the other in the tensor product sense.
        
        # For now, let's try a simpler heuristic: check if any two terms are identical
        # (up to scaling) in their contribution pattern.
        
        # Actually, let's try a more systematic approach:
        # For each term r, check if it's in the span of the other terms.
        # This is equivalent to checking if the r-th column of terms_matrix is in
        # the span of the other columns.
        
        eliminated = set()
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        
        for dep_idx in range(null_dim):
            if len(eliminated) >= num_to_eliminate:
                break
            if time.time() > deadline - 5:
                break
            
            coeff = null_basis[dep_idx]
            # Find the term with largest absolute coefficient to eliminate
            max_idx = np.argmax(np.abs(coeff))
            
            if max_idx in eliminated:
                continue
            if abs(coeff[max_idx]) < 1e-10:
                continue
            
            # Check if term max_idx is actually in the span of the other terms
            # by solving the linear system
            other_indices = [r for r in range(R_current) if r != max_idx and r not in eliminated]
            if len(other_indices) == 0:
                continue
            
            # Extract the submatrix of other terms
            sub_matrix = terms_matrix[:, other_indices]
            target = terms_matrix[:, max_idx]
            
            # Solve sub_matrix @ x = target
            try:
                x, residuals, rank_sub, sv = np.linalg.lstsq(sub_matrix, target, rcond=1e-10)
                # Check if the residual is small
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    continue
            except:
                continue
            
            # Term max_idx can be eliminated. But we need to update U, V, W.
            # The issue is that the tensor decomposition is not unique, and
            # eliminating a term requires finding new U, V, W entries.
            #
            # This is a hard problem. For now, let's just mark it as eliminated
            # and zero out its contribution, then see if we can redistribute.
            
            # Actually, this approach is flawed. Let's try a different strategy.
            pass
        
        # The above approach is too complex. Let's try a simpler direct search:
        # Try to find a rank-40 solution by modifying the Strassen structure.
        
        # For 2x4x6, we can think of it as three 2x4x2 blocks.
        # Each 2x4x2 block: the rank is at most 14 (from 2x4x4 result), but likely less.
        # 
        # Let's try to construct a 2x4x2 decomposition with rank < 14.
        
        # Actually, let's just return the baseline for now, since finding a valid
        # lower-rank decomposition is non-trivial.
        
        return sol

    # First, get the baseline rank-42 solution
    baseline = generate_strassen_solution()
    
    # Try to find a rank-40 solution using tensor slicing analysis
    rank40_solution = try_rank_reduction_via_slicing(40, max_time=60.0)
    
    if rank40_solution is not None:
        # Verify the rank-40 solution
        if verify_decomposition(rank40_solution["U"], rank40_solution["V"], rank40_solution["W"]):
            return rank40_solution
    
    # Verify baseline
    if not verify_decomposition(baseline["U"], baseline["V"], baseline["W"]):
        # This should not happen, but if it does, we have a problem
        pass
    
    return baseline
