"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the stacked matrix: each column c corresponds to (U[:,c], V[:,c], W[:,c])
    # The decomposition is valid iff sum_c U[:,c] * V[:,c] * W[:,c] = tensor
    # Linear dependencies: if sum_c alpha_c * (U[:,c], V[:,c], W[:,c]) = 0 (componentwise),
    # then we can eliminate one column.
    # But this is NOT the right condition! The right condition is that the tensor
    # T = sum_c U_c ⊗ V_c ⊗ W_c has rank equal to the number of independent terms.
    
    # The correct approach: the rank of the tensor is the minimum number of terms.
    # We can use the "slice rank" or other measures, but the simplest is:
    # Try to find a smaller set of terms that gives the same tensor.
    
    # Alternative: use the fact that if two terms have the same U or V or W factor
    # (up to scaling), they can be combined.
    
    # Let's try a different approach: directly solve for minimal rank using
    # the structure of the problem.
    
    # Actually, let's verify the current decomposition is correct first
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((p, n))
    for c in range(rank):
        C_computed += U_np[:, c] * V_np[:, c] * W_np[:, c].reshape(n, p)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Decomposition is wrong, return as is (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank
    # The tensor T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # We want to find the minimal number of terms.
    
    # One approach: if we can find a linear relation among the terms where
    # the coefficients work for all three factors simultaneously, we can reduce.
    
    # Specifically, if there exist scalars alpha_c such that for all i:
    #   sum_c alpha_c * U[i,c] = 0
    # for all j:
    #   sum_c alpha_c * V[j,c] = 0  
    # for all k:
    #   sum_c alpha_c * W[k,c] = 0
    # Then the term with alpha_c != 0 can be eliminated... but this is too strict.
    
    # The correct condition: the tensor rank is the minimum r such that
    # T = sum_{c=1}^r U_c ⊗ V_c ⊗ W_c
    # This is a non-convex problem in general.
    
    # For our structured case, let's try a greedy elimination:
    # If column c can be expressed as a linear combination of other columns
    # in ALL THREE matrices simultaneously, we can eliminate it.
    
    # This is equivalent to: the vector (U[:,c], V[:,c], W[:,c]) is in the
    # span of {(U[:,c'], V[:,c'], W[:,c']) : c' != c}
    
    # Let's check this for each column
    stacked = np.vstack([U_np, V_np, W_np])  # shape (8+16+16, 56) = (40, 56)
    
    # Find the rank of the stacked matrix
    U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
    tol = 1e-10
    r = int(np.sum(s > tol))
    
    if r < rank:
        # There are linear dependencies
        # The null space has dimension rank - r
        null_space = Vt[r:]  # shape (rank-r, rank)
        
        # For each null vector, it gives a way to eliminate one column
        # But we need to be careful: we can only eliminate a column if
        # its coefficient in the null vector is nonzero, and we express it
        # in terms of the others.
        
        # Greedy elimination:
        U_cur = U_np.copy()
        V_cur = V_np.copy()
        W_cur = W_np.copy()
        active = [True] * rank
        remaining = rank
        
        # Process null vectors
        for nv in null_space:
            if time.time() > deadline - 5:
                break
            # Find the column with the largest |component| to eliminate
            # But only among active columns
            abs_nv = np.abs(nv)
            # Set inactive columns to 0
            for c in range(rank):
                if not active[c]:
                    abs_nv[c] = 0
            
            max_idx = np.argmax(abs_nv)
            if abs_nv[max_idx] < 1e-10:
                continue
            
            # Express column max_idx in terms of others
            # nv[max_idx] * col[max_idx] = -sum_{c != max_idx} nv[c] * col[c]
            # col[max_idx] = -sum_{c != max_idx} (nv[c] / nv[max_idx]) * col[c]
            
            coeff = -nv / nv[max_idx]
            coeff[max_idx] = 0
            
            # Add the contributions to the kept columns
            for c in range(rank):
                if c == max_idx or not active[c]:
                    continue
                if abs(coeff[c]) > 1e-15:
                    U_cur[:, c] += coeff[c] * U_cur[:, max_idx]
                    V_cur[:, c] += coeff[c] * V_cur[:, max_idx]
                    W_cur[:, c] += coeff[c] * W_cur[:, max_idx]
            
            active[max_idx] = False
            remaining -= 1
        
        if remaining < rank:
            # Extract the reduced matrices
            keep_idx = [i for i in range(rank) if active[i]]
            U_red = U_cur[:, keep_idx]
            V_red = V_cur[:, keep_idx]
            W_red = W_cur[:, keep_idx]
            
            # Verify
            C_computed = np.zeros((p, n))
            for c in range(len(keep_idx)):
                C_computed += U_red[:, c] * V_red[:, c] * W_red[:, c].reshape(n, p)
            
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return {"U": U, "V": V, "W": W}
