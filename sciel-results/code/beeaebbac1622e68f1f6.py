"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition for matmul tensor."""
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                if j == j2 and i == i3 and k2 == k3:
                                    continue
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
        return True
    
    # Build the matmul tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy: Try to find a lower-rank decomposition by exploiting
    # algebraic structure. For 3x4x4, try rank reduction by finding
    # linear dependencies or using structured decompositions.
    
    # Approach: Try to merge columns. Two columns c1, c2 corresponding to
    # (i1,j1,k1) and (i2,j2,k2) can be merged if their rank-1 tensors sum
    # to something that can be expressed with fewer terms.
    
    # More practical approach: Try random rank-(rank0-1) decompositions
    # by projecting the tensor onto a lower-dimensional subspace.
    
    # Actually, let's try a different structural approach:
    # Use the fact that for small matrices, we can try to find a decomposition
    # with rank < n*m*p by exploiting the specific structure.
    
    # For 3x4x4, let's try to see if we can find a rank-47 decomposition
    # by checking if any column is in the span of the others in a tensor sense.
    
    # A column c0 is redundant if there exist coefficients alpha_c for c != c0
    # such that for all (a,b,c):
    # U[a,c0]*V[b,c0]*W[c,c0] = sum_{c!=c0} alpha_c * U[a,c]*V[b,c]*W[c,c]
    
    # This is a system of equations. Let's try to solve it for a few columns.
    
    # For efficiency, let's try a different approach: use numerical rank reduction.
    # Represent the tensor as a 3D array and try to find a low-rank decomposition.
    
    # Try to find a rank-r decomposition for r < 48
    # Use a heuristic: try to find r rank-1 tensors that sum to T
    
    # For each candidate rank r, try to find U, V, W such that
    # sum_c U[:,c] outer V[:,c] outer W[:,c] = T
    
    # This is a non-convex optimization problem. Let's try a simple approach:
    # Start with the schoolbook decomposition and try to remove one column at a time,
    # adjusting the remaining columns to compensate.
    
    # For column c0, we want to find U', V', W' (with c0 removed) such that
    # sum_{c != c0} U'[a,c]*V'[b,c]*W'[c,c] = T[a,b,c] for all (a,b,c)
    
    # This is equivalent to finding U', V', W' such that
    # sum_{c != c0} U'[a,c]*V'[b,c]*W'[c,c] = T[a,b,c] - U0[a,c0]*V0[b,c0]*W0[c,c0]
    
    # Since U0[a,c0], V0[b,c0], W0[c,c0] are 0 or 1, the RHS is T minus a single
    # rank-1 tensor.
    
    # Let's try this for each column and see if we can find a solution.
    
    for c0 in range(rank0):
        if time.time() > deadline - 10:
            break
        
        # Build the target tensor: T - rank-1 tensor for column c0
        T_target = T.copy()
        T_target -= U0[:, c0][:, None, None] * V0[:, c0][None, :, None] * W0[:, c0][None, None, :]
        
        # Now we need to find U', V', W' with rank-1 = 47 columns such that
        # sum_c U'[a,c]*V'[b,c]*W'[c,c] = T_target[a,b,c]
        
        # Start with the other 47 columns from the schoolbook decomposition
        other_cols = [c for c in range(rank0) if c != c0]
        U_init = U0[:, other_cols].copy()
        V_init = V0[:, other_cols].copy()
        W_init = W0[:, other_cols].copy()
        
        # Check if the initial guess already works
        # (It won't, because we removed a column, but let's verify)
        
        # Try to adjust U_init, V_init, W_init to match T_target
        # This is a hard problem. Let's try a simple approach: 
        # add small perturbations and see if we can close the gap.
        
        # Actually, for an exact decomposition with integer/half-integer entries,
        # we need a different approach.
        
        # Let's try: can we express the removed column as a linear combination
        # of the others? I.e., does there exist alpha_c such that
        # U0[:,c0] = sum_c alpha_c * U0[:,c]
        # V0[:,c0] = sum_c alpha_c * V0[:,c]
        # W0[:,c0] = sum_c alpha_c * W0[:,c]
        
        # This is a system of 12 + 16 + 12 = 40 equations in 47 unknowns.
        other_cols_list = [c for c in range(rank0) if c != c0]
        A_mat = np.zeros((40, len(other_cols_list)))
        b_vec = np.zeros(40)
        
        for idx, c in enumerate(other_cols_list):
            A_mat[:12, idx] = U0[:, c]
            A_mat[12:28, idx] = V0[:, c]
            A_mat[28:40, idx] = W0[:, c]
        
        b_vec[:12] = U0[:, c0]
        b_vec[12:28] = V0[:, c0]
        b_vec[28:40] = W0[:, c0]
        
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-9:
                # Column c0 can be removed!
                new_U = U0[:, other_cols_list]
                new_V = V0[:, other_cols_list]
                new_W = W0[:, other_cols_list]
                
                if verify(new_U, new_V, new_W):
                    best_rank = rank0 - 1
                    best_U, best_V, best_W = new_U, new_V, new_W
                    print(f"Found rank {best_rank} decomposition by removing column {c0}!")
                    break
        except Exception as e:
            pass
    
    # If we didn't find a rank-47 decomposition, try other approaches.
    # For now, return the best found.
    
    if best_rank < rank0:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
# EVOLVE-BLOCK-END
