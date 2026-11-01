import time
import random
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x7), C is n x p (2x7)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Decomposition: sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,k,j
    
    deadline = time.time() + 145.0
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)
    
    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)
    
    def mat_to_list(mat):
        """Convert matrix to list of lists of Python floats/ints"""
        return mat.tolist()
    
    def try_random_rank(r, num_trials=50):
        """Try to find a rank-r decomposition using random U,V and solving for W"""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random U and V with entries in half-integers
            # Values: -1.5, -1, -0.5, 0, 0.5, 1, 1.5
            half_ints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            U_mat = half_ints[np.random.randint(0, len(half_ints), (n*m, r))]
            V_mat = half_ints[np.random.randint(0, len(half_ints), (m*p, r))]
            
            # For each (j, i) pair, solve for W[j*n+i, :]
            # We need: for each k in 0..m-1:
            #   sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1
            # This is a system: A_{j,i} @ w = b where A_{j,i}[k, c] = U[i*m+k,c] * V[k*p+j,c]
            # and b[k] = 1 for all k
            
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    
                    # Build the m x r matrix A
                    A_row = np.zeros((m, r))
                    for k in range(m):
                        A_row[k, :] = U_mat[i*m + k, :] * V_mat[k*p + j, :]
                    
                    b_row = np.ones(m)
                    
                    # Solve A_row @ w = b_row
                    try:
                        sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=1e-10)
                        # Check if the solution is valid
                        if not np.allclose(A_row @ sol, b_row, atol=1e-8):
                            consistent = False
                            break
                        W_mat[j*n + i, :] = sol
                    except Exception:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            # Verify the solution
            if verify(U_mat, V_mat, W_mat):
                # Check for half-integer entries
                if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                    return {
                        "U": mat_to_list(U_mat),
                        "V": mat_to_list(V_mat),
                        "W": mat_to_list(W_mat)
                    }
        
        return None
    
    def try_structured_rank(r):
        """Try structured decompositions with specific patterns"""
        # For small ranks, try specific algebraic structures
        
        if r <= n * m * p:
            # Try to use symmetry: since n=2, m=2, we can try to exploit structure
            # One idea: use the fact that the tensor is symmetric in some sense
            
            for trial in range(30):
                if time.time() > deadline:
                    return None
                
                # Try random sparse patterns
                U_mat = np.zeros((n*m, r))
                V_mat = np.zeros((m*p, r))
                
                # Place random non-zero entries
                for c in range(r):
                    # Each column of U has non-zero entries at random positions
                    u_pos = np.random.choice(n*m, size=min(2, n*m), replace=False)
                    v_pos = np.random.choice(m*p, size=min(3, m*p), replace=False)
                    
                    half_ints = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]) # Exclude 0 for sparse non-zeros? Or include?
                    # If we pick 0, it's not a non-zero. Let's pick from non-zero halves.
                    U_mat[u_pos, c] = half_ints[np.random.randint(0, len(half_ints), size=len(u_pos))]
                    V_mat[v_pos, c] = half_ints[np.random.randint(0, len(half_ints), size=len(v_pos))]
                
                # Solve for W
                W_mat = np.zeros((p*n, r))
                consistent = True
                
                for j in range(p):
                    for i in range(n):
                        if time.time() > deadline:
                            return None
                        
                        A_row = np.zeros((m, r))
                        for k in range(m):
                            A_row[k, :] = U_mat[i*m + k, :] * V_mat[k*p + j, :]
                        
                        b_row = np.ones(m)
                        
                        try:
                            sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=1e-10)
                            if not np.allclose(A_row @ sol, b_row, atol=1e-8):
                                consistent = False
                                break
                            W_mat[j*n + i, :] = sol
                        except Exception:
                            consistent = False
                            break
                    
                    if not consistent:
                        break
                
                if not consistent:
                    continue
                
                    if verify(U_mat, V_mat, W_mat):
                        if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                            return {
                                "U": mat_to_list(U_mat),
                                "V": mat_to_list(V_mat),
                                "W": mat_to_list(W_mat)
                            }
        
        return None
    
    # Try ranks from 27 down to 1
    for r in range(27, 0, -1):
        if time.time() > deadline:
            break
        
        # Try random approach
        sol = try_random_rank(r, num_trials=100 if r <= 20 else 50)
        if sol is not None:
            return sol
        
        # Try structured approach
        sol = try_structured_rank(r)
        if sol is not None:
            return sol
    
    return best
# EVOLVE-BLOCK-END
