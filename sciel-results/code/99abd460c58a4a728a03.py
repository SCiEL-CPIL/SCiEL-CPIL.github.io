"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill

    # --- Schoolbook decomposition (rank = n*m*p) ---
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # --- Fast verification using einsum ---
    def verify_fast(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j."""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Check for integer/half-integer entries ---
    def is_valid_entries(*mats):
        for mat in mats:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # --- Convert to list format ---
    def to_list(mat):
        # Round to nearest half-integer for cleanliness
        mat_rounded = np.round(mat * 2) / 2.0
        # Check if all are integers
        if np.allclose(mat_rounded, np.round(mat_rounded), atol=1e-9):
            return np.round(mat_rounded).astype(int).tolist()
        else:
            # Return as float list with half-integer values
            return mat_rounded.tolist()

    # --- Try to find a lower-rank decomposition ---
    def try_lower_rank(r, max_attempts=50):
        """Try to find a rank-r decomposition by random search."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + r)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random sparse matrices with entries in {-1, 0, 1}
            # Use a structured approach: try to build U and V with some structure
            U_mat = np.zeros((n * m, r))
            V_mat = np.zeros((m * p, r))
            
            # Initialize with random signs for non-zero entries
            for c in range(r):
                # For U: each row (i,k) gets a non-zero in some columns
                for i in range(n):
                    for k in range(m):
                        if np.random.random() < 0.7:  # 70% chance of non-zero
                            U_mat[i * m + k, c] = np.random.choice([-1, 1])
                
                # For V: each row (k,j) gets a non-zero in some columns
                for k in range(m):
                    for j in range(p):
                        if np.random.random() < 0.7:
                            V_mat[k * p + j, c] = np.random.choice([-1, 1])
            
            # Now solve for W
            W_mat = np.zeros((p * n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    # For fixed (j,i), we need to find W[j*n+i, :] such that:
                    # For each k in 0..m-1: sum_c U[i*m+k,c] * V[k*p+j,c] * W[j*n+i,c] = 1
                    # This is a system of m equations in r unknowns.
                    
                    A_row = np.zeros((m, r))
                    for k in range(m):
                        for c in range(r):
                            A_row[k, c] = U_mat[i * m + k, c] * V_mat[k * p + j, c]
                    
                    b_row = np.ones(m)
                    
                    # Solve A_row @ w_row = b_row
                    try:
                        sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=None)
                        # Check consistency
                        if not np.allclose(A_row @ sol, b_row, atol=1e-9):
                            consistent = False
                            break
                        W_mat[j * n + i, :] = sol
                    except:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            # Verify the full decomposition
            if verify_fast(U_mat, V_mat, W_mat):
                # Check for valid entries
                if is_valid_entries(U_mat, V_mat, W_mat):
                    return {
                        "U": to_list(U_mat),
                        "V": to_list(V_mat),
                        "W": to_list(W_mat)
                    }
        
        return None

    # --- Try ranks from low to high ---
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(4, 14, 14) = 14
    # But for matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Actually, the rank of the matrix multiplication tensor is at least max(n*m, m*p, p*n)
    # For n=2, m=2, p=7: max(4, 14, 14) = 14
    
    # Try ranks from 14 up to schoolbook rank
    for r in range(14, best_rank):
        if time.time() > deadline:
            break
        
        result = try_lower_rank(r, max_attempts=30)
        if result is not None:
            best = result
            best_rank = r
            # If we found a lower rank, we could stop early, but let's try to find even lower
            # Actually, since we're going from low to high, the first one we find is the best
            break
    
    return best
# EVOLVE-BLOCK-END
