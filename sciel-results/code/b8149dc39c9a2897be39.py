import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    def check_decomp(U, V, W):
        """Check if U, V, W satisfy the tensor equation."""
        n, m, p = DIMENSIONS
        # T[i*m+j, j'*p+k, k'*n+i] = delta_{j,j'} * delta_{k,k'}
        # Sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i, r] = delta_{j,j'} * delta_{k,k'}
        # Efficient check: for each (i, j, j', k, k'), check the sum
        # But that's 4*4*4*7*7 = 3136 checks, each with a dot product of length r
        # Alternative: form the "contraction" and compare
        # C[i,k] from decomposition = sum_{j,j',k'} ... no
        # Let's do it directly for non-zero and some zero entries
        r = U.shape[1]
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] should be 1
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check some zero entries (sample)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2:
                        continue
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
                        if abs(val) > 1e-9:
                            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for k2 in range(p):
                        if k == k2:
                            continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k2, :] * W[k*n+i, :])
                        if abs(val) > 1e-9:
                            return False
        return True
    
    def to_half_int_matrix(M):
        """Convert to half-integer representation, return None if not valid."""
        # Check if all entries are close to integers or half-integers
        M2 = M * 2
        M2_rounded = np.round(M2)
        if np.max(np.abs(M2 - M2_rounded)) > 1e-6:
            return None
        return M2_rounded / 2.0
    
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try random search for lower ranks
    np.random.seed(42)
    
    for target_rank in range(80, rank_sb):  # Try from 80 up to 111
        if time.time() > deadline - 30:
            break
        
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Generate random U, V, W with entries in {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            
            if check_decomp(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
    
    # Convert to list format
    def to_list(M):
        return [[float(M[i][j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
