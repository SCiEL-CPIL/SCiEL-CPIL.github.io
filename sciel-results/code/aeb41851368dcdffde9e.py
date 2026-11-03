"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    def check_decomp(U, V, W, r):
        """Check if decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_int(mat):
        """Round to nearest half-integer"""
        return np.round(mat * 2) / 2
    
    # Try alternating minimization for various ranks
    np.random.seed(123)
    
    for r in range(16, min(schoolbook_rank + 1, 33)):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Fix V and W, solve for U (linear)
                # For each (i,j): U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # More precisely: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then A_ij @ U[i*m+j, :].T = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ u = ones(p) for u = U[i*m+j, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Fix U and W, solve for V (linear)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((m, r))  # Wait, need to think about this
                        # For each i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # So: sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Fix U and V, solve for W (linear)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # So: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if check_decomp(U, V, W, r):
                    # Try rounding to half-integers
                    U_r = to_half_int(U)
                    V_r = to_half_int(V)
                    W_r = to_half_int(W)
                    if check_decomp(U_r, V_r, W_r, r):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        break
                    elif check_decomp(U, V, W, r):
                        # Use unrounded if it works (shouldn't happen with exact check)
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                        break
            
            if best_rank == r:
                break
        
        if best_rank == r:
            break
    
    def to_list(mat):
        result = []
        for row in mat:
            r_list = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    r_list.append(int(round(val)))
                else:
                    # Check if it's a half-integer
                    half = val * 2
                    if abs(half - round(half)) < 1e-9:
                        r_list.append(float(val))
                    else:
                        r_list.append(float(val))
            result.append(r_list)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
