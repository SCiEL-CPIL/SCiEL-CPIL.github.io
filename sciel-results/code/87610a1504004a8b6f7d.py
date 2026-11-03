import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 128
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Block-Strassen approach with explicit rank-1 term enumeration
    # 
    # For 2x2 * 2x2 Strassen, the 7 scalar multiplications and their contributions:
    # Let A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A*B
    #
    # m1 = (a10+a00)*(b01-b11)  -> contributes: +1 to C[0,1], +1 to C[1,1]
    # m2 = (a10+a00)*b11        -> contributes: +1 to C[0,0], +1 to C[0,1], +1 to C[1,0], +1 to C[1,1]
    # m3 = a00*(b01-b10)        -> contributes: +1 to C[0,1], -1 to C[1,0], -1 to C[1,1]
    # m4 = a11*(b10-b00)        -> contributes: -1 to C[0,0], +1 to C[1,0], +1 to C[1,1]
    # m5 = (a01+a11)*b00        -> contributes: -1 to C[0,0], -1 to C[0,1], -1 to C[1,0]
    # m6 = (a10-a00)*(b00+b11)  -> contributes: +1 to C[0,0], +1 to C[0,1], +1 to C[1,0], +1 to C[1,1]
    # m7 = (a01-a11)*(b10+b11)  -> contributes: -1 to C[0,1], +1 to C[1,0], +1 to C[1,1]
    #
    # Wait, I need to be more careful. Let me verify:
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    #
    # Strassen:
    # m1 = (a10+a00)*(b01-b11) = a10*b01 - a10*b11 + a00*b01 - a00*b11
    # m2 = (a10+a00)*b11 = a10*b11 + a00*b11
    # m3 = a00*(b01-b10) = a00*b01 - a00*b10
    # m4 = a11*(b10-b00) = a11*b10 - a11*b00
    # m5 = (a01+a11)*b00 = a01*b00 + a11*b00
    # m6 = (a10-a00)*(b00+b11) = a10*b00 + a10*b11 - a00*b00 - a00*b11
    # m7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    #
    # C[0,0] = m2 + m4 - m5 + m6
    #   = (a10*b11 + a00*b11) + (a11*b10 - a11*b00) - (a01*b00 + a11*b00) + (a10*b00 + a10*b11 - a00*b00 - a00*b11)
    #   = a10*b11 + a00*b11 + a11*b10 - a11*b00 - a01*b00 - a11*b00 + a10*b00 + a10*b11 - a00*b00 - a00*b11
    #   = a10*b00 + a10*b11 + a10*b11 + a11*b10 - 2*a11*b00 - a01*b00 - a00*b00
    #   This doesn't simplify to a00*b00 + a01*b10. I'm making errors.
    #
    # Let me use the standard Strassen formulas:
    # C[0,0] = m2 + m4 - m5 + m6
    # C[0,1] = m1 + m5
    # C[1,0] = m2 + m3
    # C[1,1] = m1 - m3 + m4 + m7
    #
    # Verify C[0,1] = m1 + m5:
    # m1 + m5 = (a10*b01 - a10*b11 + a00*b01 - a00*b11) + (a01*b00 + a11*b00)
    # This should equal a00*b01 + a01*b11
    # Clearly not matching. I have the Strassen formulas wrong.
    #
    # Correct Strassen for C = A*B:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a01+a11)*(b00-b10)
    # m7 = (a00+a01)*(b11-b01)
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # Verify C[0,1] = m3 + m5 = a00*(b01-b11) + (a00+a10)*b11 = a00*b01 - a00*b11 + a00*b11 + a10*b11 = a00*b01 + a10*b11
    # But C[0,1] should be a00*b01 + a01*b11. Still wrong!
    #
    # I'm confusing myself. Let me just use the well-known Strassen decomposition.
    # 
    # Actually, for the purpose of this task, let me just return the schoolbook decomposition.
    # The structural change I should make is to try a different optimization approach.
    
    # Try alternating minimization with structured initialization
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower-rank decomposition using random restarts and ALS
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        
        target_rank = 120 - attempt * 5  # Try 120, 115, 110, 105, 100
        if target_rank < 90:
            break
            
        np.random.seed(42 + attempt)
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        # Alternating Least Squares
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each row i of U, minimize ||T_target - V ⊙ W||_F
            # This is a least squares problem
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed i, we want to find U[i*m:(i+1)*m, :] such that
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all j,k
            
            # This is a linear system: U_row @ M = 1_vector
            # where M[j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] ... no, this isn't quite right.
            
            # The constraint is: for each (j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let M[j, r] = V[j*p+k, r] * W[k*n+i, r] but this depends on k too.
            
            # Actually, for fixed i, the constraint is:
            # For all j in [0,m), k in [0,p): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # 
            # This can be written as: for each j, the vector (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])_{k} = 1
            # Let A_j[r, k] = V[j*p+k, r] * W[k*n+i, r]
            # Then we need: U[i*m+j, :] @ A_j = 1_vector (length p)
            # This is a linear system for each j.
            
            # Solve for U[i*m+j, :] using least squares
            for j in range(m):
                # Build the matrix: for each r and k, element is V[j*p+k, r] * W[k*n+i, r]
                # Shape: (p, target_rank)
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Solve U[i*m+j, :] @ A_mat = ones(p)
                # This is: A_mat.T @ U[i*m+j, :].T = ones(p)
                # Use least squares
                try:
                    U[i*m+j, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat.T, np.ones(p), rcond=None)
                except:
                    pass
        
        # Check validity
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
