import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Strategy: Decompose B into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    # Rank = R(3,3,3) + R(3,3,3) + R(3,3,1)
    # R(3,3,1) = 9 (schoolbook)
    # We search for R(3,3,3) < 27
    
    def solve_3x3x3(rank_33):
        """
        Find a rank-33 decomposition for 3x3x3 matrix multiplication.
        Returns U33, V33, W33 of shape (9, rank_33) or None.
        """
        n3, m3, p3 = 3, 3, 3
        # Random search
        for _ in range(500):
            if time.time() > deadline - 10:
                return None
            
            U33 = np.random.randn(n3 * m3, rank_33)
            V33 = np.random.randn(m3 * p3, rank_33)
            
            # Solve for W33
            # Condition: sum_c U33[i*m3+j1, c] * V33[j2*p3+k, c] * W33[k*n3+i, c] = delta_{j1,j2}
            # For fixed i, k:
            # sum_c (U33[i*m3:(i+1)*m3, c]) (V33[k::p3, c])^T * W33[k*n3+i, c] = I_m3
            # Vectorized:
            # A_ik[:, c] = kron(V33[k::p3, c], U33[i*m3:(i+1)*m3, c])
            # A_ik @ w_ik = vec(I_m3)
            
            W33 = np.zeros((p3 * n3, rank_33))
            valid = True
            b_ik = np.eye(m3).flatten()
            
            for i in range(n3):
                for k in range(p3):
                    A_ik = np.zeros((m3 * m3, rank_33))
                    for c in range(rank_33):
                        v_c = V33[k::p3, c] # m3
                        u_c = U33[i*m3:(i+1)*m3, c] # m3
                        A_ik[:, c] = np.kron(v_c, u_c)
                    
                    try:
                        w_ik, residuals, rank_val, s = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                        if len(residuals) > 0:
                            if residuals[0] > 1e-8:
                                valid = False
                                break
                        else:
                            residual = np.linalg.norm(A_ik @ w_ik - b_ik)
                            if residual > 1e-8:
                                valid = False
                                break
                        W33[k*n3+i, :] = w_ik
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Final verification
                for i in range(n3):
                    for k in range(p3):
                        for j1 in range(m3):
                            for j2 in range(m3):
                                expected = 1.0 if j1 == j2 else 0.0
                                actual = 0.0
                                for c in range(rank_33):
                                    actual += U33[i*m3+j1, c] * V33[j2*p3+k, c] * W33[k*n3+i, c]
                                if abs(actual - expected) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U33, V33, W33
        return None

    # Search for 3x3x3 decomposition
    found_33 = None
    for r33 in range(21, 27):
        if time.time() > deadline - 20:
            break
        res = solve_3x3x3(r33)
        if res is not None:
            found_33 = (r33, res)
            break
    
    if found_33:
        r33, (U33, V33, W33) = found_33
        total_rank = 2 * r33 + 9
        
        # Construct full 3x3x7 decomposition
        U_full = np.zeros((9, total_rank))
        V_full = np.zeros((21, total_rank))
        W_full = np.zeros((21, total_rank))
        
        # Part 1: A * B1 (cols 0-2)
        U_full[:, 0:r33] = U33
        for j in range(3):
            for k in range(3):
                # V33 index: j*3 + k
                # V_full index: j*7 + k
                V_full[j*7 + k, 0:r33] = V33[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                # W33 index: k*3 + i
                # W_full index: k*3 + i
                W_full[k*3 + i, 0:r33] = W33[k*3 + i, :]
        
        # Part 2: A * B2 (cols 3-5)
        offset = r33
        U_full[:, offset:offset+r33] = U33
        for j in range(3):
            for k in range(3):
                # V33 index: j*3 + k
                # V_full index: j*7 + (3+k)
                V_full[j*7 + 3 + k, offset:offset+r33] = V33[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                # W33 index: k*3 + i
                # W_full index: (3+k)*3 + i
                W_full[(3+k)*3 + i, offset:offset+r33] = W33[k*3 + i, :]
        
        # Part 3: A * B3 (col 6)
        offset2 = 2 * r33
        # Schoolbook for 3x3x1
        col = 0
        for i in range(3):
            for j in range(3):
                U_full[i*3 + j, offset2 + col] = 1
                V_full[j*7 + 6, offset2 + col] = 1
                W_full[6*3 + i, offset2 + col] = 1
                col += 1
        
        best_rank = total_rank
        best_U = U_full
        best_V = V_full
        best_W = W_full

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
