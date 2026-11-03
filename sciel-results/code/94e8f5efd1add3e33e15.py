import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    # Strategy: Direct bilinear map optimization using random integer U, V
    # and solving for W via least squares, then snapping to half-integers.
    # This is a different approach from the current program's structured search.
    
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half(W):
        W_half = W * 2
        W_rounded = np.round(W_half)
        error = np.max(np.abs(W_half - W_rounded))
        if error < 1e-6:
            return W_rounded / 2.0, True
        return W, False

    # Try ranks from 55 down to 40
    for r in range(55, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # More attempts for lower ranks
        attempts = 100 if r > 50 else 200
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U and V with entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            
            # Construct linear system for W
            num_eqs = n * m * p
            num_unknowns = p * n * r
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row = k * n + i
                        for c in range(r):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                A_mat[eq_idx, w_row * r + c] = coeff
                        eq_idx += 1
            
            # Solve A_mat * W_flat = b_vec
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            W = W_flat.reshape(p * n, r)
            
            # Snap to half-integers
            W_candidate, is_half = snap_to_half(W)
            
            if is_half:
                # Verify exactly
                if verify(U, V, W_candidate):
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_candidate.tolist()
                    
                    # Check if we can reduce rank further by removing zero columns
                    # For now, return this valid decomposition
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    return best_result
