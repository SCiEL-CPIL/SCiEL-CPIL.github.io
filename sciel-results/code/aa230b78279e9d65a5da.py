"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Schoolbook as fallback
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
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # STRUCTURAL APPROACH: Fix U and V, solve linear systems for W.
    # For fixed (i,k): 4 linear equations in R unknowns (W[k*n+i,0..R-1])
    # B_{i,k,j}[c] = U[i*m+j,c] * V[j*p+k,c]
    # B_{i,k,j} · W[k*n+i] = 1, for j=0,1,2,3
    
    def try_rank(target_rank):
        """Try to find a decomposition of given rank using structured linear solve."""
        if time.time() > deadline:
            return None
        
        max_attempts = 50000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly initialize U and V with small integer entries
            entries = [-2, -1, 0, 1, 2]
            U_mat = np.random.choice(entries, size=(n*m, target_rank))
            V_mat = np.random.choice(entries, size=(m*p, target_rank))
            
            W_mat = np.zeros((p*n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 4xR system: B_{i,k,j} · w = 1
                    B = np.zeros((m, target_rank))
                    for j in range(m):
                        B[j] = U_mat[i*m+j] * V_mat[j*p+k]
                    
                    # Solve B · w = ones(m)
                    # Use least squares (should be exact if consistent)
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        # Check if the solution is valid (residuals should be ~0)
                        if len(residuals) > 0 and residuals[0] > 1e-9:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    W_mat[k*n+i] = w
            
            if not valid:
                continue
            
            # Verify the full decomposition
            all_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                        if abs(val - 1) > 1e-9:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                if not all_valid:
                    break
            
            if all_valid:
                return {"U": U_mat, "V": V_mat, "W": W_mat}
        
        return None

    # Try ranks from 44 down to 35
    for target_rank in range(44, 34, -1):
        result = try_rank(target_rank)
        if result is not None:
            best_result = {
                "U": result["U"].tolist(),
                "V": result["V"].tolist(),
                "W": result["W"].tolist()
            }
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
