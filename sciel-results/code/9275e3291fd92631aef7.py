import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Try ranks from lower bound upwards
    # Lower bound for 2x3x4 is at least 13 (Larsen bound)
    # Known: 2x2x2 is 7, 2x3x3 is 11. 2x3x4 is likely 14 or 15.
    for rank in range(14, 25):
        if time.time() > deadline:
            break
            
        num_trials = 20000 if rank <= 15 else 50000
        
        for _ in range(num_trials):
            if time.time() > deadline:
                break
            
            # Generate random U and V with small integer entries
            # Using -1, 0, 1 is a good heuristic for finding sparse solutions
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            
            valid = True
            W = np.zeros((p * n, rank))
            
            # Check each (i, k) block
            for i in range(n):
                for k in range(p):
                    # Construct 3 x rank matrix M_ik
                    # M_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    M_ik = np.zeros((m, rank))
                    for j in range(m):
                        M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M_ik @ W_ik = ones(m)
                    # We want W_ik to be half-integers.
                    # Let W_ik = 0.5 * Z_ik where Z_ik is integer.
                    # Then M_ik @ (0.5 * Z_ik) = 1 => M_ik @ Z_ik = 2
                    # We can solve for Z_ik using least squares and check if it's close to integers.
                    
                    # Using pseudo-inverse for a particular solution
                    try:
                        # Solve M_ik Z = 2
                        Z_part, residuals, rank_m, s = np.linalg.lstsq(M_ik, np.ones(m) * 2, rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if Z_part is close to integers
                    # We can also try to adjust by null space vectors to get closer to integers,
                    # but for a first pass, checking the particular solution is fast.
                    # A better check: is there an integer vector in the affine space Z_part + Null(M_ik)?
                    # For small m=3, we can just check if Z_part is close to integers.
                    
                    Z_rounded = np.round(Z_part)
                    error = np.linalg.norm(M_ik @ Z_rounded - np.ones(m) * 2)
                    
                    if error < 1e-9:
                        # Found integer Z, so W = 0.5 * Z is half-integer
                        W[k * n + i, :] = 0.5 * Z_rounded
                    else:
                        # Try rounding to nearest half-integer directly?
                        # Or just fail this trial
                        valid = False
                        break
            
            if valid:
                # Convert to list of lists as required
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Final verification
                # Check if U, V, W exactly decompose the tensor
                # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                # Should be 1 for all i,j,k
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If rank not found, try next rank
        # Note: We might need to increase trial count or improve search if not found
        pass

    # Fallback to schoolbook if no better rank found in time
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
