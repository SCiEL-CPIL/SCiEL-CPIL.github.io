import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Theoretical lower bound
    # R(n, m, p) >= ceil(n*m*p / 2) is a common heuristic bound for small matrices
    # Actually, for n=2, m=5, p=6, the rank is likely 30 or slightly higher.
    # Let's start from 30.
    min_rank = 30
    max_rank = n * m * p  # 60
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for U and V
        # We want U and V to be "generic" enough to allow a solution W
        # but structured enough to yield half-integer W.
        # Using entries from {-1, 1} is a good start.
        
        num_trials = 500 if r < 40 else 200
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries -1, 1
            U = np.random.choice([-1, 1], size=(n * m, r))
            V = np.random.choice([-1, 1], size=(m * p, r))
            
            W = np.zeros((p * n, r))
            valid = True
            
            # Solve for W for each (i, k) pair
            for i in range(n):
                for k in range(p):
                    # Construct M_ik of shape (m, r)
                    # M_ik[j, alpha] = U[i*m + j, alpha] * V[j*p + k, alpha]
                    U_part = U[i*m : (i+1)*m, :]
                    V_part = V[k : p*m : p, :] # Wait, V is indexed by j*p+k.
                    # V row index is j*p + k.
                    # For fixed k, we need rows j*p + k for j=0..m-1.
                    # These are not contiguous if p > 1?
                    # V is (m*p, r).
                    # Row j*p + k.
                    # Indices: k, p+k, 2p+k, ..., (m-1)p+k.
                    V_rows = [j * p + k for j in range(m)]
                    V_part = V[V_rows, :]
                    
                    M_ik = U_part * V_part
                    # Solve M_ik @ w = ones(m)
                    # Use least squares
                    try:
                        w, residuals, rank_m, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w is close to half-integers
                    # 2*w should be close to integers
                    two_w = 2 * w
                    rounded_two_w = np.round(two_w)
                    if not np.allclose(two_w, rounded_two_w, atol=1e-8):
                        valid = False
                        break
                    
                    W[k * n + i, :] = w
            
            if valid:
                # Convert to list of lists of floats (or ints if integer)
                # The prompt accepts integer/half-integer.
                # We can return the numpy arrays or convert to lists.
                # The seed returns lists.
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
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
