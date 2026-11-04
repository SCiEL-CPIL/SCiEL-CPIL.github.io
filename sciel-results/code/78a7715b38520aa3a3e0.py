import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 20s buffer
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound is roughly n*m*p / (n+p) ? No.
    # Let's try from 10 to 56
    # We can also try to start with a known good rank if we had one, but we don't.
    
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook solution as fallback
    def get_schoolbook():
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
    
    # Try to find a better rank
    for r in range(10, n * m * p + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        max_restarts = 50 if r < 30 else 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random U, V with small integers
            # Using {-1, 0, 1}
            U_flat = np.random.randint(-1, 2, size=(n * m, r))
            V_flat = np.random.randint(-1, 2, size=(m * p, r))
            
            # Solve for W
            W_flat = np.zeros((p * n, r))
            valid_w = True
            
            for k in range(p):
                for ip in range(n):
                    # Construct A (n*m) x r
                    A = np.zeros((n * m, r))
                    b = np.zeros(n * m)
                    
                    for i in range(n):
                        for j in range(m):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            # Hadamard product of U[idx_u] and V[idx_v]
                            A[i * m + j, :] = U_flat[idx_u, :] * V_flat[idx_v, :]
                            b[i * m + j] = 1.0 if i == ip else 0.0
                    
                    # Solve A w = b
                    # w is the row of W for index k*n + ip
                    # We use lstsq
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_flat[k * n + ip, :] = w
                    
            # Round W to nearest half-integer
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Check if rounding is close
            if not np.allclose(W_flat, W_rounded, atol=1e-6):
                continue
                
            # Verify decomposition exactly
            # Sum over c of U[i,j,c] V[j,k,c] W[k,i',c] == delta(i, i')
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ip in range(n):
                            sum_val = 0.0
                            for c in range(r):
                                sum_val += U_flat[i * m + j, c] * V_flat[j * p + k, c] * W_rounded[k * n + ip, c]
                            expected = 1.0 if i == ip else 0.0
                            if not np.isclose(sum_val, expected, atol=1e-9):
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            if valid:
                # Found a valid decomposition
                # Convert to list of lists
                U_list = U_flat.tolist()
                V_list = V_flat.tolist()
                W_list = W_rounded.tolist()
                
                # Ensure integers are ints
                for i in range(len(U_list)):
                    for j in range(len(U_list[i])):
                        if U_list[i][j] == int(U_list[i][j]):
                            U_list[i][j] = int(U_list[i][j])
                        elif U_list[i][j] * 2 == int(U_list[i][j] * 2):
                            U_list[i][j] = U_list[i][j] # keep as float or convert?
                            # Task says integer/half-integer. Floats like 0.5 are fine.
                            # But if it's 1.0, int is cleaner.
                            pass
                # Same for V, W
                
                return {"U": U_list, "V": V_list, "W": W_list}
                
        # If no restart worked for this r, try next r
        
    # If loop finishes without finding, return schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
