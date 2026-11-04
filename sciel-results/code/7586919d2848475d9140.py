import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s buffer for safety
    
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
    
    # Try structured search: use small integer/half-integer values
    # Focus on finding rank < 56 by systematic search
    
    def verify(U, V, W, rank):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        total = 0.0
                        for c in range(rank):
                            total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + ip][c]
                        expected = 1.0 if i == ip else 0.0
                        if abs(total - expected) > 1e-10:
                            return False
        return True
    
    # Strategy: Try ranks from lower bound up, using structured random search
    # with small integer values and exact verification
    
    # Lower bound estimate: at least n*p = 14 (since output is n×p)
    # Upper bound: schoolbook = 56
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Try structured search for ranks from 14 to min(40, schoolbook-1)
    for r in range(14, min(41, n * m * p)):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts with small integer values
        max_restarts = 200 if r < 25 else 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use small integer values {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            # For each row of W, solve the linear system
            W = np.zeros((p * n, r))
            valid_w = True
            
            for k in range(p):
                for ip in range(n):
                    # Build system: sum_c U[i,j,c] * V[j,k,c] * W[k*n+ip,c] = delta(i,ip)
                    # For fixed k, ip: unknowns are W[k*n+ip, :]
                    # Equations for each (i, j): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+ip,c] = 1 if i==ip else 0
                    
                    A = np.zeros((n * m, r))
                    b = np.zeros(n * m)
                    
                    for i in range(n):
                        for j in range(m):
                            A[i * m + j, :] = U[i * m + j, :] * V[j * p + k, :]
                            b[i * m + j] = 1.0 if i == ip else 0.0
                    
                    # Solve A * w = b, where w = W[k*n+ip, :]
                    # Use least squares, then check if solution is close to half-integers
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=1e-12)
                    W[k * n + ip, :] = w
            
            # Round W to nearest half-integer
            W_rounded = np.round(W * 2) / 2.0
            
            # Check if rounding is accurate
            if not np.allclose(W, W_rounded, atol=1e-6):
                continue
            
            # Verify exact decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_rounded.tolist()
            
            if verify(U_list, V_list, W_list, r):
                # Convert to clean format
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for idx, val in enumerate(row):
                            if abs(val - round(val)) < 1e-10:
                                row[idx] = int(round(val))
                            elif abs(val * 2 - round(val * 2)) < 1e-10:
                                row[idx] = val  # keep as float for half-integers
                            else:
                                # Should not happen if allclose passed
                                row[idx] = round(val * 2) / 2.0
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no restart worked for this r, continue to next r
    
    # If we get here, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
