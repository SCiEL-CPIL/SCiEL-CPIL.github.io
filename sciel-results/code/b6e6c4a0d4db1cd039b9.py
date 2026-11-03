import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-42 Strassen-based decomposition
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for g in range(2):
        for h in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = g * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = g * 2 + j_loc
                    k = h * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = h * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    # Try to reduce rank by checking for linear dependencies
    # Convert to numpy arrays for linear algebra
    U_np = np.array(U, dtype=float)  # shape (n*m, rank)
    V_np = np.array(V, dtype=float)  # shape (m*p, rank)
    W_np = np.array(W, dtype=float)  # shape (p*n, rank)
    
    # The decomposition is valid if for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (i,j) in A * (j,k) in B
    # We check if any column is a linear combination of others
    # by checking if removing it still satisfies the tensor equation
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Check if we can remove each column
    for col_to_remove in range(rank):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        # Skip this column, check if the remaining rank-41 decomposition is valid
        U_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
        V_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
        W_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
        
        # Verify the decomposition
        U_test_np = np.array(U_test, dtype=float)
        V_test_np = np.array(V_test, dtype=float)
        W_test_np = np.array(W_test, dtype=float)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected: 1 if this is a valid (i,j,k) entry, 0 otherwise
                    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in the multiplication
                    # The decomposition should satisfy:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Wait, that's not right. The matrix multiplication tensor is:
                    # T[i,j,k] = 1 for the product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
                    # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    
                    lhs = 0.0
                    for r in range(rank - 1):
                        lhs += U_test_np[i * m + j, r] * V_test_np[j * p + k, r] * W_test_np[k * n + i, r]
                    
                    if abs(lhs - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = rank - 1
            # Update U, V, W for subsequent checks
            U, V, W = U_test, V_test, W_test
            rank = best_rank
            break  # Found one, can try to find more but keep it simple
    
    return {"U": best_U, "V": best_V, "W": best_W}
