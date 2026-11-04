import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Precompute indices
    # T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    valid_indices = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                valid_set.add((alpha, beta, gamma))
                
    # Group indices by gamma for efficient linear system solving
    # For each gamma, we need to solve for W[gamma, :]
    # Constraints: sum_c U[alpha, c] * V[beta, c] * W[gamma, c] = T[alpha, beta, gamma]
    # Let A_gamma be a matrix where rows correspond to (alpha, beta) pairs for this gamma
    # and columns correspond to c in 0..r-1.
    # A_gamma[row, c] = U[alpha, c] * V[beta, c]
    # b_gamma[row] = T[alpha, beta, gamma]
    
    gamma_to_indices = {g: [] for g in range(p * n)}
    for (a, b, g) in valid_indices:
        gamma_to_indices[g].append((a, b, 1.0))
        
    # Also need invalid indices to enforce zero
    # Total indices
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                all_indices.append((a, b, g))
                
    invalid_indices = [idx for idx in all_indices if idx not in valid_set]
    for (a, b, g) in invalid_indices:
        gamma_to_indices[g].append((a, b, 0.0))

    start_time = time.time()
    timeout = 110.0
    
    # Try ranks from lower bound to schoolbook
    # Lower bound estimate: 20
    for r in range(20, 36):
        if time.time() - start_time > timeout:
            break
            
        # Number of attempts increases with r if time permits, but keep fixed for speed
        max_attempts = 500 if r < 25 else 200
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization for U and V
            # Use sparse random vectors from {-1, 0, 1}
            # Density ~ 0.3 to ensure rank properties
            U = np.random.choice([-1, 0, 1], size=(n * m, r), p=[0.15, 0.7, 0.15])
            V = np.random.choice([-1, 0, 1], size=(m * p, r), p=[0.15, 0.7, 0.15])
            
            # Solve for W
            W = np.zeros((p * n, r))
            valid_solution = True
            
            for g in range(p * n):
                indices = gamma_to_indices[g]
                if not indices:
                    continue
                    
                # Build A_g and b_g
                # Number of constraints for this g
                num_constraints = len(indices)
                A_g = np.zeros((num_constraints, r))
                b_g = np.zeros(num_constraints)
                
                for idx, (a, b, target) in enumerate(indices):
                    # Row of A_g is U[a, :] * V[b, :]
                    A_g[idx, :] = U[a, :] * V[b, :]
                    b_g[idx] = target
                    
                # Solve A_g * w_g = b_g
                # Use least squares
                try:
                    w_g, residuals, rank, sv = np.linalg.lstsq(A_g, b_g, rcond=None)
                except Exception:
                    valid_solution = False
                    break
                
                # Check residual
                if residuals is None or len(residuals) == 0 or residuals[0] > 1e-9:
                    valid_solution = False
                    break
                    
                W[g, :] = w_g
                
            if not valid_solution:
                continue
                
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            # Verify exactness
            # T_approx = np.einsum('ac,bc,dc->abd', U_s, V_s, W_s)
            # This might be slow if done naively in Python loops, but 3x3x4 is small.
            # Let's check specific indices.
            
            is_valid = True
            for (a, b, g) in valid_indices:
                val = np.dot(U_s[a, :], V_s[b, :] * W_s[g, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    is_valid = False
                    break
            
            if is_valid:
                for (a, b, g) in invalid_indices:
                    val = np.dot(U_s[a, :], V_s[b, :] * W_s[g, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        is_valid = False
                        break
                        
            if is_valid:
                return {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }

    # Fallback to schoolbook
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
