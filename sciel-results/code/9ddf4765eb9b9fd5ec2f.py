"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, R):
        """Verify exact decomposition with tolerance for floating point."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    best_rank = n * m * p
    best_result = None
    
    # Search from low ranks upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But for multiplication tensors, bounds are higher. Try from 25 upward.
    for R in range(25, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Multiple random attempts per rank
        max_attempts = 30 if R < 40 else 15
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V with entries in {-1, 0, 1}
            # Using a mix of sparse and dense patterns
            np.random.seed(attempt * 1000 + R * 7)
            
            # Try different sparsity levels
            if attempt % 3 == 0:
                # Dense: all entries in {-1, 0, 1}
                U = np.random.randint(-1, 2, size=(n*m, R))
                V = np.random.randint(-1, 2, size=(m*p, R))
            elif attempt % 3 == 1:
                # Sparser
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                for idx in range(n*m):
                    for r in range(R):
                        if np.random.random() < 0.5:
                            U[idx, r] = np.random.choice([-1, 1])
                for idx in range(m*p):
                    for r in range(R):
                        if np.random.random() < 0.5:
                            V[idx, r] = np.random.choice([-1, 1])
            else:
                # Mixed: some rows are basis-like
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                for idx in range(n*m):
                    r = np.random.randint(0, R)
                    U[idx, r] = np.random.choice([-1, 1])
                for idx in range(m*p):
                    r = np.random.randint(0, R)
                    V[idx, r] = np.random.choice([-1, 1])
            
            # Solve for W: for each (i, k), solve B[i,k] w_i = ones(m)
            W = np.zeros((p*n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build B matrix: m x R, B[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    B = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            B[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    # Solve B w = ones(m)
                    try:
                        # Use lstsq for overdetermined or square systems
                        w, residuals, rank, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        # Check if solution is exact
                        if np.allclose(B @ w, np.ones(m), atol=1e-8):
                            W[k*n+i, :] = w
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Snap to half-integers
            W_snapped = np.round(W * 2) / 2.0
            U_snapped = np.round(U * 2) / 2.0
            V_snapped = np.round(V * 2) / 2.0
            
            # Verify with snapped values
            if verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), R):
                best_rank = R
                best_result = {
                    "U": U_snapped.tolist(),
                    "V": V_snapped.tolist(),
                    "W": W_snapped.tolist()
                }
                break  # Move to next rank
        
        if best_result is not None and R <= 35:
            break  # Found a good rank, stop searching
    
    if best_result is None:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
