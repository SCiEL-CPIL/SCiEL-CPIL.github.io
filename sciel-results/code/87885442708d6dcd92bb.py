"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 42
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    def verify(U, V, W, r):
        """Check if decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0.0
                        for c in range(r):
                            val += W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(val - expected) > 1e-8:
                            return False
        return True
    
    def try_rank(r, max_trials=300):
        """Try to find a valid decomposition of rank r."""
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            # Use different random seeds for diversity
            seed = 42 + r * 10000 + trial
            np.random.seed(seed)
            
            # Use half-integer values: multiples of 0.5
            # Try different ranges for better coverage
            if trial % 3 == 0:
                U = 0.5 * np.random.randint(-4, 5, size=(nm, r))
                V = 0.5 * np.random.randint(-4, 5, size=(mp, r))
            elif trial % 3 == 1:
                U = 0.5 * np.random.randint(-6, 7, size=(nm, r))
                V = 0.5 * np.random.randint(-6, 7, size=(mp, r))
            else:
                U = 0.5 * np.random.randint(-8, 9, size=(nm, r))
                V = 0.5 * np.random.randint(-8, 9, size=(mp, r))
            
            # For each (i,k) pair, solve the linear system
            W = np.zeros((pn, r))
            solvable = True
            
            for i in range(n):
                for k in range(p):
                    # Build m*m x r matrix B where B[j*m+jp, c] = U[i*m+j, c] * V[jp*p+k, c]
                    B = np.zeros((m*m, r))
                    for j in range(m):
                        for jp in range(m):
                            idx = j * m + jp
                            B[idx, :] = U[i*m+j, :] * V[jp*p+k, :]
                    
                    # Target: vec(I_m) - identity matrix vectorized
                    target = np.zeros(m*m)
                    for j in range(m):
                        target[j*m + j] = 1.0
                    
                    # Check if system is solvable
                    # Use least squares and check residual
                    w, res, rank_B, sv = np.linalg.lstsq(B, target, rcond=1e-10)
                    
                    # Verify the solution
                    if np.linalg.norm(B @ w - target) > 1e-8:
                        solvable = False
                        break
                    
                    W[k*n+i, :] = w
                
                if not solvable:
                    break
            
            if solvable:
                # Final verification with exact check
                if verify(U, V, W, r):
                    return U, V, W
        
        return None
    
    # Try ranks from lower bound up
    # Lower bound is max(nm, mp, pn) = max(6, 21, 14) = 21
    lower_bound = max(nm, mp, pn)
    
    for r in range(lower_bound, rank_school):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
