"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: find min rank r such that there exist U(8,r), V(20,r), W(10,r)
    # with half-integer entries satisfying:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor T[i,j,k] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def to_halfint_matrix(mat):
        """Round to nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def residual(U, V, W):
        """Compute sum of squared residuals."""
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res += (val - 1.0) ** 2
        return res
    
    # Try to find lower rank decompositions
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with half-integer values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(n*m, target_rank))
            V = np.random.choice(vals, size=(m*p, target_rank))
            W = np.random.choice(vals, size=(p*n, target_rank))
            
            # Quick check
            if check_exact(U, V, W):
                found = True
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
            
            # Gradient-free local search: perturb random entries
            best_res = residual(U, V, W)
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random matrix and entry to perturb
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    mat = U
                    rows, cols = n*m, target_rank
                elif mat_choice == 'V':
                    mat = V
                    rows, cols = m*p, target_rank
                else:
                    mat = W
                    rows, cols = p*n, target_rank
                
                row = random.randint(0, rows-1)
                col = random.randint(0, cols-1)
                old_val = mat[row, col]
                # Try adding or subtracting 0.5
                delta = random.choice([-0.5, 0.5])
                new_val = old_val + delta
                if new_val < -2 or new_val > 2:
                    continue
                mat[row, col] = new_val
                
                new_res = residual(U, V, W)
                if new_res < best_res - 1e-12:
                    best_res = new_res
                    # Check if exact
                    if check_exact(U, V, W):
                        found = True
                        best_U, best_V, best_W, best_rank = U, V, W, target_rank
                        break
                else:
                    mat[row, col] = old_val
            
            if found:
                break
    
    if found:
        break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
