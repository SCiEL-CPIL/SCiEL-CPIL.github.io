"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matrix multiplication tensor.
    
    Tensor T[i,j,k] = 1 if i,j,k valid, else 0.
    Actually T[i*m+j, j*p+k, k*n+i] structure:
    For each term r: sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for valid (i,j,k).
    """
    rank = len(U[0]) if U else 0
    # Build the target tensor implicitly
    # For each (i, j, k) in valid range, check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find a better decomposition via random initialization + local search
    # Strategy: start with random rank-r matrices and use gradient-free local search
    # to minimize the error, then try to reduce rank by zeroing out columns.
    
    target_rank = 46  # Aim for lower than 54
    
    def random_init(rank, scale=1.0):
        U = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    def error(U, V, W, rank):
        """Compute max absolute error over all valid (i,j,k)."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def local_search(U, V, W, rank, steps=200, lr=0.1):
        """Simple local search: perturb random entries, keep if error decreases."""
        curr_err = error(U, V, W, rank)
        U_curr = [row[:] for row in U]
        V_curr = [row[:] for row in V]
        W_curr = [row[:] for row in W]
        
        for step in range(steps):
            if time.time() > deadline:
                break
            # Pick a random matrix, row, and column to perturb
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat, size = U_curr, n*m
            elif mat_idx == 1:
                mat, size = V_curr, m*p
            else:
                mat, size = W_curr, p*n
            
            r_idx = random.randrange(size)
            c_idx = random.randrange(rank)
            
            # Perturb
            old_val = mat[r_idx][c_idx]
            new_val = old_val + random.gauss(0, lr)
            mat[r_idx][c_idx] = new_val
            
            # Compute new error
            if mat_idx == 0:
                new_err = error(U_curr, V_curr, W_curr, rank)
            elif mat_idx == 1:
                new_err = error(U_curr, V_curr, W_curr, rank)
            else:
                new_err = error(U_curr, V_curr, W_curr, rank)
            
            if new_err < curr_err:
                curr_err = new_err
            else:
                # Revert
                mat[r_idx][c_idx] = old_val
                # Decrease learning rate occasionally
                lr = max(0.01, lr * 0.95)
        
        if mat_idx == 0:
            U_curr = [row[:] for row in U_curr]
            V_curr = [row[:] for row in V_curr]
            W_curr = [row[:] for row in W_curr]
        
        return U_curr, V_curr, W_curr, curr_err
    
    # Try multiple random restarts with rank starting from target_rank
    random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        
        # Start with a slightly higher rank for easier convergence
        start_rank = target_rank + random.choice([0, 2, 4])
        
        U, V, W = random_init(start_rank, scale=1.0)
        
        # Initial local search
        U, V, W, curr_err = local_search(U, V, W, start_rank, steps=100, lr=0.5)
        
        if curr_err < 1e-7:
            # Try to reduce rank by zeroing out columns
            for r in range(start_rank, target_rank - 1, -1):
                if time.time() > deadline - 5:
                    break
                # Try zeroing column r
                saved = []
                for i in range(n*m):
                    saved.append(U[i][r])
                    U[i][r] = 0
                for j in range(m*p):
                    saved.append(V[j][r])
                    V[j][r] = 0
                for k in range(p*n):
                    saved.append(W[k][r])
                    W[k][r] = 0
                
                new_err = error(U, V, W, r)
                if new_err < 1e-7:
                    # Keep it reduced
                    curr_err = new_err
                else:
                    # Revert
                    idx = 0
                    for i in range(n*m):
                        U[i][r] = saved[idx]; idx += 1
                    for j in range(m*p):
                        V[j][r] = saved[idx]; idx += 1
                    for k in range(p*n):
                        W[k][r] = saved[idx]; idx += 1
                    # Local search again to fix
                    U, V, W, curr_err = local_search(U, V, W, start_rank, steps=50, lr=0.2)
                    if curr_err < 1e-7:
                        continue
                    else:
                        break
        
        if curr_err < 1e-9 and start_rank < best_rank:
            # Verify and update best
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W, best_rank = U, V, W, start_rank
                if best_rank <= target_rank:
                    break
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
