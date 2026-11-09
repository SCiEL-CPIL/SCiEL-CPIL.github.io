"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook solution as fallback
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_exact(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != Fraction(1):
                        return False
        return True
    
    def compute_residual(U, V, W, rank):
        """Compute max absolute residual."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    res = abs(s - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res
    
    def als_optimize(U_init, V_init, W_init, rank, max_iter=80):
        """Alternating least squares in float with convergence check."""
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        
        prev_res = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    b = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j] = sol.tolist()
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    b = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k] = sol.tolist()
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    b = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i] = sol.tolist()
                    except:
                        pass
            
            # Check convergence
            res = compute_residual(U, V, W, rank)
            if res < 1e-10:
                break
            if abs(prev_res - res) < 1e-14:
                break
            prev_res = res
        
        return U, V, W
    
    def snap_to_half_int(U, V, W, rank):
        """Snap values to nearest half-integer."""
        U_s = [[round(U[i][c] * 2) / 2 for c in range(rank)] for i in range(n*m)]
        V_s = [[round(V[i][c] * 2) / 2 for c in range(rank)] for i in range(m*p)]
        W_s = [[round(W[i][c] * 2) / 2 for c in range(rank)] for i in range(p*n)]
        return U_s, V_s, W_s
    
    def local_search(U, V, W, rank, max_iter=200):
        """Local search: try flipping individual entries to improve residual."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        current_res = compute_residual(U, V, W, rank)
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            if current_res < 1e-15:
                break
            
            improved = False
            # Try modifying each entry in U, V, W
            for matrix_name in ['U', 'V', 'W']:
                if matrix_name == 'U':
                    M, rows = U, n*m
                elif matrix_name == 'V':
                    M, rows = V, m*p
                else:
                    M, rows = W, p*n
                
                for r in range(rows):
                    for c in range(rank):
                        old_val = M[r][c]
                        # Try nearby half-integer values
                        base = round(old_val * 2) / 2
                        for delta in [-1.0, -0.5, 0.5, 1.0]:
                            new_val = base + delta
                            if new_val == old_val:
                                continue
                            M[r][c] = new_val
                            new_res = compute_residual(U, V, W, rank)
                            if new_res < current_res:
                                current_res = new_res
                                improved = True
                            else:
                                M[r][c] = old_val
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return U, V, W, current_res
    
    def try_rank(target_rank, num_attempts=15):
        """Try to find a valid decomposition at given rank."""
        for attempt in range(num_attempts):
            if time.time() > deadline - 25:
                return None
            
            random.seed(attempt * 7919 + target_rank * 104729)
            np.random.seed(attempt * 7919 + target_rank * 104729)
            
            # Different initialization strategies
            if attempt % 3 == 0 and target_rank >= 30:
                # Near schoolbook with perturbation
                U = [row[:target_rank][:] for row in U_sb]
                V = [row[:target_rank][:] for row in V_sb]
                W = [row[:target_rank][:] for row in W_sb]
                for i in range(n*m):
                    for c in range(target_rank):
                        U[i][c] += random.uniform(-0.5, 0.5)
                for i in range(m*p):
                    for c in range(target_rank):
                        V[i][c] += random.uniform(-0.5, 0.5)
                for i in range(p*n):
                    for c in range(target_rank):
                        W[i][c] += random.uniform(-0.5, 0.5)
            elif attempt % 3 == 1:
                # Random initialization with moderate scale
                U = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
                V = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
                W = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            else:
                # Sparse initialization: most entries 0, some ±1
                U = [[0] * target_rank for _ in range(n*m)]
                V = [[0] * target_rank for _ in range(m*p)]
                W = [[0] * target_rank for _ in range(p*n)]
                for i in range(n*m):
                    num_nonzero = random.randint(1, min(3, target_rank))
                    cols = random.sample(range(target_rank), num_nonzero)
                    for c in cols:
                        U[i][c] = random.choice([-1, 1])
                for i in range(m*p):
                    num_nonzero = random.randint(1, min(3, target_rank))
                    cols = random.sample(range(target_rank), num_nonzero)
                    for c in cols:
                        V[i][c] = random.choice([-1, 1])
                for i in range(p*n):
                    num_nonzero = random.randint(1, min(3, target_rank))
                    cols = random.sample(range(target_rank), num_nonzero)
                    for c in cols:
                        W[i][c] = random.choice([-1, 1])
            
            # Run ALS
            U_opt, V_opt, W_opt = als_optimize(U, V, W, target_rank, max_iter=60)
            
            # Snap to half-integers
            U_snap, V_snap, W_snap = snap_to_half_int(U_opt, V_opt, W_opt, target_rank)
            
            # Local search to improve
            U_ls, V_ls, W_ls, res = local_search(U_snap, V_snap, W_snap, target_rank, max_iter=100)
            
            # Verify exactly
            if verify_exact(U_ls, V_ls, W_ls, target_rank):
                return U_ls, V_ls, W_ls
        
        return None
    
    # Try ranks from 35 down to 23
    for target_rank in range(35, 22, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    # Convert to output format (int or float for half-integers)
    def convert_output(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float):
                    if val == int(val):
                        new_row.append(int(val))
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {
        "U": convert_output(best_U),
        "V": convert_output(best_V),
        "W": convert_output(best_W)
    }
# EVOLVE-BLOCK-END
