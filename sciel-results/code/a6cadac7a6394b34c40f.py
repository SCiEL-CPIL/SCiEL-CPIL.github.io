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
        """Exact verification using Fraction arithmetic, vectorized where possible."""
        # Convert to Fraction arrays for exactness
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        
        # To keep it fast, we can check one (i,j,k) at a time but use lists
        # Or, since rank is small (~30), the loop is fast enough.
        # The bottleneck in the original was likely the ALS, not verification.
        # But let's make it slightly more efficient by local variable binding.
        U_flat = U
        V_flat = V
        W_flat = W
        n_val = n
        m_val = m
        p_val = p
        rank_val = rank
        
        for i in range(n_val):
            for j in range(m_val):
                u_row = U_flat[i*m_val+j]
                for k in range(p_val):
                    v_row = V_flat[j*p_val+k]
                    w_row = W_flat[k*n_val+i]
                    s = Fraction(0)
                    for c in range(rank_val):
                        s += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if s != Fraction(1):
                        return False
        return True
    
    # Strategy: Use ALS with exact rational polishing
    # 1. Run ALS in float to get near-solution
    # 2. Snap to half-integers
    # 3. Verify exactly
    # 4. If fails, try local search in the neighborhood
    
    def als_optimize(U_init, V_init, W_init, rank, max_iter=50):
        """Alternating least squares in float."""
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build system: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # For fixed i,j: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    # This is: A * x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], x = U[i*m+j,:], b = ones(p)
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
        
        return U, V, W
    
    def snap_to_half_int(U, V, W, rank):
        """Snap values to nearest half-integer."""
        U_s = [[round(U[i][c] * 2) / 2 for c in range(rank)] for i in range(n*m)]
        V_s = [[round(V[i][c] * 2) / 2 for c in range(rank)] for i in range(m*p)]
        W_s = [[round(W[i][c] * 2) / 2 for c in range(rank)] for i in range(p*n)]
        return U_s, V_s, W_s
    
    # Try ranks from 35 down
    # Given the difficulty of finding rank < 36 for 3x3x4, we limit the search time significantly
    # to ensure we return the valid schoolbook solution if no better one is found quickly.
    for target_rank in range(35, 25, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Reduce seeds to keep time low, as ALS is expensive
        num_seeds = 4
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            random.seed(seed * 1000 + target_rank * 13)
            np.random.seed(seed * 1000 + target_rank * 13)
            
            # Initialize near schoolbook
            if target_rank >= 32:
                U = [row[:target_rank][:] for row in U_sb]
                V = [row[:target_rank][:] for row in V_sb]
                W = [row[:target_rank][:] for row in W_sb]
                # Add small perturbation
                for i in range(n*m):
                    for c in range(target_rank):
                        U[i][c] += random.uniform(-0.1, 0.1)
                for i in range(m*p):
                    for c in range(target_rank):
                        V[i][c] += random.uniform(-0.1, 0.1)
                for i in range(p*n):
                    for c in range(target_rank):
                        W[i][c] += random.uniform(-0.1, 0.1)
            else:
                U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
                V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
                W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Run ALS with fewer iterations to save time
            U_opt, V_opt, W_opt = als_optimize(U, V, W, target_rank, max_iter=20)
            
            # Snap to half-integers
            U_snap, V_snap, W_snap = snap_to_half_int(U_opt, V_opt, W_opt, target_rank)
            
            # Verify exactly
            if verify_exact(U_snap, V_snap, W_snap, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                found = True
                break
        
        if found:
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
