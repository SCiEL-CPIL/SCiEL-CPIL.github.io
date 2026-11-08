"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_lists(U, V, W, rank):
        return {
            'U': [[float(U[i, r]) for r in range(rank)] for i in range(len(U))],
            'V': [[float(V[i, r]) for r in range(rank)] for i in range(len(V))],
            'W': [[float(W[i, r]) for r in range(rank)] for i in range(len(W))]
        }
    
    def verify(U, V, W, rank, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Best solution
    best_rank = rank_sb
    best_result = to_lists(U_sb, V_sb, W_sb, rank_sb)
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    for rank in range(24, rank_sb + 1):
        if time.time() > DEADLINE:
            break
        
        found = False
        # More attempts for smaller ranks
        num_attempts = 100 if rank <= 30 else 50 if rank <= 40 else 20
        
        for attempt in range(num_attempts):
            if time.time() > DEADLINE:
                break
            
            # Random half-integer U and V
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, rank))
            
            # Build linear system for W
            num_W_vars = p * n * rank
            A_matrix = np.zeros((n*m*p, num_W_vars))
            b_vec = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row = k * n + i
                        for r in range(rank):
                            coeff = U[i*m+j, r] * V[j*p+k, r]
                            if coeff != 0:
                                col_idx = w_row * rank + r
                                A_matrix[eq_idx, col_idx] = coeff
                        eq_idx += 1
            
            # Solve using least squares
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_matrix, b_vec, rcond=None)
                W = W_flat.reshape(p*n, rank)
            except:
                continue
            
            # Check if W is close to half-integers
            W_half = np.round(W * 2.0) / 2.0
            if np.allclose(W, W_half, atol=1e-3):
                # Verify
                if verify(U, V, W_half, rank):
                    best_rank = rank
                    best_result = to_lists(U, V, W_half, rank)
                    found = True
                    break
        
        if found:
            break
        if time.time() > DEADLINE:
            break
    
    return best_result

# EVOLVE-BLOCK-END
