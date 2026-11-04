import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 110  # Safety margin
    
    # Indices for the 27 non-zero entries of the tensor T
    eq_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eqs = len(eq_indices)
    
    best_rank = 27
    best_U, best_V, best_W = None, None, None
    
    # Seed with naive solution
    U_naive = np.zeros((n*m, 27), dtype=int)
    V_naive = np.zeros((m*p, 27), dtype=int)
    W_naive = np.zeros((p*n, 27), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 27
    
    def build_M(U, V, rank):
        """Vectorized construction of the linear system matrix M"""
        M = np.zeros((num_eqs, p * n * rank))
        for e_idx, (i, j, k) in enumerate(eq_indices):
            w_idx_base = k * n + i
            # For each component c, coefficient is U[i*m+j, c] * V[j*p+k, c]
            coeffs = U[i*m+j, :] * V[j*p+k, :]  # shape (rank,)
            # Place each coeff at position c*p*n + w_idx_base
            for c in range(rank):
                if coeffs[c] != 0:
                    M[e_idx, c * p * n + w_idx_base] = coeffs[c]
        return M
    
    def try_solve(U, V, rank, tight_tol=1e-6, loose_tol=1e-3):
        """Try to solve for W given U and V. Returns (W, success) or (None, False)"""
        M = build_M(U, V, rank)
        
        try:
            # Solve M * Y = 2*ones where Y = 2*W
            Y_vec, residuals, rank_m, s = np.linalg.lstsq(M, 2 * np.ones(num_eqs), rcond=None)
            
            # Check if Y_vec is close to integers
            Y_rounded = np.round(Y_vec)
            error = np.max(np.abs(Y_vec - Y_rounded))
            
            if error < tight_tol:
                W_solution = Y_rounded / 2.0
                # Verify
                valid = True
                for e_idx, (i, j, k) in enumerate(eq_indices):
                    w_idx = k * n + i
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W_solution[:, w_idx])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if valid:
                    return W_solution, True
                else:
                    return None, False
            elif error < loose_tol:
                # Close but not quite - try rounding and verifying
                W_solution = Y_rounded / 2.0
                valid = True
                for e_idx, (i, j, k) in enumerate(eq_indices):
                    w_idx = k * n + i
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W_solution[:, w_idx])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if valid:
                    return W_solution, True
                else:
                    return None, False
            else:
                return None, False
        except Exception:
            return None, False
    
    def random_UV(rank, sparsity=0.5, use_structure=False):
        """Generate random U and V matrices"""
        if use_structure:
            # Use structured patterns more likely to work
            # Try patterns based on known decompositions
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            
            # Randomly assign entries with bias toward certain patterns
            for c in range(rank):
                # Each column gets a random subset of rows
                u_mask = np.random.random(n*m) < sparsity
                v_mask = np.random.random(m*p) < sparsity
                U[:, c] = np.where(u_mask, np.random.choice([-1, 1], n*m), 0)
                V[:, c] = np.where(v_mask, np.random.choice([-1, 1], m*p), 0)
        else:
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            for c in range(rank):
                u_mask = np.random.random(n*m) < sparsity
                v_mask = np.random.random(m*p) < sparsity
                U[:, c] = np.where(u_mask, np.random.choice([-1, 1], n*m), 0)
                V[:, c] = np.where(v_mask, np.random.choice([-1, 1], m*p), 0)
        
        return U, V
    
    # Search for lower rank, starting from 23 (known minimum)
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        # Multiple phases with different strategies
        phases = [
            (500, 0.3, False),   # Sparse, unstructured
            (500, 0.5, False),   # Medium sparsity
            (300, 0.7, True),    # Denser, structured
            (200, 1.0, True),    # Dense, structured
        ]
        
        for num_trials, sparsity, use_struct in phases:
            if time.time() > deadline:
                break
            
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                U, V = random_UV(rank, sparsity, use_struct)
                
                W_sol, success = try_solve(U, V, rank)
                
                if success:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    break
            
            if best_rank == rank:
                break
        
        if best_rank == rank:
            break
    
    # Convert to lists for return
    def to_list(mat):
        return mat.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
