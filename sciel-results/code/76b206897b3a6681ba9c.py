"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # The matrix multiplication tensor T[i,j,k] where:
    # i indexes A (n*m = 9), j indexes B (m*p = 18), k indexes C (p*n = 18)
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[k]
    # Specifically: A row i//m, col i%m; B row j//p, col j%p
    # C row j%p, col i//m (using W[k*n+i] convention where k = j%p, i = i//m)
    # Wait, let me re-read: W[k*n+i, r] where output is C[row, col]
    # C[row, col] = sum_j A[row, j] * B[j, col]
    # row = 0..n-1, col = 0..p-1
    # A index: i = row * m + j  (row-major for A)
    # B index: j * p + col
    # C index: W[k*n + i] where k = col, i = row? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So C[row, col] is at position col * n + row in W
    
    # Let me build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for row in range(n):
        for col in range(p):
            for j in range(m):
                a_idx = row * m + j
                b_idx = j * p + col
                c_idx = col * n + row  # W[k*n+i] with k=col, i=row
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try to find low-rank decomposition
    # Rank r means: T = sum_{s=0}^{r-1} u_s ⊗ v_s ⊗ w_s
    # where u_s in R^nm, v_s in R^mp, w_s in R^pn
    
    # Start with schoolbook rank = 54
    best_U = None
    best_V = None
    best_W = None
    best_rank = nm * mp  # 162? No, schoolbook is n*m*p = 54
    
    # Actually schoolbook: for each (i,j,k) triple
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # That's n*m*p = 3*3*6 = 54 terms
    
    # Let's try to find lower rank using optimization
    # We'll try ranks from 30 up to 54
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U @ diag @ V @ diag @ W reconstructs T"""
        # T_approx[i,j,k] = sum_s U[i,s] * V[j,s] * W[k,s]
        # Use einsum
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, schoolbook_rank = generate_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), schoolbook_rank
    
    # Try to reduce rank by optimization
    # Use a simple approach: try various ranks and optimize
    
    def optimize_decomposition(rank, max_iter=1000, tol=1e-8):
        """Try to find a rank-r decomposition using iterative optimization"""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random factors
        np.random.seed(42)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Use ALS (Alternating Least Squares) style updates
        # For fixed V, W: minimize ||T - sum_s v_s ⊗ w_s ⊗ u_s||_F^2 over u_s
        # This is a linear system
        
        # Reshape T to (nm, mp*pn) for ALS on U
        # T[i, j, k] = sum_s U[i,s] * V[j,s] * W[k,s]
        # For fixed s, the contribution to row i is U[i,s] * (V[:,s] ⊗ W[:,s])
        
        # Let's use a simpler approach: gradient descent
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Compute gradient for U
            # dT/dU[i,s] = sum_{j,k} (T[i,j,k] - sum_s' U[i,s']*V[j,s']*W[k,s']) * V[j,s] * W[k,s]
            T_approx = np.einsum('is,js,ks->ijk', U, V, W)
            err = T - T_approx
            
            # Gradient for U: grad_U[i,s] = sum_{j,k} err[i,j,k] * V[j,s] * W[k,s]
            grad_U = np.einsum('ijk,js,ks->is', err, V, W)
            grad_V = np.einsum('ijk,is,ks->js', err, U, W)
            grad_W = np.einsum('ijk,is,js->ks', err, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 100 == 0:
                max_err = np.max(np.abs(T_approx - T))
                if max_err < tol:
                    break
        
        # Check if we found a valid decomposition
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        max_err = np.max(np.abs(T_approx - T))
        
        if max_err < 1e-4:  # somewhat loose, will refine
            # Try to clean up: set small values to 0
            U_clean = np.where(np.abs(U) < 1e-6, 0, U)
            V_clean = np.where(np.abs(V) < 1e-6, 0, V)
            W_clean = np.where(np.abs(W) < 1e-6, 0, W)
            
            T_clean = np.einsum('is,js,ks->ijk', U_clean, V_clean, W_clean)
            if np.max(np.abs(T_clean - T)) < 1e-3:
                return U_clean, V_clean, W_clean, max_err
        return None
    
    # Try various ranks
    for rank in range(30, 55):
        if time.time() > deadline - 20:
            break
        
        result = optimize_decomposition(rank)
        if result is not None:
            U_new, V_new, W_new, err = result
            if err < 1e-4:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
                break  # Found a valid lower rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
