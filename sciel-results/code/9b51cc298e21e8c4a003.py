import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 110
    np.random.seed(42)
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def compute_tensor(U, V, W, rank):
        """Compute the reconstructed tensor from factors."""
        S = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            S += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
        return S
    
    def verify(U, V, W, rank, tol=1e-9):
        """Verify exact decomposition."""
        S = compute_tensor(U, V, W, rank)
        return np.all(np.abs(S - 1.0) < tol)
    
    def snap_half(x):
        """Snap to nearest half-integer."""
        return round(x * 2) / 2.0
    
    def snap_matrix(M):
        """Snap all entries to half-integers."""
        return np.array([[snap_half(v) for v in row] for row in M])
    
    def compute_error(U, V, W, rank):
        """Compute sum of squared errors."""
        S = compute_tensor(U, V, W, rank)
        return np.sum((S - 1.0) ** 2)
    
    # Strategy: Try ranks from 60 down to 50
    # For each rank, use random initialization + Newton-like refinement
    # Key structural change: use a "factorize and re-optimize" approach where
    # we treat the problem as a matrix completion / low-rank tensor problem
    
    # Phase 1: Try to find low-rank solutions via iterative refinement
    # The insight: if we have an approximate solution, we can use the 
    # "alternating minimization" approach where we fix two factors and 
    # solve for the third via least squares.
    
    def alternating_minimize(U, V, W, rank, max_iter=50, tol=1e-10):
        """
        Alternating minimization:
        - Fix V, W: solve for U via linear least squares
        - Fix U, W: solve for V via linear least squares  
        - Fix U, V: solve for W via linear least squares
        
        The key: for fixed V, W, the equations are LINEAR in U!
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        For fixed V, W, this is: T[i,j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
        This is a linear system in U!
        """
        U = U.copy().astype(np.float64)
        V = V.copy().astype(np.float64)
        W = W.copy().astype(np.float64)
        
        for it in range(max_iter):
            old_err = compute_error(U, V, W, rank)
            
            # Solve for U given V, W
            # For each (i,j), U[i*m+j, :] appears in equations for all k
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Let A[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
            # Then T[i,j,k] = sum_c A[i,j,k,c] * U[i*m+j, c]
            # For fixed (i,j), this is a linear system: sum_c A[i,j,k,c] * U[i*m+j,c] = 1 for all k
            # That's p equations in rank unknowns for U[i*m+j, :]
            # If rank > p, it's underdetermined. Use least squares.
            # Actually, we have n*m*p equations total, but they group by (i,j) for U.
            
            # Build the system for U:
            # For each i, j: vector of p equations: sum_c (V[j*p+k,c]*W[k*n+i,c]) * U[i*m+j,c] = 1
            A_U = np.zeros((n, m, p, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_U[i, j, k, c] = V[j*p+k, c] * W[k*n+i, c]
            # Reshape: for each (i,j), we have p equations in rank unknowns
            # Stack all: (n*m*p) equations, (n*m*rank) unknowns
            A_U_flat = A_U.reshape(n*m*p, rank)  # This is wrong, need to be careful
            # Actually each equation (i,j,k) involves U[i*m+j, c] for all c
            # So the unknown vector is U flattened: n*m*rank variables
            # Equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Let's build it properly
            n_eq = n * m * p
            n_var_U = n * m * rank
            n_var_V = m * p * rank
            n_var_W = p * n * rank
            
            # For U: equation (i,j,k) has coefficient for U[i*m+j, c] = V[j*p+k,c]*W[k*n+i,c]
            # Build A_U_mat: n_eq x n_var_U
            A_U_mat = np.zeros((n_eq, n_var_U))
            A_V_mat = np.zeros((n_eq, n_var_V))
            A_W_mat = np.zeros((n_eq, n_var_W))
            
            b = np.ones(n_eq)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq = i * m * p + j * p + k
                        for c in range(rank):
                            # U variable index: (i*m+j) * rank + c
                            u_idx = (i * m + j) * rank + c
                            # V variable index: (j*p+k) * rank + c  
                            v_idx = (j * p + k) * rank + c
                            # W variable index: (k*n+i) * rank + c
                            w_idx = (k * n + i) * rank + c
                            
                            coeff = 1.0
                            # For U equation: coefficient of U[u_idx] is V[j*p+k,c]*W[k*n+i,c]
                            A_U_mat[eq, u_idx] = V[j*p+k, c] * W[k*n+i, c]
                            # For V equation: coefficient of V[v_idx] is U[i*m+j,c]*W[k*n+i,c]
                            A_V_mat[eq, v_idx] = U[i*m+j, c] * W[k*n+i, c]
                            # For W equation: coefficient of W[w_idx] is U[i*m+j,c]*V[j*p+k,c]
                            A_W_mat[eq, w_idx] = U[i*m+j, c] * V[j*p+k, c]
            
            # Solve for U: A_U_mat @ U_flat = b
            U_flat, _, _, _ = np.linalg.lstsq(A_U_mat, b, rcond=None)
            U = U_flat.reshape(n*m, rank)
            
            # Solve for V: A_V_mat @ V_flat = b
            V_flat, _, _, _ = np.linalg.lstsq(A_V_mat, b, rcond=None)
            V = V_flat.reshape(m*p, rank)
            
            # Solve for W: A_W_mat @ W_flat = b
            W_flat, _, _, _ = np.linalg.lstsq(A_W_mat, b, rcond=None)
            W = W_flat.reshape(p*n, rank)
            
            new_err = compute_error(U, V, W, rank)
            
            if new_err < tol:
                break
            if new_err > old_err * 1.1:
                break
        return U, V, W, compute_error(U, V, W, rank)
    
    def try_rank(rank, n_trials=5, time_per_trial=20):
        """Try to find a valid rank-r decomposition."""
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                return None
            
            # Random initialization with sparse structure
            # Most entries should be 0, 1, or -1 for matrix mult
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for c in range(rank):
                # Random sparse pattern
                nz_U = random.randint(1, 4)
                nz_V = random.randint(1, 4)
                nz_W = random.randint(1, 4)
                
                U_nz = random.sample(range(n*m), min(nz_U, n*m))
                V_nz = random.sample(range(m*p), min(nz_V, m*p))
                W_nz = random.sample(range(p*n), min(nz_W, p*n))
                
                for r in U_nz:
                    U[r, c] = random.choice([-1, 0.5, 1, 1.5, 2])
                for r in V_nz:
                    V[r, c] = random.choice([-1, 0.5, 1, 1.5, 2])
                for r in W_nz:
                    W[r, c] = random.choice([-1, 0.5, 1, 1.5, 2])
            
            # Run alternating minimization
            start_t = time.time()
            U, V, W, err = alternating_minimize(U, V, W, rank, max_iter=30, tol=1e-10)
            
            if err < 1e-8:
                # Try to snap to half-integers
                U_snap = snap_matrix(U)
                V_snap = snap_matrix(V)
                W_snap = snap_matrix(W)
                if verify(U_snap, V_snap, W_snap, rank):
                    return U_snap, V_snap, W_snap
                
                # Try without snapping (might still be exact)
                if verify(U, V, W, rank):
                    # Check if entries are already half-integers
                    all_half = all(abs(x*2 - round(x*2)) < 1e-9 for x in np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
                    if all_half:
                        return U, V, W
            
            # Also try: start from a perturbation of a known structure
            if trial == 0:
                # Try structured init based on Strassen-like patterns
                for c in range(rank):
                    U[r, c] = 0
                    V[r, c] = 0  
                    W[r, c] = 0
                    # Add small random perturbation
                    for r in range(n*m):
                        if random.random() < 0.3:
                            U[r, c] = random.uniform(-1, 1)
                    for r in range(m*p):
                        if random.random() < 0.3:
                            V[r, c] = random.uniform(-1, 1)
                    for r in range(p*n):
                        if random.random() < 0.3:
                            W[r, c] = random.uniform(-1, 1)
                
                U, V, W, err = alternating_minimize(U, V, W, rank, max_iter=30, tol=1e-10)
                if err < 1e-8:
                    U_snap = snap_matrix(U)
                    V_snap = snap_matrix(V)
                    W_snap = snap_matrix(W)
                    if verify(U_snap, V_snap, W_snap, rank):
                        return U_snap, V_snap, W_snap
    
    # Try ranks from 60 down to 50
    best_result = None
    for rank in range(60, 49, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank, n_trials=3, time_per_trial=20)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, rank)
            print(f"Found valid decomposition at rank {rank}")
            break
        else:
            print(f"Rank {rank} failed after {time.time():.1f}s")
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook rank 72
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
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
