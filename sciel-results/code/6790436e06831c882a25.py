"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    # Deadline guard: 150s wall-clock limit, use 115s for safety
    start_time = time.time()
    deadline = start_time + 115
    
    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = schoolbook_rank
    
    # Precompute the target tensor T flattened
    # T[i, j, k] = 1 if i,j,k correspond to a valid multiplication term?
    # The tensor for matrix mult C = A * B is defined such that C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition is C[i, k] = sum_r U[i, j_r] V[j_r, k_r] W[k_r, i]?
    # Wait, the interface defines U as (n*m, rank), V as (m*p, rank), W as (p*n, rank).
    # Indices:
    # U row index: i*m + j  (A index)
    # V row index: j*p + k  (B index)
    # W row index: k*n + i  (C index)
    # The tensor element T[A_idx, B_idx, C_idx] is 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i) for some i,j,k.
    # Specifically, the condition is that the indices match:
    # A_idx = i*m + j
    # B_idx = j*p + k
    # C_idx = k*n + i
    # So T is a 3-way tensor. We are looking for rank-r decomposition:
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # Flatten T to a vector of length nm * mp * pn
    # T_flat[a*mp*pn + b*pn + c] = 1 if there exists i,j,k such that:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    T_flat = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[a * mp * pn + b * pn + c] = 1.0

    def verify_fast(U, V, W, tol=1e-9):
        """
        Fast verification using NumPy.
        U: (nm, r), V: (mp, r), W: (pn, r)
        Computes T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        Flattens and compares to T_flat.
        """
        r = U.shape[1]
        # T_approx[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We can compute this by iterating over r or using einsum.
        # einsum('ar,br,cr->abc', U, V, W) is efficient.
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        T_approx_flat = T_approx.flatten()
        return np.allclose(T_approx_flat, T_flat, atol=tol)

    # Structured random search with numerical refinement
    # Strategy: For a target rank r, initialize U, V, W randomly,
    # then use alternating least squares to minimize the residual.
    # If we can drive the residual to zero, we have a valid decomposition.
    
    def residual_fast(U, V, W):
        """Compute the Frobenius norm of the residual T_approx - T."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.linalg.norm(T_approx - T_flat.reshape(nm, mp, pn))
    
    def try_als(target_rank, num_restarts=5, max_iter=200):
        """
        Try to find a rank-r decomposition using Alternating Least Squares.
        Initialize U, V, W randomly, then iteratively minimize the residual.
        """
        if time.time() > deadline:
            return None
            
        best_residual = float('inf')
        best_solution = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with small values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            prev_residual = residual_fast(U, V, W)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Update U: for each a, solve (V^T W^T) U[:,r] = T[a,:,:].reshape(-1)
                # Actually, we need to be more careful. The residual is:
                # R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c]
                # For fixed V, W, the optimal U[a,:] minimizes:
                # sum_{b,c} (sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
                # This is a linear least squares problem for U[a,:].
                
                # Build the design matrix for U:
                # For each a, we need to solve:
                # X_U U[a,:] = y_U
                # where X_U has rows for each (b,c) pair, and the row is [V[b,0]*W[c,0], ..., V[b,r-1]*W[c,r-1]]
                
                # This is expensive to build for each a. Let's use a different approach.
                # Use gradient-based optimization instead.
                
                # Compute gradients using automatic differentiation-like approach
                # dR/da = U[a,r] * V[b,r] * W[c,r]
                # This is complex. Let's use a simpler heuristic.
                
                # Alternative: Use scipy.optimize.minimize with L-BFGS-B
                # But this might be too slow.
                
                # Let's try a different strategy: fixed support search
                # For a given rank r, choose random binary supports for U, V, W
                # Then solve the linear system for the values.
                
                break  # Skip to support-based approach
            
            # Support-based approach
            # For this restart, generate random supports
            U_mask = np.zeros((nm, target_rank), dtype=bool)
            V_mask = np.zeros((mp, target_rank), dtype=bool)
            W_mask = np.zeros((pn, target_rank), dtype=bool)
            
            # Density: each entry has probability 0.3 of being non-zero
            U_mask = np.random.rand(nm, target_rank) < 0.3
            V_mask = np.random.rand(mp, target_rank) < 0.3
            W_mask = np.random.rand(pn, target_rank) < 0.3
            
            # Ensure each column has at least one non-zero in each matrix
            for r in range(target_rank):
                if not U_mask[:, r].any():
                    U_mask[np.random.randint(nm), r] = True
                if not V_mask[:, r].any():
                    V_mask[np.random.randint(mp), r] = True
                if not W_mask[:, r].any():
                    W_mask[np.random.randint(pn), r] = True
            
            # Build the linear system:
            # For each (a,b,c) with T[a,b,c] != 0:
            # sum_r U[a,r] V[b,r] W[c,r] = 1
            # For each (a,b,c) with T[a,b,c] == 0:
            # sum_r U[a,r] V[b,r] W[c,r] = 0
            # This is a system of equations in the products U[a,r]*V[b,r]*W[c,r]
            # But these are not independent variables.
            
            # Instead, fix U and V, then solve for W, and iterate.
            # Initialize U, V with random values where masks are True
            U_curr = np.zeros((nm, target_rank))
            V_curr = np.zeros((mp, target_rank))
            W_curr = np.zeros((pn, target_rank))
            U_curr[U_mask] = np.random.randn(U_mask.sum())
            V_curr[V_mask] = np.random.randn(V_mask.sum())
            
            # Iterate: given U, V, solve for W
            # For each c, the equation is:
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all (a,b)
            # This is a linear system in W[c,:]
            
            for inner_iter in range(50):
                if time.time() > deadline:
                    break
                    
                # Solve for W given U, V
                # For each c, build the system:
                # For each (a,b): sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
                # This is a system with target_rank unknowns (W[c,0], ..., W[c,r-1])
                # and nm*mp equations (20*25 = 500 equations)
                
                W_new = np.zeros((pn, target_rank))
                for c in range(pn):
                    # Build design matrix A of shape (nm*mp, target_rank)
                    # A[(a,b), r] = U[a,r] * V[b,r]
                    # But we only need the non-zero T entries for the right-hand side
                    # However, we also need to satisfy the zero entries.
                    
                    # Let's use all entries for a proper least squares solution
                    A = np.zeros((nm * mp, target_rank))
                    for a in range(nm):
                        for b in range(mp):
                            idx = a * mp + b
                            A[idx, :] = U_curr[a, :] * V_curr[b, :]
                    
                    # Right-hand side: T[a,b,c] for all (a,b)
                    y = np.zeros(nm * mp)
                    for a in range(nm):
                        for b in range(mp):
                            idx = a * mp + b
                            y[idx] = T_flat[a * mp * pn + b * pn + c]
                    
                    # Solve least squares
                    try:
                        W_c, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                        W_new[c, :] = W_c
                    except:
                        W_new[c, :] = W_curr[c, :]
                
                W_curr = W_new
                
                # Solve for U given V, W
                U_new = np.zeros((nm, target_rank))
                for a in range(nm):
                    A = np.zeros((mp * pn, target_rank))
                    for b in range(mp):
                        for c in range(pn):
                            idx = b * pn + c
                            A[idx, :] = V_curr[b, :] * W_curr[c, :]
                    
                    y = np.zeros(mp * pn)
                    for b in range(mp):
                        for c in range(pn):
                            idx = b * pn + c
                            y[idx] = T_flat[a * mp * pn + b * pn + c]
                    
                    try:
                        U_a, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                        U_new[a, :] = U_a
                    except:
                        U_new[a, :] = U_curr[a, :]
                
                U_curr = U_new
                
                # Solve for V given U, W
                V_new = np.zeros((mp, target_rank))
                for b in range(mp):
                    A = np.zeros((nm * pn, target_rank))
                    for a in range(nm):
                        for c in range(pn):
                            idx = a * pn + c
                            A[idx, :] = U_curr[a, :] * W_curr[c, :]
                    
                    y = np.zeros(nm * pn)
                    for a in range(nm):
                        for c in range(pn):
                            idx = a * pn + c
                            y[idx] = T_flat[a * mp * pn + b * pn + c]
                    
                    try:
                        V_b, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                        V_new[b, :] = V_b
                    except:
                        V_new[b, :] = V_curr[b, :]
                
                V_curr = V_new
                
                # Check residual
                curr_residual = residual_fast(U_curr, V_curr, W_curr)
                if curr_residual < best_residual:
                    best_residual = curr_residual
                    best_solution = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                
                if curr_residual < 1e-10:
                    break
            
            if time.time() > deadline:
                break
        
        if best_solution is not None and best_residual < 1e-8:
            U_sol, V_sol, W_sol = best_solution
            # Verify
            if verify_fast(U_sol, V_sol, W_sol):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        
        return None
    
    # Try ranks from 99 down to 90
    for target_rank in range(99, 89, -1):
        if time.time() > deadline:
            break
        
        result = try_als(target_rank, num_restarts=3, max_iter=100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
