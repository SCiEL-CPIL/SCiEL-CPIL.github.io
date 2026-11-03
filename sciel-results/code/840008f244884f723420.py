"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for return
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1.0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition as baseline (Rank 42)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def snap_to_half_integers(mat):
        """Snap values to nearest 0.5 increment."""
        return np.round(mat * 2) / 2.0

    def verify_snapped(U, V, W):
        """Verify the snapped decomposition is exact."""
        if not check_decomposition(U, V, W, tol=1e-9):
            return False
        # Check if all values are effectively half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    def als_iteration(U, V, W, T, max_iter=20, tol=1e-12):
        """
        Perform Alternating Least Squares to minimize ||U x V x W - T||_F.
        U: (nm, r), V: (mp, r), W: (pn, r)
        T: (nm, mp, pn)
        """
        nm, mp, pn = T.shape
        r = U.shape[1]
        
        # Precompute indices for vectorization
        # We need to solve linear systems for each row of U, V, W
        
        # Residual calculation helper
        def compute_residual(U, V, W):
            return np.einsum('ac,bc,dc->abd', U, V, W) - T

        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            # For each i in [0, nm):
            # Minimize || T[i,:,:] - sum_r U[i,r] V[:,r] W[:,r] ||_F
            # Let M[i, j, k, r] = V[j, r] * W[k, r]
            # We want to solve for U[i, :] given V, W
            # This is a linear least squares problem:
            # y = T[i, :, :] (flattened to mp*pn)
            # X = reshape(V[:, r] * W[:, r] for r in range(r)) -> (mp*pn, r)
            # Note: V[:, r] is (mp, 1), W[:, r] is (pn, 1). Their outer product is (mp, pn).
            # We can construct X efficiently.
            
            # Construct design matrix X_U of shape (mp*pn, r)
            # X_U[idx, r] = V[j, r] * W[k, r] where idx = j*pn + k
            # This can be done via broadcasting
            # V is (mp, r), W is (pn, r)
            # V[:, :, None] is (mp, r, 1)
            # W[None, :, :] is (1, pn, r) -- Wait, W is (pn, r).
            # We need outer product of V[:, r] and W[:, r] for each r.
            # Let's just build it directly for small sizes.
            
            X_U = np.zeros((mp * pn, r))
            for r_idx in range(r):
                # Outer product of V[:, r_idx] (mp) and W[:, r_idx] (pn)
                outer = np.outer(V[:, r_idx], W[:, r_idx])
                X_U[:, r_idx] = outer.flatten()
                
            for i in range(nm):
                y = T[i, :, :].flatten()
                # Solve X_U @ U[i, :] = y
                # Use lstsq for minimum norm solution
                sol, residuals, rank, sv = np.linalg.lstsq(X_U, y, rcond=None)
                U[i, :] = sol
            
            # Update V
            # For each j in [0, mp):
            # Minimize || T[:, j, :] - sum_r U[:, r] V[j, r] W[:, r] ||_F
            # y = T[:, j, :] flattened (nm*pn)
            # X_V[idx, r] = U[i, r] * W[k, r] where idx = i*pn + k
            
            X_V = np.zeros((nm * pn, r))
            for r_idx in range(r):
                outer = np.outer(U[:, r_idx], W[:, r_idx])
                X_V[:, r_idx] = outer.flatten()
                
            for j in range(mp):
                y = T[:, j, :].flatten()
                sol, residuals, rank, sv = np.linalg.lstsq(X_V, y, rcond=None)
                V[j, :] = sol

            # Update W
            # For each k in [0, pn):
            # Minimize || T[:, :, k] - sum_r U[:, r] V[:, r] W[k, r] ||_F
            # y = T[:, :, k] flattened (nm*mp)
            # X_W[idx, r] = U[i, r] * V[j, r] where idx = i*mp + j
            
            X_W = np.zeros((nm * mp, r))
            for r_idx in range(r):
                outer = np.outer(U[:, r_idx], V[:, r_idx])
                X_W[:, r_idx] = outer.flatten()
                
            for k in range(pn):
                y = T[:, :, k].flatten()
                sol, residuals, rank, sv = np.linalg.lstsq(X_W, y, rcond=None)
                W[k, :] = sol

            # Check convergence
            err = np.linalg.norm(compute_residual(U, V, W))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Search for lower rank decompositions
    # Known optimal rank for 2x3x7 is 33.
    # We try ranks from 33 up to best_rank - 1.
    
    for target_rank in range(33, best_rank):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random initializations
        num_attempts = 10
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, T, max_iter=30)
            
            # Check if it's a good solution
            if check_decomposition(U, V, W, tol=1e-8):
                # Try to snap to half-integers
                U_snapped = snap_to_half_integers(U)
                V_snapped = snap_to_half_integers(V)
                W_snapped = snap_to_half_integers(W)
                
                if verify_snapped(U_snapped, V_snapped, W_snapped):
                    best_rank = target_rank
                    best_U = U_snapped
                    best_V = V_snapped
                    best_W = W_snapped
                    found = True
                    break
                else:
                    # If snapping failed, maybe the floating point solution is good enough for the evaluator?
                    # The contract says "integer/half-integer entries". So we MUST snap.
                    # If snapping breaks exactness, this rank is not found with this init.
                    pass
        
        if found:
            break

    # Convert to lists
    def to_list(mat):
        return [[0 if abs(x) < 1e-12 else (1 if abs(x-1) < 1e-12 else (-1 if abs(x+1) < 1e-12 else (0.5 if abs(x-0.5) < 1e-12 else (-0.5 if abs(x+0.5) < 1e-12 else x)))) for x in row] for row in mat]
        
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
