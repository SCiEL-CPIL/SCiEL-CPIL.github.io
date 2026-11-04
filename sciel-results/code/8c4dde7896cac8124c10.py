"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()

    start_time = time.time()
    deadline = start_time + 145 # Safety margin

    # Precompute target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T_target[a * m + b, b * p + c, c * n + a] = 1.0
                
    # Unfold T_target for HOOI
    # T_target shape: (n*m, m*p, p*n)
    # We need mode-1, mode-2, mode-3 unfoldings
    # Mode 1: (n*m) x (m*p * p*n)
    T1 = T_target.reshape(n * m, -1)
    # Mode 2: (m*p) x (n*m * p*n)
    T2 = np.transpose(T_target, (1, 0, 2)).reshape(m * p, -1)
    # Mode 3: (p*n) x (n*m * m*p)
    T3 = np.transpose(T_target, (2, 0, 1)).reshape(p * n, -1)

    def hooi(U, V, W, T1, T2, T3, max_iter=100):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            # B_U[:, r] = vec(V[:, r] * W[:, r])
            # V[:, r] shape (m*p, 1), W[:, r] shape (p*n, 1)
            # Outer product shape (m*p, p*n)
            # We need to construct B_U efficiently.
            # B_U = np.zeros((m*p*p*n, R))
            # for r in range(R):
            #     B_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Vectorized:
            # V shape (m*p, R), W shape (p*n, R)
            # We want a matrix of size (m*p*p*n, R)
            # This is the Khatri-Rao product of V and W? No.
            # It's the element-wise product of V and W flattened?
            # Actually, the tensor product of V[:, r] and W[:, r] is a matrix.
            # Flattening it gives a vector.
            # B_U = np.kron? No.
            
            # Let's just loop over R if R is small, or use broadcasting.
            # m*p = 30, p*n = 18. Product = 540.
            # R is up to 90.
            # Loop is 90 * 540 ops. Very fast.
            
            B_U = np.zeros((m * p * p * n, R))
            for r in range(R):
                B_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
            # Solve U = (B_U^T B_U)^-1 B_U^T T1
            # T1 shape (n*m, m*p*p*n)
            # We need to solve for each i in n*m?
            # No, U has shape (n*m, R).
            # The equation is T1 @ (B_U^T B_U)^-1 B_U^T?
            # Let X = (B_U^T B_U)^-1 B_U^T. Shape (R, m*p*p*n)
            # Then U = T1 @ X.T. Shape (n*m, R).
            
            try:
                BtB = B_U.T @ B_U
                BtT = B_U.T @ T1.T # Shape (R, n*m)
                # U = (BtB)^-1 BtT
                U_new = np.linalg.solve(BtB, BtT).T
            except np.linalg.LinAlgError:
                U_new = U # Fallback if singular
                
            U = U_new
            
            # Update V
            B_V = np.zeros((n * m * p * n, R))
            for r in range(R):
                B_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
            try:
                BtB = B_V.T @ B_V
                BtT = B_V.T @ T2.T
                V_new = np.linalg.solve(BtB, BtT).T
            except np.linalg.LinAlgError:
                V_new = V
                
            V = V_new
            
            # Update W
            B_W = np.zeros((n * m * m * p, R))
            for r in range(R):
                B_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
            try:
                BtB = B_W.T @ B_W
                BtT = B_W.T @ T3.T
                W_new = np.linalg.solve(BtB, BtT).T
            except np.linalg.LinAlgError:
                W_new = W
                
            W = W_new
            
        return U, V, W

    # Try ranks from 89 down to 1
    for R in range(rank_trivial - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Initialize
        # Use random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Run HOOI
        U, V, W = hooi(U, V, W, T1, T2, T3, max_iter=50)
        
        # Check error
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.sum((T_approx - T_target) ** 2)
        
        if error < 1e-9:
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_approx_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            error_snap = np.sum((T_approx_snap - T_target) ** 2)
            
            if error_snap < 1e-9:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            else:
                # If snap fails, try to refine the snapped solution?
                # Or just accept if error is very small?
                # The contract requires exact decomposition.
                # If snap fails, we might need to search locally around the snapped solution.
                # But for now, if HOOI found a solution, it's likely the snap is close.
                # Let's try a few local steps on the snapped solution.
                pass

    # Fallback to trivial
    return {
        "U": to_list(U_trivial),
        "V": to_list(V_trivial),
        "W": to_list(W_trivial)
    }
# EVOLVE-BLOCK-END
