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

    def check_solution(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T_target"""
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.sum((T_approx - T_target) ** 2)
        return error < tol, error

    def snap_to_halfint(U, V, W):
        """Snap to nearest half-integers and check validity"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        valid, err = check_solution(U_snap, V_snap, W_snap)
        return valid, (U_snap, V_snap, W_snap), err

    def local_search(U, V, W, max_iter=1000, step=0.1):
        """Local search to refine solution, trying to snap to half-integers"""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        # First try to snap
        valid, (U_snap, V_snap, W_snap), err = snap_to_halfint(U, V, W)
        if valid:
            return U_snap, V_snap, W_snap
        
        # Local search with random perturbations
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random perturbation
            dU = np.random.randn(*U.shape) * step
            dV = np.random.randn(*V.shape) * step
            dW = np.random.randn(*W.shape) * step
            
            U_new = U + dU
            V_new = V + dV
            W_new = W + dW
            
            _, err = check_solution(U_new, V_new, W_new)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                
                # Try to snap
                valid, (U_s, V_s, W_s), err_s = snap_to_halfint(U_new, V_new, W_new)
                if valid:
                    return U_s, V_s, W_s
            else:
                # Accept with some probability (simulated annealing-like)
                if random.random() < 0.1:
                    U, V, W = U_new, V_new, W_new
                else:
                    U, V, W = best_U, best_V, best_W
        
        return best_U, best_V, best_W

    # Strategy: Try structured approaches first
    # For 3x5x6, try to find solutions with rank < 90
    
    # Approach 1: Try specific rank values with multiple random restarts
    # Known: schoolbook is 90, so try 89, 88, ...
    
    # Precompute unfoldings for HOOI
    T1 = T_target.reshape(n * m, -1)
    T2 = np.transpose(T_target, (1, 0, 2)).reshape(m * p, -1)
    T3 = np.transpose(T_target, (2, 0, 1)).reshape(p * n, -1)

    def hooi(U, V, W, T1, T2, T3, max_iter=100):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            B_U = np.zeros((m * p * p * n, R))
            for r in range(R):
                B_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            try:
                BtB = B_U.T @ B_U
                BtT = B_U.T @ T1.T
                U_new = np.linalg.solve(BtB, BtT).T
            except np.linalg.LinAlgError:
                U_new = U
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

    # Try ranks from 89 down, with multiple restarts per rank
    for R in range(rank_trivial - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Multiple restarts for this rank
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Run HOOI
            U, V, W = hooi(U, V, W, T1, T2, T3, max_iter=30)
            
            # Check if close to a solution
            valid, err = check_solution(U, V, W)
            if valid:
                # Snap to half-integers
                valid_snap, (U_s, V_s, W_s), err_snap = snap_to_halfint(U, V, W)
                if valid_snap:
                    return {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
            
            # If close but not exact, try local search
            if err < 1e-4:
                U_l, V_l, W_l = local_search(U, V, W, max_iter=200, step=0.05)
                valid_l, err_l = check_solution(U_l, V_l, W_l)
                if valid_l:
                    valid_snap, (U_s, V_s, W_s), err_snap = snap_to_halfint(U_l, V_l, W_l)
                    if valid_snap:
                        return {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }

    # Fallback to trivial
    return {
        "U": to_list(U_trivial),
        "V": to_list(V_trivial),
        "W": to_list(W_trivial)
    }
# EVOLVE-BLOCK-END
