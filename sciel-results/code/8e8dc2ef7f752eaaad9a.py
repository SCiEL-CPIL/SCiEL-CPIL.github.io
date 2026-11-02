import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify_exact(U, V, W, R):
        """Verify exact decomposition with tight tolerance."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :R], np.multiply(V[v_idx, :R], W[w_idx, :R]))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Try to construct a rank 28 decomposition using Strassen-like structure
    # The 2x3x5 multiplication can be decomposed using 2x2 Strassen blocks
    # plus auxiliary terms
    
    # Strategy: Use the known rank-28 decomposition structure for 2x3x5
    # This exploits the block structure: partition into overlapping 2x2x2 Strassen sub-tensors
    
    # For 2x3x5, we can use a structured decomposition:
    # - Use Strassen's 7-term algorithm for 2x2 blocks
    # - Add auxiliary terms for the remaining dimensions
    
    # Let's try a direct construction based on known results
    target_rank = 28
    
    # Construct using a combination of Strassen and schoolbook terms
    # The 2x3 matrix A can be split into 2x2 and 2x1 blocks
    # The 3x5 matrix B can be split into 2x5 and 1x5 blocks
    
    # A = [A11 A12 A13]  (2x3)
    # B = [B1; B2; B3]   (3x5, each Bi is 1x5)
    
    # Actually, let's think of it as:
    # A is 2x3, B is 3x5
    # C = A * B is 2x5
    
    # We can write A as [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # B as [B_top; B_bottom] where B_top is 2x5, B_bottom is 1x5
    
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (2x2 * 2x5), we can use Strassen-like approach
    # For A_right * B_bottom (2x1 * 1x5), this is simpler
    
    # Let me try a different approach: use the known rank-28 construction
    
    # Based on the inspiration that Strassen reduces 2x2 sub-blocks,
    # let's construct explicitly
    
    # The 2x3x5 tensor can be seen as:
    # Sum over i=0..1, j=0..2, k=0..4 of A[i,j] * B[j,k] * C[i,k]
    
    # We'll use a hybrid approach:
    # - For the 2x2 part (i=0,1; j=0,1; k=0,1,2,3), use Strassen-inspired terms
    # - For the remaining terms, use schoolbook
    
    # Actually, let's try to find a rank-28 solution via optimization
    # but with a better initialization based on structure
    
    def get_error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :R], np.multiply(V[v_idx, :R], W[w_idx, :R]))
                    err += (s - 1.0) ** 2
        return err
    
    def gradient_descent(U, V, W, R, max_time, lr=0.01):
        """Gradient-based optimization to minimize the error."""
        start_time = time.time()
        deadline = start_time + max_time
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        best_err = get_error(U, V, W, R)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        while time.time() < deadline:
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        s = np.dot(U[u_idx, :R], np.multiply(V[v_idx, :R], W[w_idx, :R]))
                        ds = 2 * (s - 1.0)
                        
                        # dE/du = 2(s-1) * V*W
                        grad_U[u_idx, :R] += ds * np.multiply(V[v_idx, :R], W[w_idx, :R])
                        # dE/dv = 2(s-1) * U*W
                        grad_V[v_idx, :R] += ds * np.multiply(U[u_idx, :R], W[w_idx, :R])
                        # dE/dw = 2(s-1) * U*V
                        grad_W[w_idx, :R] += ds * np.multiply(U[u_idx, :R], V[v_idx, :R])
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_err = get_error(U, V, W, R)
            
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # Reduce learning rate on improvement
                lr *= 0.95
            else:
                # Revert and reduce learning rate
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                lr *= 0.8
            
            if best_err < 1e-12:
                break
        
        return best_U, best_V, best_W, best_err
    
    def local_search(U_init, V_init, W_init, R, max_time):
        """Combined gradient descent + random perturbation search."""
        best_U = U_init.copy()
        best_V = V_init.copy()
        best_W = W_init.copy()
        best_err = get_error(best_U, best_V, best_W, R)
        
        if best_err < 1e-9:
            return best_U, best_V, best_W, best_err

        start_time = time.time()
        deadline = start_time + max_time
        
        # Phase 1: Gradient descent
        U_gd, V_gd, W_gd, err_gd = gradient_descent(best_U, best_V, best_W, R, min(max_time * 0.5, 10.0))
        if err_gd < best_err:
            best_U, best_V, best_W, best_err = U_gd, V_gd, W_gd, err_gd
        
        if best_err < 1e-9:
            return best_U, best_V, best_W, best_err
        
        # Phase 2: Random perturbation + gradient polish
        random.seed(42)
        
        while time.time() < deadline:
            # Random perturbation
            U_pert = best_U + np.random.randn(n * m, R) * 0.1
            V_pert = best_V + np.random.randn(m * p, R) * 0.1
            W_pert = best_W + np.random.randn(p * n, R) * 0.1
            
            # Gradient polish from perturbed state
            time_left = deadline - time.time()
            if time_left < 1.0:
                break
                
            U_sol, V_sol, W_sol, err_sol = gradient_descent(U_pert, V_pert, W_pert, R, min(time_left * 0.4, 5.0))
            
            if err_sol < best_err:
                best_U, best_V, best_W, best_err = U_sol, V_sol, W_sol, err_sol
                if best_err < 1e-9:
                    break
        
        return best_U, best_V, best_W, best_err
    
    def snap_to_rational(U, V, W, R):
        """Try to snap values to nearby rational numbers with small denominators."""
        U_snap = U.copy()
        V_snap = V.copy()
        W_snap = W.copy()
        
        denominators = [1, 2, 3, 4, 5, 6, 8, 10, 12]
        
        for arr in [U_snap, V_snap, W_snap]:
            for idx in range(arr.size):
                val = arr.flat[idx]
                best_rational = val
                best_dist = abs(val - round(val))
                
                for d in denominators:
                    num = round(val * d)
                    rational = num / d
                    dist = abs(val - rational)
                    if dist < best_dist:
                        best_dist = dist
                        best_rational = rational
                
                arr.flat[idx] = best_rational
        
        return U_snap, V_snap, W_snap
    
    # Try to find rank 28 decomposition
    random.seed(123)
    np.random.seed(123)
    
    best_solution = None
    best_rank = rank_schoolbook
    
    # Generate diverse starting points by removing different column pairs
    # and adding random perturbations
    attempts = 0
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    all_cols = list(range(rank_schoolbook))
    
    while time.time() < deadline and attempts < 50:
        attempts += 1
        
        # Pick two random columns to remove
        cols_to_remove = random.sample(all_cols, 2)
        cols_to_keep = [c for c in all_cols if c not in cols_to_remove]
        
        U_init = U_school[:, cols_to_keep].copy()
        V_init = V_school[:, cols_to_keep].copy()
        W_init = W_school[:, cols_to_keep].copy()
        
        # Add small random perturbation for diversity
        if attempts > 5:
            U_init += np.random.randn(n * m, target_rank) * 0.05
            V_init += np.random.randn(m * p, target_rank) * 0.05
            W_init += np.random.randn(p * n, target_rank) * 0.05
        
        # Run local search with time budget
        time_left = deadline - time.time()
        search_time = min(time_left * 0.7, 8.0)
        
        if search_time < 2.0:
            break
            
        U_sol, V_sol, W_sol, err = local_search(U_init, V_init, W_init, target_rank, search_time)
        
        if err < 1e-8:
            # Try to snap to rational values
            U_snap, V_snap, W_snap = snap_to_rational(U_sol, V_sol, W_sol, target_rank)
            
            if verify_exact(U_snap, V_snap, W_snap, target_rank):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # Also check the unsnapped version
            if verify_exact(U_sol, V_sol, W_sol, target_rank):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        
        # Keep track of best solution found
        if best_solution is None or err < best_solution[3]:
            best_solution = (U_sol, V_sol, W_sol, err)
    
    # If we found a good approximate solution, try to refine it
    if best_solution is not None:
        U_best, V_best, W_best, err_best = best_solution
        
        if err_best < 1e-6:
            # Final high-precision polish
            time_left = deadline - time.time()
            if time_left > 5.0:
                U_final, V_final, W_final, err_final = gradient_descent(
                    U_best, V_best, W_best, target_rank, time_left * 0.8
                )
                
                if err_final < 1e-8:
                    U_snap, V_snap, W_snap = snap_to_rational(U_final, V_final, W_final, target_rank)
                    if verify_exact(U_snap, V_snap, W_snap, target_rank):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    if verify_exact(U_final, V_final, W_final, target_rank):
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Fallback: return schoolbook decomposition
    U_out = U_school.tolist()
    V_out = V_school.tolist()
    W_out = W_school.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
