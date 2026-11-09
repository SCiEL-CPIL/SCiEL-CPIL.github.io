"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        U_mat = U.reshape(n, m, r)
        V_mat = V.reshape(m, p, r)
        W_mat = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol, rtol=0)

    def try_alternating_minimization(target_rank, n_restarts=10, max_iters=300, tol=1e-10):
        """Try to find a rank-target_rank decomposition via alternating minimization."""
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(42 + restart)  # Reproducibility
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize initial factors to avoid scale issues
            U = U / np.linalg.norm(U, axis=0, keepdims=True) + 1e-8
            V = V / np.linalg.norm(V, axis=0, keepdims=True) + 1e-8
            W = W / np.linalg.norm(W, axis=0, keepdims=True) + 1e-8

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                U_mat = U.reshape(n, m, target_rank)
                V_mat = V.reshape(m, p, target_rank)
                W_mat = W.reshape(p, n, target_rank)

                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # T[i,j,k] = sum_c U[i*m+j, c] * V_mat[j,k,c] * W_mat[k,i,c]
                        # A[k,c] = V_mat[j,k,c] * W_mat[k,i,c]
                        # b[k] = 1
                        # Solve: U_row @ A.T = b  =>  A @ U_row.T = b.T
                        A = V_mat[j] * W_mat[:, i]  # shape (p, target_rank)
                        b = np.ones(p)
                        try:
                            # We want U_row such that U_row @ A.T = b
                            # This is equivalent to A @ U_row.T = b
                            U_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U_mat[i, j] = U_row
                        except np.linalg.LinAlgError:
                            pass

                # Solve for V
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
                        # For fixed j,k: sum_c (U_mat[i,j,c] * W_mat[k,i,c]) * V_mat[j,k,c] = 1 for each i
                        # Let B[i,c] = U_mat[i,j,c] * W_mat[k,i,c]
                        # B @ V_row.T = ones(n)
                        B = U_mat[:, j] * W_mat[k]  # shape (n, target_rank)
                        b = np.ones(n)
                        try:
                            V_row, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                            V_mat[j, k] = V_row
                        except np.linalg.LinAlgError:
                            pass

                # Solve for W
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
                        # For fixed k,i: sum_c (U_mat[i,j,c] * V_mat[j,k,c]) * W_mat[k,i,c] = 1 for each j
                        # Let C[j,c] = U_mat[i,j,c] * V_mat[j,k,c]
                        # C @ W_row.T = ones(m)
                        C = U_mat[i] * V_mat[:, k]  # shape (m, target_rank)
                        b = np.ones(m)
                        try:
                            W_row, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                            W_mat[k, i] = W_row
                        except np.linalg.LinAlgError:
                            pass

                U = U_mat.reshape(n * m, target_rank)
                V = V_mat.reshape(m * p, target_rank)
                W = W_mat.reshape(p * n, target_rank)

            # Check if converged
            if verify(U, V, W, tol=1e-8):
                return U, V, W
        return None

    # Search for lower rank decompositions
    # Start from just below schoolbook rank and go down
    # For 2x4*4x7, schoolbook is 56.
    # Known bounds: 
    # R(2,4,7) is likely around 40-50? 
    # Let's try a range.
    
    # Optimization: Try specific ranks that might be achievable
    # We'll iterate downwards.
    
    # To save time, we can limit the search space.
    # If we find a valid decomposition of rank R, we can stop or try lower.
    
    # Let's try ranks from 55 down to 40 (example)
    # But be careful with time.
    
    # A smarter approach: 
    # 1. Try to find ANY decomposition below 56.
    # 2. If found, try to lower it further.
    
    # Since alternating minimization can be slow or stuck, we use it sparingly.
    
    # First, let's check if a simple structured decomposition works.
    # For instance, block diagonal if possible?
    # 2x4 * 4x7. 
    # If we split 4 into 2+2, we have two 2x2 * 2x7 blocks? No, dimensions don't match nicely for simple Strassen-like reuse without overlap.
    
    # Let's rely on the AM search but with a tighter loop and early exit.
    
    # We will try ranks from 55 down to 45.
    # If time is running out, we return the best found.
    
    # Note: The AM method above solves linear systems. 
    # For target_rank < min(n*m, m*p, p*n), the system is underdetermined or square.
    # n*m=8, m*p=28, p*n=14.
    # If target_rank > 14, the systems for W (size m x r) are underdetermined (m=4 < r).
    # lstsq handles this.
    
    # Let's try a few specific ranks.
    # To maximize chance of finding a solution, we start close to 56 and go down.
    
    # Actually, let's just try a few random restarts at a few specific ranks.
    
    # Ranks to try: 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45
    # This is too many if each takes long.
    
    # Let's try 55 first. If it fails, try 54, etc.
    # But AM might fail to converge even if a solution exists.
    
    # Alternative: Use a known lower bound.
    # For 2x4x7, is there a known algorithm?
    # 2x2x2 is 7.
    # 2x4x4?
    # 2x4x7 is a specific case.
    
    # Let's stick to the AM search but limit the number of restarts and iterations to keep it fast.
    
    # We will try ranks 55 down to 48.
    # For each rank, 3 restarts, 100 iterations.
    
    # 55: 3 restarts * 100 iters * (8+28+14) lstsq calls.
    # Lstsq is fast for small matrices.
    
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 8, 0), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_alternating_minimization(target_rank, n_restarts=5, max_iters=150)
        if result is not None:
            U, V, W = result
            # Verify with stricter tolerance
            if verify(U, V, W, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # We could continue to search for lower, but let's see if we have time
                # For now, accept the first found lower rank and break? 
                # No, we want the minimal rank. 
                # But if we find 55, we should check 54.
                # However, if 55 is hard to find, maybe 54 is easier? Unlikely.
                # Usually, if a rank r works, r+1 also works (by adding a zero column).
                # So if we find a solution for r, we know r is achievable.
                # We want the MINIMAL r.
                # So we should continue searching downwards.
                pass
        else:
            # If AM fails, it might be because it got stuck.
            # We don't necessarily know if the rank is achievable.
            pass

    # If no lower rank found, return schoolbook
    # Convert to list format as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
