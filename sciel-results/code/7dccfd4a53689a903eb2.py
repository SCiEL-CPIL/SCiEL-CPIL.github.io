"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 60).
    2. Attempt to find a lower rank decomposition by perturbing the schoolbook solution 
       and using local optimization to satisfy the tensor identity.
    3. If a lower rank decomposition is found and can be rounded to integer/half-integer entries 
       while maintaining exactness, return it. Otherwise, return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    rank_schoolbook = n * m * p  # 60
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U_flat = U.reshape(n * m, -1)
        V_flat = V.reshape(m * p, -1)
        W_flat = W.reshape(p * n, -1)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_flat[i * m + j]
                    v_row = V_flat[j * p + k]
                    w_row = W_flat[k * n + i]
                    
                    val = np.dot(u_row * v_row, w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        return np.round(x * 2) / 2

    def solve_perturbed(target_rank, U_init, V_init, W_init, max_iters=1000):
        """
        Try to adjust U, V, W (initially from schoolbook truncated) to satisfy the identity.
        Uses a simple iterative correction.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Compute errors
        # E[i,j,k] = 1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want E to be 0.
        # We can update one variable at a time.
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute current errors
            errors = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_row = W[k * n + i]
                        val = np.dot(u_row * v_row, w_row)
                        errors[i, j, k] = 1.0 - val
            
            if np.max(np.abs(errors)) < 1e-9:
                return U, V, W, True
            
            # Update W (easiest as it's linear in W for fixed U, V)
            # For each (k, i), we want sum_j? No.
            # The term for W[k*n+i, r] is coupled with U[i*m+j, r] and V[j*p+k, r].
            # Let's fix U and V, and solve for W.
            # For a fixed (k, i), the contribution to C_{ik} from all j is:
            # sum_j (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]) = 1
            # This is sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1
            # Let A_{r} = sum_j U[i*m+j, r] V[j*p+k, r]
            # Then sum_r A_{r} W_{r} = 1.
            # This is a linear system for W? No, W has index (k, i, r).
            # For each (k, i), we have a vector W_{ki} of length R.
            # We have one equation: A_{ki} dot W_{ki} = 1.
            # This is underdetermined. We can set W_{ki} to the minimum norm solution:
            # W_{ki} = A_{ki} / (A_{ki} dot A_{ki})
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros(target_rank, dtype=float)
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        A += u_row * v_row
                    norm_sq = np.dot(A, A)
                    if norm_sq > 1e-12:
                        W[k * n + i] = A / norm_sq
                    else:
                        # Singular, skip or handle
                        pass
            
            # Similarly update U and V?
            # Update U: Fix V, W.
            # For fixed (i, j), we want sum_k (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]) = 1?
            # No, the identity is for each (i, j, k).
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
            # This is linear in U_{ij} if V, W fixed.
            # But U_{ij} appears in equations for all k.
            # For fixed i, j, we have p equations (one for each k).
            # sum_r U_{ij, r} (V[j*p+k, r] W[k*n+i, r]) = 1 for all k.
            # Let B_{k, r} = V[j*p+k, r] W[k*n+i, r].
            # Then B U_{ij}^T = 1 (vector of ones).
            # This is a system of p equations with R unknowns.
            # If R >= p, we can find a solution.
            
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, target_rank), dtype=float)
                    for k in range(p):
                        for r in range(target_rank):
                            B[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    # Solve B U_{ij}^T = 1
                    # Minimum norm solution: U_{ij}^T = B^T (B B^T)^{-1} 1
                    try:
                        BBT = B.dot(B.T)
                        # Add regularization
                        BBT += 1e-9 * np.eye(p)
                        U[i * m + j] = np.linalg.solve(BBT, np.ones(p)).dot(B)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V: Fix U, W.
            # For fixed j, k, we have n equations (one for each i).
            # sum_r V[j*p+k, r] (U[i*m+j, r] W[k*n+i, r]) = 1 for all i.
            # Let C_{i, r} = U[i*m+j, r] W[k*n+i, r].
            # Then C V_{jk}^T = 1.
            
            for j in range(m):
                for k in range(p):
                    C = np.zeros((n, target_rank), dtype=float)
                    for i in range(n):
                        for r in range(target_rank):
                            C[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        CCT = C.dot(C.T)
                        CCT += 1e-9 * np.eye(n)
                        V[j * p + k] = np.linalg.solve(CCT, np.ones(n)).dot(C)
                    except np.linalg.LinAlgError:
                        pass
        
        # Final check
        if check_decomposition(U, V, W):
            return U, V, W, True
        return U, V, W, False

    # Try to find a lower rank decomposition
    for target_rank in range(59, 54, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with first target_rank columns of schoolbook
        U_init = U[:, :target_rank].copy()
        V_init = V[:, :target_rank].copy()
        W_init = W[:, :target_rank].copy()
        
        # Try to solve
        U_sol, V_sol, W_sol, success = solve_perturbed(target_rank, U_init, V_init, W_init)
        
        if success:
            # Try rounding
            U_rounded = round_to_half(U_sol)
            V_rounded = round_to_half(V_sol)
            W_rounded = round_to_half(W_sol)
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break # Found a valid lower rank, stop searching for lower
        
    # Convert to lists for the required interface
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
