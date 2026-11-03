"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Tensor shape: (n, m, p) where T[i,j,k] corresponds to output element (i,k) from input (i,j) and (j,k)
    # The tensor is all ones: T[i,j,k] = 1
    
    # We want to find rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Strategy: Start with schoolbook (rank=60) and try to reduce rank
    # Use a structured approach with random initialization and iterative refinement
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # Generate schoolbook as baseline
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
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try various ranks from low to high
    # Known lower bounds suggest rank around 45-50 for 3x4x5
    # Try ranks from 40 upward
    
    for target_rank in range(40, 61):
        if time.time() > deadline - 20.0:
            break
        
        # Try multiple random initializations for this rank
        max_trials = 50 if target_rank <= 50 else 20
        if time.time() > deadline - 30.0:
            max_trials = min(max_trials, 10)
        
        for trial in range(max_trials):
            if time.time() > deadline - 15.0:
                break
            
            # Random initialization with structured patterns
            np.random.seed(trial * 1000 + target_rank)
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Iterative refinement using alternating least squares
            # Minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||^2
            # where T[i,j,k] = 1
            
            for iteration in range(200):
                if time.time() > deadline - 10.0:
                    break
                
                # Update U: for each (i,j), minimize sum_k (T[i,j,k] - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
                for i in range(n):
                    for j in range(m):
                        # Build the system: min_U sum_k (1 - sum_c U_c * A_c)^2
                        # where A_c = V[j,k,c] * W[k,i,c]
                        # This is a least squares problem
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Solve: min_U ||A @ U - ones(p)||^2
                        # Normal equations: (A^T A) U = A^T ones
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(p)
                            U[i*m+j, :] = np.linalg.solve(ATA + 1e-10*np.eye(target_rank), ATb)
                        except:
                            pass
                
                # Update V: for each (j,k), minimize sum_i (T[i,j,k] - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(n)
                            V[j*p+k, :] = np.linalg.solve(ATA + 1e-10*np.eye(target_rank), ATb)
                        except:
                            pass
                
                # Update W: for each (k,i), minimize sum_j (T[i,j,k] - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(m)
                            W[k*n+i, :] = np.linalg.solve(ATA + 1e-10*np.eye(target_rank), ATb)
                        except:
                            pass
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            error += (1.0 - val) ** 2
                
                if error < 1e-12:
                    # Found exact solution!
                    best_result = (U, V, W)
                    best_rank = target_rank
                    break
                elif error < 1e-8:
                    # Very close, try to polish
                    if time.time() > deadline - 5.0:
                        break
                    # Additional refinement iterations
                    for _ in range(100):
                        if time.time() > deadline - 3.0:
                            break
                        for i in range(n):
                            for j in range(m):
                                A = np.zeros((p, target_rank))
                                for k in range(p):
                                    for c in range(target_rank):
                                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                                try:
                                    ATA = A.T @ A
                                    ATb = A.T @ np.ones(p)
                                    U[i*m+j, :] = np.linalg.solve(ATA + 1e-12*np.eye(target_rank), ATb)
                                except:
                                    pass
                        for j in range(m):
                            for k in range(p):
                                A = np.zeros((n, target_rank))
                                for i in range(n):
                                    for c in range(target_rank):
                                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                                try:
                                    ATA = A.T @ A
                                    ATb = A.T @ np.ones(n)
                                    V[j*p+k, :] = np.linalg.solve(ATA + 1e-12*np.eye(target_rank), ATb)
                                except:
                                    pass
                        for k in range(p):
                            for i in range(n):
                                A = np.zeros((m, target_rank))
                                for j in range(m):
                                    for c in range(target_rank):
                                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                                try:
                                    ATA = A.T @ A
                                    ATb = A.T @ np.ones(m)
                                    W[k*n+i, :] = np.linalg.solve(ATA + 1e-12*np.eye(target_rank), ATb)
                                except:
                                    pass
                        
                        error = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for c in range(target_rank):
                                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                    error += (1.0 - val) ** 2
                        if error < 1e-15:
                            break
                    
                    if error < 1e-10:
                        best_result = (U, V, W)
                        best_rank = target_rank
                        break
        
        if best_rank <= target_rank:
            # Found a solution at this rank or lower
            break
    
    # Convert to lists and round to nearest half-integer if possible
    U_list = [[round(float(x), 6) for x in row] for row in best_result[0]]
    V_list = [[round(float(x), 6) for x in row] for row in best_result[1]]
    W_list = [[round(float(x), 6) for x in row] for row in best_result[2]]
    
    # Try to snap to half-integers
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                val = mat[i][j]
                # Try to snap to nearest half-integer
                half_int = round(val * 2) / 2
                if abs(val - half_int) < 1e-6:
                    mat[i][j] = half_int
                else:
                    mat[i][j] = val
    
    # Verify the solution
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    if not verify(U_list, V_list, W_list):
        # Fall back to schoolbook
        U_list = [[0.0] * rank_sb for _ in range(n*m)]
        V_list = [[0.0] * rank_sb for _ in range(m*p)]
        W_list = [[0.0] * rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list[i*m+j][col] = 1.0
                    V_list[j*p+k][col] = 1.0
                    W_list[k*n+i][col] = 1.0
                    col += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
