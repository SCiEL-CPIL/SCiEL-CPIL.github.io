# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from fractions import Fraction

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x4 matrix.
    Uses a randomized search to find a lower-rank decomposition if possible.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    # First, get the schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_best, V_best, W_best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, n, m, p, tol=1e-10):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: For a given rank r, fix U and V randomly, then solve for W
    # The equation is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W!
    
    # For fixed U, V, the system for W is:
    # For each (i,k): sum_j [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] ... 
    # Wait, let me re-examine. The equation is:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W[k*n+i,c] for each (i,k), but coupled across j.
    
    # Actually, for fixed (i,k), we have m equations (one for each j):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j = 0,1,...,m-1
    # This is a system of m equations in (p*n*rank) unknowns for W, but W[k*n+i,c] 
    # appears in all m equations for fixed (i,k).
    
    # Let me define: for fixed (i,k), let w_{i,k,c} = W[k*n+i,c]
    # Then for each j: sum_c U[i*m+j,c] * V[j*p+k,c] * w_{i,k,c} = 1
    # This is m equations in (rank) unknowns w_{i,k,0},...,w_{i,k,rank-1}
    
    # So for each (i,k) pair, we have a system of m equations in rank unknowns.
    # If rank >= m, this system may have a solution.
    
    # Let's try ranks from 24 up to 47 and see if we can find a valid decomposition.
    
    random.seed(42)
    
    for target_rank in range(24, 48):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Randomly initialize U and V with small values
            U_np = np.random.randn(n * m, target_rank) * 0.1
            V_np = np.random.randn(m * p, target_rank) * 0.1
            
            # For each (i,k), solve for W[k*n+i,:]
            W_np = np.zeros((p * n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the system: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * w_c = 1
                    # This is a linear system: A * w = b where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U_np[i * m + j, c] * V_np[j * p + k, c]
                    
                    b = np.ones(m)
                    
                    # Solve A * w = b using least squares
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution is valid (residuals should be small)
                        if len(residuals) > 0 and residuals[0] > 1e-8:
                            valid = False
                            break
                        W_np[k * n + i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_list = U_np.tolist()
            V_list = V_np.tolist()
            W_list = W_np.tolist()
            
            if verify(U_list, V_list, W_list, n, m, p):
                best_rank = target_rank
                U_best, V_best, W_best = U_list, V_list, W_list
                print(f"Found valid decomposition with rank {target_rank}")
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
