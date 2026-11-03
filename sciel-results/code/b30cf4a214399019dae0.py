"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using randomized search / local search
    # For 3x3 * 3x5, known bounds:
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Upper bound: 45 (schoolbook)
    # Known best for 3x3x3 is 23 (Strassen variant), but for 3x3x5 it's less clear.
    # Let's try to find a decomposition with rank < 45.
    
    # Strategy: Randomized local search
    # Represent decomposition as rank r, with U (n*m x r), V (m*p x r), W (p*n x r)
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, the tensor for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's try small ranks first
    # Try ranks from 15 to 44
    # Use randomized initialization and local search
    
    np.random.seed(42)
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 for all i,j,k
        nm = n * m
        mp = m * p
        pn = p * n
        r = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = len(U[0])
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this as:
        # For each c, we have a rank-1 tensor U_c * V_c * W_c
        # U_c is nm x 1, V_c is mp x 1, W_c is pn x 1
        # We need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
        
        # Let's build the full tensor
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            
            # Outer product contribution
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        
        return np.allclose(T, 1.0, atol=tol)
    
    # Try to find a decomposition with rank < 45
    # Start with rank 20 and go up
    for target_rank in range(15, 45):
        if time.time() > deadline - 10:
            break
        
        # Randomized search
        num_tries = 50
        for attempt in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Random initialization
            for c in range(target_rank):
                U[:, c] = np.random.randn(n*m) * 0.5
                V[:, c] = np.random.randn(m*p) * 0.5
                W[:, c] = np.random.randn(p*n) * 0.5
            
            # Local search: randomly perturb one entry and check if it improves
            # Objective: minimize sum (T[i,j,k] - 1)^2
            def objective(U, V, W):
                nm = n * m
                mp = m * p
                pn = p * n
                r = len(U[0])
                
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                
                T = np.zeros((n, m, p))
                for c in range(r):
                    u_c = U_arr[:, c].reshape(n, m)
                    v_c = V_arr[:, c].reshape(m, p)
                    w_c = W_arr[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
                
                return np.sum((T - 1.0)**2)
            
            current_obj = objective(U, V, W)
            
            # Local search iterations
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Randomly pick a matrix and entry to perturb
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row = np.random.randint(n*m)
                    col = np.random.randint(target_rank)
                    delta = np.random.randn() * 0.1
                    new_val = U[row, col] + delta
                    U[row, col] = new_val
                elif mat_choice == 1:
                    row = np.random.randint(m*p)
                    col = np.random.randint(target_rank)
                    delta = np.random.randn() * 0.1
                    new_val = V[row, col] + delta
                    V[row, col] = new_val
                else:
                    row = np.random.randint(p*n)
                    col = np.random.randint(target_rank)
                    delta = np.random.randn() * 0.1
                    new_val = W[row, col] + delta
                    W[row, col] = new_val
                
                new_obj = objective(U, V, W)
                
                if new_obj < current_obj:
                    current_obj = new_obj
                    if current_obj < 1e-6:
                        # Found a solution!
                        # Convert to list of lists
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        # Verify
                        if verify_fast(U_list, V_list, W_list):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            return best_result
                else:
                    # Revert
                    if mat_choice == 0:
                        U[row, col] -= delta
                    elif mat_choice == 1:
                        V[row, col] -= delta
                    else:
                        W[row, col] -= delta
    
    return best_result
# EVOLVE-BLOCK-END
