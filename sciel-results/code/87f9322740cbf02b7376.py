import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer before 120s limit
    
    def verify(U, V, W, rank):
        """
        U: (nm, rank)
        V: (mp, rank)
        W: (pn, rank)
        Check if sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] == 1 for all i,j,k
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # We can vectorize the check.
        # For each l, we have a term T_l[i, j, k] = U[i, j, l] * V[j, k, l] * W[k, i, l]
        # Sum over l should be 1.
        
        # Reshape to 3D for easier indexing?
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # Result[i, j, k] = sum_l U_3d[i, j, l] * V_3d[j, k, l] * W_3d[k, i, l]
        # This is a tensor contraction.
        
        # For small sizes, a loop is fine and less memory intensive than creating full tensors if rank is large.
        # But let's try to be efficient.
        # Since n=2, m=3, p=7, total elements 42.
        # Rank <= 42.
        
        res = np.zeros((n, m, p))
        for l in range(rank):
            # u_l shape (n, m)
            # v_l shape (m, p)
            # w_l shape (p, n)
            u_l = U_3d[:, :, l]
            v_l = V_3d[:, :, l]
            w_l = W_3d[:, :, l]
            
            # term[i, j, k] = u_l[i, j] * v_l[j, k] * w_l[k, i]
            # We can compute this element-wise.
            # u_l[:, :, None] * v_l[:, None, :] * w_l[None, :, :]
            # Wait, indices:
            # u_l[i, j]
            # v_l[j, k]
            # w_l[k, i]
            
            # Let's just loop over i, j, k for simplicity and correctness first, 
            # as 42 * rank is very small.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u_l[i, j] * v_l[j, k] * w_l[k, i]
                        
        if np.allclose(res, 1.0, atol=1e-9):
            return True
        return False

    def to_half_int(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2.0

    def to_int(x):
        """Round to nearest integer."""
        return np.round(x)

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Strategy:
    # 1. Try to find a solution for rank < 42.
    # 2. Use block coordinate descent.
    #    Fix U, V -> solve for W (linear)
    #    Fix U, W -> solve for V (linear)
    #    Fix V, W -> solve for U (linear)
    # 3. The solutions will be rational. We need them to be half-integers.
    # 4. We can initialize with random half-integers and refine.
    
    # Let's iterate over target ranks from 41 down to some lower bound.
    # The border rank of 2x3 * 3x7 is likely 42, but let's check.
    # Actually, for matrix multiplication, the rank is often n*m*p.
    # However, there are known cases where it is lower.
    # For 2x2 * 2x2, rank is 7 < 8.
    # For 2x3 * 3x7, is it 42?
    # Let's try to find a rank 41 solution.
    
    # We will use a random restart strategy with block coordinate descent.
    
    for target_rank in range(41, 20, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Number of restarts
        num_restarts = 50
        if time.time() > deadline - 20:
            num_restarts = 10
            
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization with small half-integers
            # Using 0.5 * integers
            U = 0.5 * np.random.randint(-4, 5, size=(nm, target_rank))
            V = 0.5 * np.random.randint(-4, 5, size=(mp, target_rank))
            W = 0.5 * np.random.randint(-4, 5, size=(pn, target_rank))
            
            # Block Coordinate Descent
            # We want to minimize the error, but we want an EXACT solution.
            # So we solve the linear systems exactly.
            # If the linear system has a unique solution, we get a rational number.
            # We then check if it is close to a half-integer.
            
            for iteration in range(50):
                if time.time() > deadline - 2:
                    break
                    
                # 1. Solve for W
                # For each (k, i), we have m equations:
                # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
                # Let A_W[k*n+i, j, l] = U[i*m+j, l] * V[j*p+k, l]
                # Then for fixed (k, i), we solve A_W[k*n+i, :, :] * W[k*n+i, :] = 1
                # A_W[k*n+i] is (m, target_rank)
                
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        # Solve A * W_col = 1
                        # Use lstsq for stability
                        try:
                            W_col, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[idx_w, :] = W_col
                        except:
                            pass

                # 2. Solve for V
                # For each (j, k), we have n equations:
                # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
                # Let A_V[j*p+k, i, l] = U[i*m+j, l] * W[k*n+i, l]
                # Solve A_V[j*p+k] * V[j*p+k, :] = 1
                
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            V_col, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[idx_v, :] = V_col
                        except:
                            pass

                # 3. Solve for U
                # For each (i, j), we have p equations:
                # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
                # Let A_U[i*m+j, k, l] = V[j*p+k, l] * W[k*n+i, l]
                # Solve A_U[i*m+j] * U[i*m+j, :] = 1
                
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        try:
                            U_col, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[idx_u, :] = U_col
                        except:
                            pass

            # After convergence, check if the solution is close to half-integers
            U_hi = to_half_int(U)
            V_hi = to_half_int(V)
            W_hi = to_half_int(W)
            
            if verify(U_hi, V_hi, W_hi, target_rank):
                best_rank = target_rank
                best_U = U_hi
                best_V = V_hi
                best_W = W_hi
                found = True
                break
            
            # Also try integer rounding
            U_int = to_int(U)
            V_int = to_int(V)
            W_int = to_int(W)
            
            if verify(U_int, V_int, W_int, target_rank):
                best_rank = target_rank
                best_U = U_int
                best_V = V_int
                best_W = W_int
                found = True
                break
                
        if found:
            break

    # Convert best solution to list of lists
    def mat_to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {
        "U": mat_to_list(best_U),
        "V": mat_to_list(best_V),
        "W": mat_to_list(best_W)
    }
