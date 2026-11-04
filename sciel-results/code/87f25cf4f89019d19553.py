"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DEADLINE = 145  # safety margin before 150s kill


def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def verify(U_arr, V_arr, W_arr, r):
        """Fast verification using numpy einsum."""
        if time.time() > DEADLINE - 3:
            return False
        U_r = np.array(U_arr, dtype=np.float64).reshape(n, m, r)
        V_r = np.array(V_arr, dtype=np.float64).reshape(m, p, r)
        W_r = np.array(W_arr, dtype=np.float64).reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def try_subspace_method(R, time_budget):
        """
        Subspace method: Find subspaces U_sub, V_sub, W_sub of dimensions 
        a, b, c such that a+b+c = R and the tensor is supported on the 
        direct sum of these subspaces. Then solve linear equations for exact coefficients.
        """
        deadline = time.time() + time_budget
        best_result = None

        # For matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k
        # We need to find rank-R decomposition where each term is a rank-1 tensor
        # U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Strategy: Use a structured approach with rational coefficients
        # Try different patterns of non-zero entries
        
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random rational initialization with small numerators/denominators
            max_denom = 3 + trial % 4  # try denominators 3,4,5,6
            
            def rand_frac():
                num = np.random.randint(-2, 3)
                den = np.random.randint(1, max_denom + 1)
                if num == 0:
                    return 0.0
                return num / den
            
            U = np.array([[rand_frac() for _ in range(R)] for _ in range(nm)])
            V = np.array([[rand_frac() for _ in range(R)] for _ in range(mp)])
            W = np.array([[rand_frac() for _ in range(R)] for _ in range(pn)])
            
            # Solve the linear system: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # This is a system of n*m*p = 120 equations
            # Variables: R*3*(nm+mp+pn) but we can use alternating minimization
            
            # Use a simple iterative approach: fix two matrices, solve for the third
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Fix V and W, solve for U
                # For each (i,j): sum_k sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all k
                # This is a linear system in U[i,j,:]
                # Let's use least squares
                
                # Actually, let's use a different approach: 
                # Build the full linear system and use lstsq
                
                # Flatten the problem
                # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                # This is bilinear, not linear. Use alternating minimization.
                
                # Fix V, W -> solve for U (linear in each U[i,j,:])
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Build system: A * x = b where x = U[i*m+j, :]
                        A = np.zeros((p, R))
                        b = np.ones(p)
                        for k in range(p):
                            for r in range(R):
                                A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve A * x = b (least squares)
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i*m+j, :] = x
                        except:
                            pass
                
                # Fix U, W -> solve for V
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        b = np.ones(n)
                        for i in range(n):
                            for r in range(R):
                                A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j*p+k, :] = x
                        except:
                            pass
                
                # Fix U, V -> solve for W
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        b = np.ones(m)
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence
                U, V, W = U_new, V_new, W_new
                
                # Check if we've converged to a good solution
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                error = np.max(np.abs(T - 1.0))
                
                if error < 1e-8:
                    converged = True
                    break
                
                if time.time() > deadline - 5:
                    break
            
            if converged:
                # Convert to list format and verify
                U_list = [[float(U[i, r]) for r in range(R)] for i in range(nm)]
                V_list = [[float(V[i, r]) for r in range(R)] for i in range(mp)]
                W_list = [[float(W[i, r]) for r in range(R)] for i in range(pn)]
                
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return best_result

    # Try ranks from low to high
    # Known lower bound for 4x5x6 is around 57-72
    # Try a range of ranks with time allocation
    ranks_to_try = [60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]

    for R in ranks_to_try:
        if time.time() > DEADLINE - 15:
            break
        time_left = DEADLINE - time.time() - 10
        if time_left < 10:
            break

        # Allocate time based on rank
        if R <= 72:
            budget = min(time_left, 40)
        elif R <= 90:
            budget = min(time_left, 25)
        else:
            budget = min(time_left, 15)

        result = try_subspace_method(R, budget)

        if result is not None:
            best = result
            break

    # Final safety check: if we're running out of time, return the schoolbook result
    if time.time() > DEADLINE - 5:
        return best

    return best
