"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition using exact arithmetic
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find a lower rank decomposition (e.g., rank 83)
    # We use a continuous optimization approach to find a solution,
    # then round to half-integers and verify.
    
    def try_find_rank(r):
        if time.time() > deadline - 10.0:
            return None
            
        # Flatten indices for numpy
        # U is (N, r), V is (M, r), W is (P, r)
        # We need to solve sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Initialize with random values
        # We can also try initializing with schoolbook terms for the first r columns
        # and zeroing the rest, then optimizing.
        
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            # Scale values to be reasonable (e.g., -2 to 2)
            U_init = np.random.uniform(-2, 2, (N, r))
            V_init = np.random.uniform(-2, 2, (M, r))
            W_init = np.random.uniform(-2, 2, (P, r))
            
            # Define the loss function
            def loss(params):
                # params is a flat array of size N*r + M*r + P*r
                size_U = N * r
                size_V = M * r
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U + size_V]
                W_flat = params[size_U + size_V:]
                
                U_mat = U_flat.reshape(N, r)
                V_mat = V_flat.reshape(M, r)
                W_mat = W_flat.reshape(P, r)
                
                # Calculate T_ijk = sum_c U_ij,c * V_jk,c * W_ki,c
                # We need to sum over c for each i, j, k
                # Efficient way:
                # For each i, j, we have a vector u_ij (length r)
                # For each j, k, we have a vector v_jk (length r)
                # For each k, i, we have a vector w_ki (length r)
                # We want sum_c u_ij[c] * v_jk[c] * w_ki[c] = dot(u_ij, v_jk * w_ki)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        u_vec = U_mat[i*m + j]
                        for k in range(p):
                            v_vec = V_mat[j*p + k]
                            w_vec = W_mat[k*n + i]
                            # Element-wise product of v and w, then dot with u
                            prod = v_vec * w_vec
                            s = np.dot(u_vec, prod)
                            err += (s - 1.0) ** 2
                return err

            # We can use gradient descent or scipy
            # Since the problem is underdetermined, it should be easy to find a zero.
            # Let's try a few random restarts with a simple solver.
            # scipy.optimize.minimize might be slow if called many times.
            # Let's try a simple iterative refinement or just check if a random guess works? No.
            
            # Use L-BFGS-B
            try:
                res = minimize(loss, np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]), 
                               method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
                
                if res.success and res.fun < 1e-6:
                    # Extract solution
                    size_U = N * r
                    size_V = M * r
                    U_sol = res.x[:size_U].reshape(N, r)
                    V_sol = res.x[size_U:size_U + size_V].reshape(M, r)
                    W_sol = res.x[size_U + size_V:].reshape(P, r)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Convert to lists of Fractions for verification
                    U_list = [[Fraction(x) for x in row] for row in U_round]
                    V_list = [[Fraction(x) for x in row] for row in V_round]
                    W_list = [[Fraction(x) for x in row] for row in W_round]
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
                
        return None

    # Try rank 83
    if best_rank > 83:
        res83 = try_find_rank(83)
        if res83 is not None:
            best_result = res83
            best_rank = 83

    # If time permits, try 82? Unlikely to succeed quickly.
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
