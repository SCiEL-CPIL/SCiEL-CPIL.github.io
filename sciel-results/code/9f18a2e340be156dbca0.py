import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i else 0
    # I in [0, n*m), J in [0, m*p), K in [0, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1
    
    # Try to find a decomposition with rank R < rank_schoolbook
    # We try ranks from 25 to 49
    # We use a random search for U, V with entries in {-1, 0, 1}
    # Then solve for W? Or check if a solution exists?
    # Solving for W is linear: sum_r U[I,r] V[J,r] W[K,r] = T[I,J,K]
    # This is a system of equations.
    
    start_time = time.time()
    time_limit = 100  # Leave margin
    
    for R in range(25, min(rank_schoolbook, 50)):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random instances for this R
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random sparse U and V
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            
            # Randomly assign non-zero values
            # Each column of U and V should have some non-zeros
            for r in range(R):
                # Pick random rows for U
                u_rows = np.random.choice(n*m, size=min(3, n*m), replace=False)
                for idx in u_rows:
                    U[idx, r] = random.choice([-1, 1])
                    
                # Pick random rows for V
                v_rows = np.random.choice(m*p, size=min(3, m*p), replace=False)
                for idx in v_rows:
                    V[idx, r] = random.choice([-1, 1])
            
            # Now we have U and V. We need to find W such that
            # sum_r U[I,r] V[J,r] W[K,r] = T[I,J,K]
            # This is equivalent to:
            # For each K, W[K, :] is a vector.
            # The equation is linear in W.
            # Let M[I, J, K, r] = U[I, r] * V[J, r]
            # Then sum_r M[I, J, K, r] W[K, r] = T[I, J, K]
            
            # We can try to solve this using least squares or exact integer check?
            # Since we want exact integers, we can try to solve over rationals?
            # Or just check if there is an integer solution?
            # Given the small size, we can try to solve the linear system for W
            # for each K? No, W[K, r] couples all I, J.
            
            # Let's form the matrix A_sys where rows are (I, J, K) and cols are (K, r)?
            # No, the unknowns are W[K, r].
            # The equation is: sum_r (U[I, r] V[J, r]) W[K, r] = T[I, J, K]
            # This is a linear equation in the variables W[K, r].
            # Let's collect all equations.
            # There are n*m*m*p*p*n = 2500 equations.
            # There are p*n*R = 10*R unknowns.
            
            # We can use numpy to solve the least squares problem and check if the residual is 0
            # and if the solution is integer.
            
            # Construct the system matrix
            # Rows: I, J, K. Cols: K_idx, r_idx.
            # But the coefficient of W[K, r] is U[I, r] * V[J, r] ONLY if the row index K matches the K in W[K, r].
            # So for a fixed (I, J), the equation involves W[K, r] for all K?
            # No, the term is W[K, r] where K is the index of the output.
            # So for a fixed (I, J), we have one equation for each K?
            # No, T[I, J, K] is the target for that specific K.
            # So for each (I, J, K), we have:
            # sum_r U[I, r] V[J, r] W[K, r] = T[I, J, K]
            
            # This means that for a fixed K, the variables are W[K, 0], ..., W[K, R-1].
            # And the equations involving W[K, :] are those where the third index is K.
            # So we can solve for W[K, :] independently for each K?
            # Yes! Because W[K, r] only appears in equations where the output index is K.
            
            # So for each K in range(p*n):
            #   We have equations: sum_r U[I, r] V[J, r] W[K, r] = T[I, J, K] for all I, J.
            #   This is a linear system for W[K, :].
            #   The matrix is of size (n*m) x R.
            #   A_sys[I, r] = U[I, r] * V[J, r]? No, V depends on J.
            #   So the system is:
            #   sum_r (U[I, r] V[J, r]) W[K, r] = T[I, J, K]
            #   This is NOT a linear system in W[K, r] with constant coefficients because V[J, r] depends on J.
            #   Wait, for a fixed K, the equations are for all I, J.
            #   The unknowns are W[K, r].
            #   The coefficient of W[K, r] in the equation for (I, J, K) is U[I, r] * V[J, r].
            #   This coefficient depends on I and J.
            #   So we have a system:
            #   For each (I, J): sum_r (U[I, r] V[J, r]) W[K, r] = T[I, J, K]
            #   This is a set of n*m equations for R unknowns W[K, r].
            #   But the coefficients depend on I and J.
            #   This is not a single linear system for W[K, :].
            #   It's a set of equations that must hold simultaneously.
            
            # Actually, we can view this as:
            # Let M_IJ be the vector (U[I, r] V[J, r])_{r=1..R}.
            # Then M_IJ dot W[K, :] = T[I, J, K].
            # This must hold for all I, J.
            # So we need to find W[K, :] such that for all I, J:
            # (U[I, :] * V[J, :]) dot W[K, :] = T[I, J, K]
            
            # This is a system of n*m equations for R unknowns.
            # We can try to solve this using least squares and check if it's exact.
            
            valid = True
            W = np.zeros((p*n, R), dtype=float)
            
            for K in range(p*n):
                # Collect equations for this K
                # Equations: (U[I, :] * V[J, :]) dot W[K, :] = T[I, J, K]
                # There are n*m such equations.
                A_sys = np.zeros((n*m, R))
                b_sys = np.zeros(n*m)
                
                idx = 0
                for I in range(n*m):
                    for J in range(m*p):
                        # Coefficients
                        A_sys[idx, :] = U[I, :] * V[J, :]
                        b_sys[idx] = T[I, J, K]
                        idx += 1
                
                # Solve A_sys @ W[K, :] = b_sys
                # Use least squares
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    W[K, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if the solution is integer (or close to it)
            # And if it exactly satisfies the equations
            # Since we used lstsq, it might not be exact.
            # Check residuals
            max_err = 0
            for K in range(p*n):
                for I in range(n*m):
                    for J in range(m*p):
                        lhs = np.dot(U[I, :] * V[J, :], W[K, :])
                        rhs = T[I, J, K]
                        err = abs(lhs - rhs)
                        if err > max_err:
                            max_err = err
                        if err > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check if W is integer
                W_int = np.rint(W)
                if np.allclose(W, W_int, atol=1e-6):
                    # Return the solution
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_int.tolist()
                    }
    
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
