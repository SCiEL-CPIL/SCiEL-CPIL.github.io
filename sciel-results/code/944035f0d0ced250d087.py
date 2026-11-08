import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (14, rank)
    
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer before 120s limit
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 42
    
    # We know that matrix multiplication can sometimes be done with fewer multiplications.
    # For 2x3 * 3x7, the rank is at most 42.
    # Let's try to find a rank < 42.
    
    # The tensor T has entries T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, the tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition is:
    # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    # for all i, j, k.
    
    # This is a system of n*m*p = 42 equations.
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (6 + 21 + 14) = 41 * rank.
    # For rank=41, we have 41*41 = 1681 unknowns and 42 equations.
    # This is highly underdetermined.
    
    # However, the solutions must be integer or half-integer.
    # This suggests using a structured search or a specific algebraic approach.
    
    # One common approach is to use the "Kronecker" or "block" structure.
    # But here, let's try a randomized search with local optimization, specifically targeting integer solutions.
    
    # Since the problem is small, we can try to fix some variables and solve for others.
    # Or, we can use a lattice reduction approach if we can set up the equations as linear constraints on a lattice.
    # The equations are cubic: sum_l U_l V_l W_l = 1.
    
    # Let's try a different strategy:
    # We know that the rank of matrix multiplication is at least the border rank.
    # For 2x3 * 3x7, the border rank is likely 42 or close to it.
    # Actually, for general matrix multiplication, the rank is often n*m*p.
    # However, for specific small dimensions, it can be lower.
    
    # Let's check if rank 41 is possible.
    # We can try to solve the system for rank=41.
    # Since the system is nonlinear, we can use a grid search or random sampling with local refinement.
    
    # A more efficient way:
    # We can fix W to be an identity-like matrix (or permutation) to simplify the equations.
    # If we set W to be a permutation matrix of size 14x14, then W[k*n+i, l] is 1 for one specific l.
    # This reduces the problem to sum_l U[i*m+j, l] * V[j*p+k, l] = 1 for specific l.
    # This is still nonlinear if we don't fix U and V.
    
    # Let's try a randomized approach with integer constraints.
    # We will search for integer solutions in a small range.
    
    # Since the dimension is small, we can try to use the fact that the tensor is "sparse" in some sense.
    
    # Actually, a known result is that the rank of 2x3 * 3x7 is 42.
    # But let's verify if we can find a rank 41 solution.
    
    # We'll try to use a local search from a random integer initialization.
    
    def verify(U, V, W, rank):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(rank):
                        val += U[i*m + j][l] * V[j*p + k][l] * W[k*n + i][l]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try rank 41
    for target_rank in [41, 40, 39]:
        if time.time() > deadline:
            break
            
        # Random integer initialization
        # We can try to start from the schoolbook solution and perturb it.
        # Schoolbook:
        # U[i*m+j, i*m+j] = 1 (if index < rank)
        # V[j*p+k, i*m+j] = 1
        # W[k*n+i, i*m+j] = 1
        
        # But we want rank < 42.
        
        # Let's try a random search with local moves.
        # We'll represent U, V, W as lists of lists of integers (or half-integers).
        
        # Since the search space is huge, we need a smarter approach.
        
        # Idea: Use the fact that the equations are linear in each factor if the other two are fixed.
        # We can use a block coordinate descent method.
        
        # Step 1: Initialize U, V, W randomly with small integers.
        # Step 2: Fix U and V, solve for W.
        # The equation is: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1.
        # This is not linear in W because W appears multiplied by U and V.
        # Wait, if we fix U and V, the equation for each (i,j,k) is:
        # sum_l (U[i*m+j, l] * V[j*p+k, l]) * W[k*n+i, l] = 1.
        # This is a linear system in W!
        # Let A_{(i,j,k), l} = U[i*m+j, l] * V[j*p+k, l].
        # Then A * W = 1, where W is a vector of length rank for each (k,n+i).
        # Actually, W is a matrix of size (p*n, rank).
        # For a fixed (k, i), the variable is W[k*n+i, :].
        # The equation is: sum_l A_{(i,j,k), l} * W[k*n+i, l] = 1.
        # This must hold for all j.
        # So for each (k, i), we have m equations:
        # sum_l A_{(i,j,k), l} * W[k*n+i, l] = 1, for j=0..m-1.
        # This is a system of m linear equations in rank unknowns.
        # If m <= rank, we can solve for W[k*n+i, :].
        
        # Similarly, we can fix U and W, and solve for V.
        # And fix V and W, and solve for U.
        
        # This suggests a block coordinate descent algorithm:
        # 1. Initialize U, V, W.
        # 2. Fix U, V. Solve for W.
        # 3. Fix U, W. Solve for V.
        # 4. Fix V, W. Solve for U.
        # 5. Repeat until convergence.
        
        # The solutions for W, V, U will be rational numbers (if the linear systems are solvable).
        # We can then round to the nearest integer or half-integer and check if the solution is valid.
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            # We'll use small integers
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Block coordinate descent
            for iteration in range(20):
                if time.time() > deadline:
                    break
                
                # Solve for W
                # For each (k, i), solve A * W[k*n+i, :] = 1
                # A is m x rank
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            for l in range(target_rank):
                                A[j, l] = U[i*m + j, l] * V[j*p + k, l]
                        # Solve A * W_col = 1
                        # Use least squares
                        try:
                            W_col, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n + i, :] = W_col
                        except:
                            pass
                
                # Solve for V
                # For each (j, k), solve A * V[j*p+k, :] = 1
                # A is n x rank
                # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
                # For fixed (j, k), the variable is V[j*p+k, :].
                # The equation is: sum_l (U[i*m+j, l] * W[k*n+i, l]) * V[j*p+k, l] = 1
                # for all i.
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            for l in range(target_rank):
                                A[i, l] = U[i*m + j, l] * W[k*n + i, l]
                        try:
                            V_col, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p + k, :] = V_col
                        except:
                            pass
                
                # Solve for U
                # For each (i, j), solve A * U[i*m+j, :] = 1
                # A is p x rank
                # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
                # For fixed (i, j), the variable is U[i*m+j, :].
                # The equation is: sum_l (V[j*p+k, l] * W[k*n+i, l]) * U[i*m+j, l] = 1
                # for all k.
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for l in range(target_rank):
                                A[k, l] = V[j*p + k, l] * W[k*n + i, l]
                        try:
                            U_col, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m + j, :] = U_col
                        except:
                            pass
            
            # After convergence, check if the solution is close to integers or half-integers
            # and verify it.
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), target_rank):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # Also try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), target_rank):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
    # Fallback to schoolbook
    rank = n * m * p
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
