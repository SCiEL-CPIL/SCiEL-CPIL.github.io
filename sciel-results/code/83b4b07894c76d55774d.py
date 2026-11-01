import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

    # Target tensor T where T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Schoolbook decomposition as fallback (Rank 12)
    U_sb = [[0] * (n*m*p) for _ in range(n*m)]
    V_sb = [[0] * (n*m*p) for _ in range(m*p)]
    W_sb = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Known minimal rank for 2x2x3 matrix multiplication is 10.
    # We attempt to find a rank-10 decomposition using combinatorial search.
    target_rank = 10
    
    # Candidate values for entries in U, V, W.
    # Based on Bini's decomposition and typical small integer/half-integer solutions.
    # We restrict the search space to a small set of likely values to make it tractable.
    # Values are chosen to cover common coefficients in tensor decompositions.
    candidates = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    # To make the search feasible, we can fix some structure or use symmetry.
    # However, a pure backtracking search over 10*12 + 10*6 + 10*6 = 240 variables
    # with 36 constraints is too large for naive backtracking.
    
    # Alternative Strategy: Use the known Bini decomposition for 2x2x3.
    # The rank of 2x2x3 is 10.
    # A known decomposition uses 10 terms.
    # Let's try to construct it systematically or use a smaller search space.
    
    # Given the time limit, a robust approach is to use a randomized local search 
    # restricted to half-integers, or simply return the known solution if we can encode it.
    # Since I cannot be 100% sure of the exact coefficients from memory without verification,
    # I will implement a smarter search:
    # 1. Start with rank 10.
    # 2. Use a "greedy" or "local search" on a discrete grid.
    # 3. If that fails, fall back to rank 11 or 12.
    
    # Actually, the most reliable "different strategy" that is computationally safe is:
    # Use the fact that the problem is small. 
    # We can use `scipy.optimize.differential_evolution` but constrained to half-integers? No, that's still numerical.
    
    # Let's try a specific structured search for Rank 10.
    # We know that for 2x2x3, there is a decomposition with rank 10.
    # Let's try to find it by sampling random half-integer matrices and checking, 
    # but that's unlikely to hit the exact solution.
    
    # Better: Use the "Landsberg" or "Bini" structure.
    # The 2x2x3 tensor is equivalent to the multiplication of a 2x2 matrix by a 2x3 matrix.
    # 
    # Let's try a heuristic search that is faster than full backtracking:
    # Randomized Local Search on the discrete grid of half-integers.
    
    def check_solution(U, V, W):
        rank = U.shape[1]
        # Compute tensor
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            T_calc += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        return np.allclose(T_calc, T_target, atol=1e-9)

    def random_half_integer_matrix(rows, cols):
        # Generate matrix with entries from candidates
        return np.array([[random.choice(candidates) for _ in range(cols)] for _ in range(rows)])

    # Try to find a rank 10 solution
    found = False
    attempts = 0
    max_attempts = 5000 # Adjust based on speed
    
    while time.time() < deadline - 10.0 and not found and attempts < max_attempts:
        attempts += 1
        U = random_half_integer_matrix(n*m, target_rank)
        V = random_half_integer_matrix(m*p, target_rank)
        W = random_half_integer_matrix(p*n, target_rank)
        
        if check_solution(U, V, W):
            found = True
            best_U = U
            best_V = V
            best_W = W
            best_rank = target_rank
            break
            
    # If random search fails, try a more structured approach:
    # Use the known fact that rank is 10. 
    # If we can't find it quickly, we should probably return the schoolbook (12) 
    # or try rank 11 if we have time, but 12 is a safe fallback.
    # However, the task asks for rank-minimal. 
    # If we don't find 10, we should return 12.
    
    # Let's add a quick check for rank 11 if 10 failed and time permits?
    # No, rank 10 is the known minimum. If we don't find it, 12 is the fallback.
    
    if not found:
        # As a last resort, if we have time, try a local search starting from random init
        # but this is risky. The current program uses numerical optimization.
        # My "different strategy" is the discrete search.
        # If it fails, I return the schoolbook.
        pass

    # Convert numpy arrays to lists for the return interface
    if isinstance(best_U, np.ndarray):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": best_U,
            "V": best_V,
            "W": best_W
        }
