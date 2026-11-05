"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_dim = n * m * p  # 84
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Theoretical lower bounds
    # L1 = max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # L2 = ceil(n*m*p / max(n,m,p)) = ceil(84/7) = 12 (weak)
    # L3 = ceil(sqrt(n*m*p) * min(n,m,p)) ? No.
    # Known bound: rank >= max(n*m, m*p, p*n) = 28
    # So we search from 28 upwards.
    
    # Pre-compute the target tensor entries
    # T[i, j, k] = 1 if we multiply A[i,j] and B[j,k] to get C[i,k]
    # In the flattened vector space of U, V, W:
    # U has rows i*m+j (12 rows), cols r
    # V has rows j*p+k (28 rows), cols r
    # W has rows k*n+i (21 rows), cols r
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # We model this as finding sparse matrices U, V, W with rank R
    # such that the trilinear form matches.
    
    # Strategy:
    # For a fixed rank R, we try to find a decomposition.
    # This is a hard combinatorial problem.
    # However, for small matrices, we can use a greedy or backtracking approach
    # if we assume a specific structure, or we can use a randomized search
    # with local improvement.
    
    # Given the time limit (150s), we cannot do an exhaustive search over all
    # possible support patterns for R=28, 29, etc. if the space is large.
    # The number of non-zero entries in U is at most R * (number of non-zeros per column).
    # If we assume each column of U has at most 1 non-zero, each of V at most 1, etc.,
    # that's not enough.
    
    # Let's try a different approach:
    # We know that the trivial decomposition has rank 84.
    # We want to reduce this.
    # A common technique is to use the "Strassen-like" recursive decomposition
    # or specific algebraic identities.
    
    # For 3x4 * 4x7, there are no known Strassen-like improvements that are
    # trivial to implement without deep algebraic geometry knowledge.
    
    # However, we can try to find a decomposition of rank R by solving a system
    # of equations.
    # Let the rank be R.
    # We have R columns in U, V, W.
    # Let u_r, v_r, w_r be the r-th column vectors.
    # The condition is:
    # for all i, j, k: sum_{r=1}^R u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # This is a system of 84 equations (for each i,j,k) in R * (12+28+21) = R*61 variables.
    # This is highly underdetermined for small R, but non-linear (trilinear).
    
    # We can linearize by fixing two of the matrices.
    # For example, if we fix U and V, the equations become linear in W.
    # But finding U and V is the hard part.
    
    # Alternative: Randomized search with refinement.
    # 1. Initialize U, V, W with random sparse entries.
    # 2. Optimize W to minimize error.
    # 3. Optimize V to minimize error.
    # 4. Optimize U to minimize error.
    # 5. Repeat.
    # 6. Check if error is 0.
    
    # Given the strict requirement for EXACT integer/half-integer entries,
    # floating point optimization is risky.
    
    # Let's stick to the trivial solution if we can't find a better one quickly.
    # But the task asks for a "stronger algorithm".
    
    # Let's try to implement a simple backtracking search for small ranks
    # assuming a very sparse structure (e.g., each column of U, V, W has
    # only 1 non-zero entry). This corresponds to the trivial decomposition
    # essentially, but maybe we can find a better one?
    # No, if each column has 1 non-zero in U, V, W, then each term in the sum
    # is a product of 3 single entries. This is exactly the trivial decomposition
    # if we match them up correctly.
    
    # What if we allow 2 non-zeros per column?
    # This increases the search space significantly.
    
    # Given the time constraint, let's just return the trivial solution
    # unless we can find a quick win.
    # Actually, let's try to use the `sympy` or `scipy` if available?
    # The prompt says NumPy and SciPy are available. Sympy is not explicitly mentioned
    # but "Python standard library" is.
    
    # Let's try a randomized search for rank 28-30 with sparse patterns.
    # We will generate random sparse matrices U, V, W and solve for the best fit.
    
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize best with trivial
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
                
    # Try to find a better decomposition
    # We'll try ranks from 28 to 35
    for rank in range(28, 36):
        if time.time() > deadline:
            break
            
        # Try a few random sparse patterns
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Generate random sparse U, V, W
            # Each column has a small number of non-zeros
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Randomly place non-zeros
            # Let's say each column has 1-2 non-zeros
            for r in range(rank):
                # U
                nz_u = np.random.choice(n * m, size=min(2, n*m), replace=False)
                for idx in nz_u:
                    U[idx, r] = 1
                # V
                nz_v = np.random.choice(m * p, size=min(2, m*p), replace=False)
                for idx in nz_v:
                    V[idx, r] = 1
                # W
                nz_w = np.random.choice(p * n, size=min(2, p*n), replace=False)
                for idx in nz_w:
                    W[idx, r] = 1
            
            # Now we have fixed U, V, W.
            # We need to check if there exist scalars such that the product matches.
            # Wait, the entries are fixed to 1. We can scale the columns.
            # Let u_r, v_r, w_r be the base vectors.
            # We want sum_r alpha_r * beta_r * gamma_r * u_r v_r^T w_r^T = T
            # This is still non-linear.
            
            # Instead, let's fix U and V, and solve for W.
            # For each (i,k), the entry C[i,k] = sum_j A[i,j] B[j,k]
            # In our tensor: T[i,j,k] = 1
            # The contribution of rank r to T[i,j,k] is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Let's build the linear system for W.
            # For each (i,j,k), we have:
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
            # Let A_{(i,j,k), r} = U[i*m+j, r] * V[j*p+k, r]
            # Then A @ W_flat = 1
            # But W is not a flat vector, it's a matrix.
            # The variable is W[k*n+i, r].
            # The equation for (i,j,k) involves only W[k*n+i, :].
            # So for each i,k, we have a system of m equations (for j=0..m-1)
            # with rank variables W[k*n+i, 0..rank-1].
            
            # Let's check if this system is solvable.
            solvable = True
            W_sol = np.zeros((p * n, rank))
            
            for i in range(n):
                for k in range(p):
                    # Variables: W[k*n+i, 0..rank-1]
                    # Equations: for j in 0..m-1
                    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    
                    A_sub = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A_sub[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    b_sub = np.ones(m)
                    
                    # Solve A_sub @ x = b_sub
                    # Least squares if not square
                    try:
                        x, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                        # Check if exact
                        if np.allclose(A_sub @ x, b_sub):
                            W_sol[k*n+i, :] = x
                        else:
                            solvable = False
                            break
                    except:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if solvable:
                # We found a solution!
                # Check if entries are integer/half-integer
                # Convert to list of lists
                U_list = [[int(U[i, r]) if abs(U[i, r] - round(U[i, r])) < 1e-9 else 0 for r in range(rank)] for i in range(n*m)]
                V_list = [[int(V[i, r]) if abs(V[i, r] - round(V[i, r])) < 1e-9 else 0 for r in range(rank)] for i in range(m*p)]
                W_list = [[int(W_sol[i, r]) if abs(W_sol[i, r] - round(W_sol[i, r])) < 1e-9 else 0 for r in range(rank)] for i in range(p*n)]
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                            if s != 1:
                                valid = False
                                break
                    if not valid: break
                if not valid: break
                
                if valid:
                    best_rank = rank
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    break # Found a valid one for this rank, try lower? No, we are going up.
                    # Actually we start from 28. If we find one at 28, we are done.
                    # If not, we continue to 29, etc.
        
        if best_rank == rank:
            break # Found the minimal rank in our search range
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
