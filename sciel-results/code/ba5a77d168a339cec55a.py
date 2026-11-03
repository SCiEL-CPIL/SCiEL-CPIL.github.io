"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    n, m, p = DIMENSIONS
    # Matrix multiplication tensor T[i,j,k] = delta_{i,k} (for A[i,:] * B[:,j] -> C[i,j])
    # Wait, standard matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor T_{i,j,k} such that T[i,j,k] = 1 if we are multiplying A[i,k] and B[k,j].
    # The decomposition is C[i,j] = sum_r U[i,r] * V[r,j] * W[r,k]? No.
    # Standard CP decomposition of matrix multiplication tensor T_{i,j,k}:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where U is n x R, V is m x R, W is p x R?
    # Let's check the indices in the prompt:
    # U shape: (n*m, rank) -> U[i*m + j, r]
    # V shape: (m*p, rank) -> V[j*p + k, r]
    # W shape: (p*n, rank) -> W[k*n + i, r]
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal the matrix multiplication tensor entry.
    # For matrix multiplication C = A B, C[i, j'] = sum_k A[i, k] B[k, j'].
    # Here the output index is (i, j') where j' is the column of B.
    # The input indices are (i, k) for A and (k, j') for B.
    # The tensor T has indices (i, j', k) where T[i, j', k] = 1 if the term A[i,k]B[k,j'] contributes to C[i,j'].
    # Actually, T[i, j', k] = 1 for all i, j', k? No.
    # The tensor for matrix multiplication M_{n,m,p} has entries T_{i, j, k} where i in 0..n-1, j in 0..p-1, k in 0..m-1.
    # T_{i, j, k} = 1.
    # The decomposition is: T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} ?
    # Let's look at the seed:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # This corresponds to r = (i, j, k) in the schoolbook case.
    # So U is indexed by (i, j), V by (j, k), W by (k, i).
    # The product is sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}.
    # This sum must equal 1 if (i, j, k) is a valid term in the multiplication, else 0.
    # Wait, the matrix multiplication tensor is T_{i, j, k} = 1 for ALL i, j, k?
    # No, C_{i, j} = sum_k A_{i, k} B_{k, j}.
    # The tensor T is defined such that the bilinear form is sum_{i,j,k} T_{i,j,k} A_{i,k} B_{k,j}.
    # So T_{i,j,k} = 1 for all i, j, k.
    # Let's verify: sum_{i,j,k} 1 * A_{i,k} B_{k,j} = sum_k (sum_i A_{i,k}) (sum_j B_{k,j})? No.
    # The indices in the sum are:
    # sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} A_{i, k} B_{j, ?}
    # This is getting confusing. Let's stick to the prompt's definition.
    # "exactly decompose the matrix multiplication tensor"
    # The tensor T has dimensions n x p x m (or similar).
    # Entry T[i, j, k] = 1.
    # Decomposition: T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    # Wait, the indices in U, V, W in the prompt are:
    # U: (n*m, rank) -> index (i, j) where i is row of A, j is col of A (which is row of B)?
    # Let's assume the standard indices:
    # A is n x m, B is m x p. C is n x p.
    # C[i, j] = sum_k A[i, k] B[k, j].
    # The tensor T has indices (i, j, k). T[i, j, k] = 1.
    # The decomposition is T[i, j, k] = sum_r U[i, k, r] V[k, j, r] W[i, j, r]?
    # Let's look at the seed again carefully.
    # U[i * m + j, c] = 1. Index is (i, j).
    # V[j * p + k, c] = 1. Index is (j, k).
    # W[k * n + i, c] = 1. Index is (k, i).
    # The sum is over r (c).
    # The term is U[i, j, r] V[j, k, r] W[k, i, r].
    # If we set A and B to identity matrices, we can test.
    # Actually, the "matrix multiplication tensor" T is the structure tensor.
    # T_{i, j, k} = 1 if the product A_{i, k} B_{k, j} appears in C_{i, j}.
    # So T_{i, j, k} = 1 for all i, j, k.
    # The decomposition must satisfy:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    # This is a very specific system of equations.
    
    # Let's verify this interpretation with the schoolbook solution.
    # Rank = n*m*p.
    # For each (i, j, k), there is a unique r.
    # U[i, j, r_{ijk}] = 1, others 0.
    # V[j, k, r_{ijk}] = 1, others 0.
    # W[k, i, r_{ijk}] = 1, others 0.
    # Then sum_r U[i, j, r] V[j, k, r] W[k, i, r] = U[i, j, r_{ijk}] V[j, k, r_{ijk}] W[k, i, r_{ijk}] = 1 * 1 * 1 = 1.
    # And for any other r, at least one factor is 0.
    # So yes, the condition is:
    # For all i in [0, n), j in [0, m), k in [0, p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We want to minimize rank R.
    # Strategy:
    # 1. Try small R.
    # 2. Randomly initialize U, V with small integers (e.g., -1, 0, 1).
    # 3. Solve for W? No, it's a trilinear system.
    #    However, if we fix U and V, the equations are linear in W?
    #    No, the equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    #    For a fixed (k, i), the unknowns are W[k,i, r] for r=0..R-1.
    #    But the coefficients U[i,j,r] V[j,k,r] depend on j.
    #    So for a fixed (k, i), we have a system of m equations (one for each j) with R unknowns (W[k,i,r]).
    #    Wait, the equation is for ALL i, j, k.
    #    Let's group by (k, i).
    #    For fixed k, i, the equations are:
    #    sum_{j=0}^{m-1} (sum_{r=0}^{R-1} U[i,j,r] V[j,k,r] W[k,i,r]) = 1? No.
    #    The equation is:
    #    sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #    This must hold for EVERY (i, j, k).
    #    So for a fixed (i, k), we have m equations (one for each j) involving W[k*n+i, r].
    #    Let's write it as:
    #    For each i, k:
    #       Let w = W[k*n+i, :] (vector of length R).
    #       For each j:
    #          sum_r (U[i*m+j, r] * V[j*p+k, r]) * w[r] = 1
    #    This is a linear system A_{i,k} w = 1_vector (length m).
    #    If R >= m and A_{i,k} has full rank, we can solve for w.
    #    But we need the SAME w to work for all i, k? No, w depends on (k, i).
    #    So we can solve for W independently for each (k, i) pair!
    #    There are n*p pairs of (k, i).
    #    For each (k, i), we solve a linear system of m equations with R unknowns.
    #    If R < m, it's overdetermined and likely inconsistent.
    #    If R >= m, it might be solvable.
    #    But we also need the entries of U, V, W to be "integer/half-integer".
    #    And we want to minimize R.
    #    
    #    Key Insight:
    #    If we fix U and V, W is determined (if the linear systems are consistent).
    #    We can check if the solution W has rational/integer entries.
    #    Since U, V are integers, the system A w = 1 has integer coefficients.
    #    If A is invertible (or has a unique solution), w will be rational.
    #    We need w to be half-integers or integers.
    #    
    #    Algorithm:
    #    1. Iterate R from lower bound to upper bound.
    #    2. Lower bound: max(n, m, p) = 6? Or maybe higher.
    #       Actually, the rank of matrix multiplication is known to be at least something.
    #       For 2x5x6, schoolbook is 60.
    #       Let's try R from 10 upwards.
    #    3. For each R:
    #       a. Generate random U (n*m x R) and V (m*p x R) with entries in {-1, 0, 1}.
    #       b. For each (k, i) in range(p) x range(n):
    #          - Construct matrix A of size m x R where A[j, r] = U[i*m+j, r] * V[j*p+k, r].
    #          - Solve A w = 1 (vector of ones of length m).
    #          - If no solution or solution has non-rational entries (shouldn't happen if A is rational), skip.
    #          - Check if w entries are half-integers (i.e., 2*w are integers).
    #       c. If all (k, i) pairs yield valid W vectors, we have a candidate.
    #       d. Verify the full decomposition exactly using rational arithmetic.
    #       e. If valid, return U, V, W.
    #    
    #    Optimization:
    #    - Use numpy for speed, but verify with fractions.
    #    - The number of (k, i) pairs is n*p = 12.
    #    - The size of A is 5 x R.
    #    - Solving 12 linear systems of size 5 x R is fast.
    #    
    #    Let's refine the "half-integer" check.
    #    The prompt says "integer/half-integer entries".
    #    So 2*W must be integer.
    #    
    #    What if R is small?
    #    If R < m (5), the system A w = 1 is overdetermined.
    #    We can use least squares, but we need an EXACT solution.
    #    So we should check if 1 is in the column space of A.
    #    This is equivalent to rank(A) == rank([A | 1]).
    #    
    #    Let's implement this search.
    
    def is_half_integer(v, tol=1e-9):
        # Check if 2*v is close to an integer
        return all(abs(2*xi - round(2*xi)) < tol for xi in v)
    
    def verify_decomposition(U, V, W, n, m, p):
        # Exact verification using Fraction
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    best_result = None
    best_rank = n * m * p
    
    # Try to find a lower rank
    # Start from a reasonable lower bound
    # Theoretical lower bound is max(n, m, p) = 6, but likely higher.
    # Let's try R from 6 to 30.
    
    for R in range(6, 31):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            # Random U and V with entries in {-1, 0, 1}
            # To increase chance of solvability, maybe use {-1, 1} or dense.
            # Let's use random integers in {-1, 0, 1}
            U_rand = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n*m)])
            V_rand = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m*p)])
            
            # Check if we can solve for W
            W_solution = {}
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A (m x R)
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U_rand[i*m+j, r] * V_rand[j*p+k, r]
                    
                    # Solve A w = 1
                    # Check consistency
                    # Use np.linalg.lstsq or check rank
                    try:
                        # If R >= m, use pinv or lstsq
                        # But we need exact integer/half-integer solution
                        # Let's try to find an exact rational solution
                        # If R < m, it's overdetermined
                        # Use fractions for exactness? Too slow for search.
                        # Use numpy first to find a candidate, then verify.
                        
                        if R >= m:
                            # Check if 1 is in column space
                            # A w = 1
                            # If A has full row rank, there are many solutions.
                            # We want one with small entries (half-integers).
                            # Let's find the minimum norm solution using pinv
                            w = np.linalg.pinv(A) @ np.ones(m)
                            
                            # Check if w is close to half-integers
                            if is_half_integer(w):
                                W_solution[(k, i)] = w
                            else:
                                # Try to find another solution?
                                # The general solution is w = w_p + N z
                                # where N is null space of A.
                                # We can try to search for z such that w is half-integer.
                                # This is a lattice problem.
                                # For now, just skip if pinv doesn't work.
                                valid = False
                                break
                        else:
                            # Overdetermined
                            # Check if rank(A) == rank([A|1])
                            A_aug = np.column_stack((A, np.ones(m)))
                            if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A_aug):
                                valid = False
                                break
                            else:
                                # Find a solution
                                w, residuals, rank, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                                if is_half_integer(w):
                                    W_solution[(k, i)] = w
                                else:
                                    valid = False
                                    break
                    except Exception:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Construct W matrix
            W_mat = np.zeros((p*n, R))
            for (k, i), w in W_solution.items():
                W_mat[k*n+i, :] = w
                
            # Convert to list of lists of Fractions for verification
            # But first, snap to half-integers
            U_int = [[int(round(x)) for x in row] for row in U_rand]
            V_int = [[int(round(x)) for x in row] for row in V_rand]
            W_int = [[int(round(2*x))/2 for x in row] for row in W_mat]
            
            # Verify exactly
            if verify_decomposition(U_int, V_int, W_int, n, m, p):
                best_result = {"U": U_int, "V": V_int, "W": W_int}
                best_rank = R
                break  # Found a valid one for this R, try smaller R? No, R is increasing.
                # Wait, I'm iterating R from 6 upwards. So the first one found is the best.
                # But I should break out of the R loop too.
                
            # If not valid, continue searching
            
        if best_result is not None:
            break
            
    if best_result is None:
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
    else:
        return best_result

# EVOLVE-BLOCK-END
