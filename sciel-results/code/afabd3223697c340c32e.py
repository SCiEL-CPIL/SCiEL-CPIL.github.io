"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i, j, k] = 1 for all i,j,k 
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The seed program gives rank = n*m*p = 2*4*6 = 48
    # We want to find a lower rank.
    
    # Strategy: 
    # We know that the rank of matrix multiplication tensor <n,m,p> is at most n*m*p.
    # For small dimensions, we can search for lower ranks using structured linear algebra.
    # 
    # Let's try to fix U and V to have specific sparse structures and solve for W.
    # However, a more robust approach for exact minimal rank is to use the 
    # "border rank" or "tensor rank" properties.
    # 
    # Actually, for (2,4,6), the trivial rank is 48.
    # Is it possible to go lower?
    # The lower bound is max(n*m, m*p, n*p) = max(8, 24, 12) = 24.
    # But this is a weak bound.
    # 
    # Let's use a specific algorithm: 
    # 1. Assume a rank R.
    # 2. We can try to find a decomposition where U and V are "generic" enough 
    #    to make the system solvable, or use specific patterns.
    #
    # A powerful technique for small tensors is to fix one matrix (say U) to be 
    # a partial identity or a structured sparse matrix, and then the problem 
    # becomes finding V and W.
    #
    # However, the most reliable way to find a *better* rank than 48 for such 
    # small dimensions is to check if a rank R < 48 solution exists using 
    # randomized linear algebra over rationals (or reals) and then verifying.
    #
    # Let's try to find a solution with rank R.
    # We can formulate this as:
    # For a fixed R, does there exist U, V, W such that the tensor equation holds?
    # This is a system of polynomial equations.
    #
    # Alternative: Use the fact that if we fix U and V, W is determined linearly 
    # IF we treat the equation as linear in W for each (i,k) given the sums over j?
    # No, the sum is over r.
    #
    # Let's try a specific heuristic:
    # The rank of <n, m, p> is often equal to n*m*p for "generic" small cases unless 
    # there's a factorization.
    # 2, 4, 6. 
    # 4 = 2*2.
    # Can we view the multiplication as a combination of smaller multiplications?
    #
    # Let's try to construct a solution with rank 47, 46...
    # We will use a "relaxation" approach:
    # 1. Start with the schoolbook solution (rank 48).
    # 2. Try to merge columns.
    #
    # Actually, let's look at the structure of the tensor T_{ijk} = 1.
    # This is the "all-ones" tensor? No, it's the matrix multiplication tensor.
    # T_{ijk} is the coefficient of x_i y_j z_k in (sum x_i y_j) (sum y_j z_k)?
    # No, the matrix multiplication tensor T_{ijk} is defined such that 
    # (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k * (1 if it's the mult tensor).
    # Wait, the standard tensor for matrix multiplication <n,m,p> has non-zero entries 
    # T_{ijk} = 1 if the indices correspond to the multiplication structure.
    # Specifically, T_{i,j,k} = 1 for all i in [n], j in [m], k in [p] is NOT correct 
    # for the standard tensor definition used in rank calculations.
    # The standard tensor is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i \otimes e_j \otimes e_k * delta_{?}
    # Actually, the tensor is:
    # T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Since every A_{ij} multiplies every B_{jk} to contribute to C_{ik}, 
    # the tensor T_{ijk} is 1 for ALL i,j,k.
    # Yes, the "matrix multiplication tensor" in the context of tensor rank 
    # (Strassen's algorithm) is defined on vector spaces V1, V2, V3 with dimensions 
    # n*m, m*p, p*n.
    # The tensor T in V1* x V2* x V3* is:
    # T = sum_{i,j,k} (e_i \otimes e_j \otimes e_k) * 1 ?
    # No, the tensor is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{ij} \otimes E_{jk} \otimes E_{ki}^* ?
    #
    # Let's stick to the contract:
    # "exactly decompose the matrix multiplication tensor"
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r equals 1 for all i,j,k.
    # This confirms T_{ijk} = 1 for all i,j,k in the flattened index space.
    #
    # So we are decomposing the "all-ones" tensor of size (nm) x (mp) x (pn) 
    # into a sum of rank-1 tensors?
    # No, that's not right. The indices are coupled.
    # The equation is:
    # sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1
    #
    # This is exactly the definition of the rank of the matrix multiplication tensor.
    #
    # For n=2, m=4, p=6.
    # Trivial rank = 48.
    #
    # Let's try to find a rank < 48.
    # We can use a solver.
    # Let R be the rank.
    # We have 24 equations (for each i,j,k).
    # Variables: R * (8 + 24 + 12) = 44R variables.
    #
    # If we fix U and V to be random sparse matrices with entries in {0, 1, -1},
    # can we solve for W?
    # The equation is linear in W if we fix U and V?
    # sum_r (U_{ij, r} V_{jk, r}) W_{ki, r} = 1
    # This is a linear system for the entries of W?
    # No, W_{ki, r} is indexed by r.
    # For a fixed (i,k), we have:
    # sum_r (U_{ij, r} V_{jk, r}) W_{ki, r} = 1 for all j.
    # This is a system of m=4 equations for the R variables W_{ki, 0..R-1}.
    # We have n*p = 12 such systems (one for each i,k).
    #
    # So, if we fix U and V, we can solve for W independently for each (i,k) pair!
    #
    # Algorithm:
    # 1. Guess a rank R.
    # 2. Generate random sparse U and V (e.g., Bernoulli or small integers).
    # 3. For each i in [n], k in [p]:
    #    Construct matrix A_{ik} of size m x R.
    #    A_{ik}[j, r] = U[i*m+j, r] * V[j*p+k, r]
    #    Solve A_{ik} * w_{ik} = 1 (vector of ones of length m).
    #    If a solution exists (and is "nice", i.e., rational/half-integer), 
    #    then we have a candidate W.
    # 4. Check if the resulting W (assembled from all w_{ik}) is consistent?
    #    Wait, W is a single matrix of size (p*n) x R.
    #    The entry W[k*n+i, r] is the r-th component of the solution w_{ik}.
    #    So we just assign W[k*n+i, :] = w_{ik}.
    # 5. Verify the full decomposition.
    #
    # This reduces the non-linear problem to a series of linear systems!
    # The only non-linearity was the coupling between i,j,k, but by fixing U,V, 
    # the dependence on W is linear for each (i,k) block.
    #
    # We need the solutions w_{ik} to be "simple" (integers or half-integers) 
    # to satisfy the contract? The contract says "integer/half-integer entries".
    # If the linear system solution is rational, we can check if the denominator 
    # is 1 or 2.
    #
    # Let's implement this.
    
    def verify(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    # Search for lower ranks
    # We will try ranks from 47 down to 24 (theoretical lower bound).
    # For each rank, we do random restarts of U, V.
    
    # To ensure exactness with half-integers, we can use Fraction or just 
    # solve with floats and check if close to x.0 or x.5.
    # Using Fraction is safer for exactness.
    
    import numpy.linalg as la
    
    # We'll use a helper to solve linear system A x = b using Fraction
    # But numpy doesn't support Fraction.
    # We can use sympy? No, standard library + numpy/scipy.
    # We can solve using Gaussian elimination manually with Fractions for small systems.
    # m=4, R is at most 48. 4x48 system. Very small.
    
    def solve_linear_system_fraction(A, b):
        """
        Solves A x = b where A is m x R, b is m x 1.
        Returns x (R x 1) as list of Fractions if unique solution or 
        returns a particular solution if underdetermined (using least squares? 
        No, we want ANY solution that satisfies exactly).
        If the system is consistent, we can find a solution.
        If R > m, there are infinite solutions. We can set free variables to 0 
        and solve for the rest.
        """
        m_len = len(b)
        R_len = len(A[0])
        
        # Augmented matrix
        # We want to solve A x = b.
        # If A has full row rank, we can find a particular solution.
        # We can use Gauss-Jordan elimination.
        
        # Copy A and b to Fraction
        M = [[Fraction(A[i][j]) for j in range(R_len)] + [Fraction(b[i])] for i in range(m_len)]
        
        # Pivot
        pivot_row = 0
        pivot_cols = []
        
        for col in range(R_len):
            if pivot_row >= m_len:
                break
            # Find pivot
            pivot = -1
            for row in range(pivot_row, m_len):
                if M[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            
            # Swap
            M[pivot_row], M[pivot] = M[pivot], M[pivot_row]
            pivot_cols.append(col)
            
            # Normalize pivot row
            pivot_val = M[pivot_row][col]
            for j in range(col, R_len + 1):
                M[pivot_row][j] = M[pivot_row][j] / pivot_val
                
            # Eliminate other rows
            for row in range(m_len):
                if row == pivot_row:
                    continue
                factor = M[row][col]
                if factor != 0:
                    for j in range(col, R_len + 1):
                        M[row][j] = M[row][j] - factor * M[pivot_row][j]
                        
            pivot_row += 1
            
        # Check consistency
        for i in range(pivot_row, m_len):
            if M[i][R_len] != 0:
                return None # Inconsistent
        
        # Construct solution
        # Variables corresponding to pivot_cols are determined.
        # Others are free (set to 0).
        x = [Fraction(0) for _ in range(R_len)]
        
        for i, col in enumerate(pivot_cols):
            x[col] = M[i][R_len]
            
        return x

    # Search loop
    # We want to find the minimal rank.
    # Let's try ranks from 47 down to 30 first.
    # If we find one, we return it.
    
    # To speed up, we can restrict U and V entries to {-1, 0, 1}.
    # This increases the chance of the linear system having a "nice" solution.
    
    values = [-1, 0, 1]
    
    # We will try a few random restarts for each rank.
    # The number of restarts depends on time.
    
    # Let's estimate time per trial.
    # 12 systems of size 4 x R.
    # R=47. 12 * (4*47) ~ 2200 ops. Very fast.
    # We can do thousands of trials.
    
    # Let's try ranks 47, 46, ...
    # But maybe a much lower rank is possible?
    # For 2x4x6, is the rank 48?
    # Let's check if rank 47 is possible.
    
    # We'll iterate R from 47 down to 24.
    # For each R, we do N trials.
    # If we find a valid decomposition with half-integer entries, we return it.
    
    # Note: The solution x from solve_linear_system_fraction might have denominators 
    # other than 1 or 2. We must check if all entries are in {k/2 | k in Z}.
    
    def is_half_integer(frac):
        num, den = frac.numerator, frac.denominator
        if den == 1:
            return True
        if den == 2:
            return True
        return False

    # We can also try to "scale" the solution if it's rational?
    # No, the entries of U,V,W are fixed. We can't scale W independently.
    # So we need the specific solution to be half-integer.
    # If the system is underdetermined (R > m), we have degrees of freedom.
    # We can try to find a solution in the affine space that is half-integer.
    # This is a harder problem.
    # For now, we assume the particular solution (free vars = 0) works, 
    # or we try a few random settings for free variables.
    
    # Actually, if R > m, the system A x = b has a solution space x_p + x_h.
    # We want x_p + x_h to be half-integer.
    # This is a linear Diophantine problem over Q.
    # Given the small size, we can just try the particular solution first.
    
    # Let's run the search.
    
    # We'll prioritize lower ranks if we can, but start from 47 to ensure we find *something* better than 48 quickly.
    # Actually, finding rank 47 might be hard.
    # Let's try a range.
    
    # To be safe and fast, let's try 47, 46, 45, 44, 43, 42, 41, 40.
    
    # We will do 1000 trials for each rank.
    
    # Optimization: Pre-calculate indices.
    
    # Let's run.
    
    for R in range(47, 23, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Number of trials: increase if R is close to 48, decrease if lower?
        # Let's do 500 trials per rank.
        num_trials = 500
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            # Generate random U, V
            U = [[random.choice(values) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(R)] for _ in range(m * p)]
            
            # Solve for W
            W = [[0] * R for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct A (m x R)
                    A = [[0] * R for _ in range(m)]
                    b = [1] * m
                    
                    for j in range(m):
                        for r in range(R):
                            A[j][r] = U[i * m + j][r] * V[j * p + k][r]
                            
                    # Solve A x = b
                    x = solve_linear_system_fraction(A, b)
                    
                    if x is None:
                        valid = False
                        break
                        
                    # Check if x entries are half-integers
                    for r in range(R):
                        if not is_half_integer(x[r]):
                            valid = False
                            break
                            
                    if not valid:
                        break
                        
                    # Assign to W
                    for r in range(R):
                        W[k * n + i][r] = x[r]
                        
            if not valid:
                continue
                
            # Verify full decomposition
            if verify(U, V, W):
                # Convert Fractions to float or int for output?
                # The contract says "integer/half-integer entries".
                # Usually, returning float is fine if it's exactly representable?
                # Or should we return Fraction objects?
                # "entries that exactly decompose"
                # Let's return floats for simplicity, as 0.5 is exact in binary.
                # But wait, the verify function uses abs(s-1) > 1e-9.
                # If we return Fractions, verify needs to handle them.
                # My verify uses standard multiplication.
                # Let's convert to float.
                
                U_out = [[float(val) for val in row] for row in U]
                V_out = [[float(val) for val in row] for row in V]
                W_out = [[float(val) for val in row] for row in W]
                
                # Final check with floats
                if verify(U_out, V_out, W_out):
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = R
                    found = True
                    break # Break trial loop
                    
            if found:
                break # Break R loop? No, we want the MINIMAL rank.
                # But we are iterating R downwards. So the first one we find is the best for this R.
                # We should break the R loop if we found one?
                # The task asks for "rank-minimal".
                # If we find rank 47, we should check 46?
                # Yes, we should continue searching for lower ranks.
                # But we update best_result.
        
        # If we found a result for this R, we can continue to lower R?
        # Yes.
        
    return best_result

# EVOLVE-BLOCK-END
