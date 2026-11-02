"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # We try to find a rank-minimal exact tensor decomposition for 2x3 * 3x10.
    # The schoolbook rank is 2*3*10 = 60.
    # We attempt to find a lower rank decomposition using a structured search.
    #
    # The tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j] B[j,k])
    # is decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For a rank-r decomposition, we need:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==k else 0
    #   (for all i in [0,n), j in [0,m), k in [0,p))
    #
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    #
    # We try ranks from a lower bound up to 60 and use a randomized search
    # with exact rational arithmetic to find a valid decomposition.

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60

    # Deadline guard: must finish within 120s wall-clock, use 110s budget
    deadline = time.time() + 110

    # Try to find a decomposition with rank < 60
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Also, the tensor rank is at least the matrix rank of any slice.
    # For the (i,k)-slice (fixing i,k), we get a matrix of size m x m... 
    # Actually the tensor is T: R^{n*m} x R^{m*p} x R^{p*n}
    # The (i,k) fiber is a vector in R^{m*p} indexed by j, but that's not quite right.
    # Let me think about lower bounds more carefully.
    #
    # The multiplication tensor T_{n,m,p} has entries T_{(i,j),(j,k),(k,i)} = 1.
    # A lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
    # Another lower bound: the tensor rank >= max dimension of any mode.
    # Mode 1: n*m = 6, Mode 2: m*p = 30, Mode 3: p*n = 20.
    # So rank >= 30.
    #
    # For 2x3*3x10, let's try to find rank 30, 31, ..., 59.

    # We'll use a structured approach: try to exploit the symmetry.
    # For the standard multiplication, one known approach is to use
    # the fact that C = A * B where A is 2x3 and B is 3x10.
    #
    # Let's try a direct search for small rank reductions from 60.
    # We'll set up the linear system and try to solve it.

    def try_rank(r, time_left):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        if time_left() <= 1:
            return None
        
        # Number of variables: r * (n*m + m*p + p*n)
        # Number of equations: n*m*p = 60
        # For r >= 30, we have 30*(6+30+20) = 30*56 = 1680 variables, 60 equations.
        # This is highly underdetermined. We need to find a solution where
        # the entries are "nice" (integers or half-integers).
        
        # Strategy: Fix most of U to be a partial identity, then solve for the rest.
        # For rank r, let's set U[i*m+j, c] = 1 for c < min(r, n*m) and 0 otherwise
        # in a structured way, then solve for V and W.
        
        # Actually, let's try a different approach.
        # For the schoolbook, each term (i,j,k) contributes one rank-1 tensor.
        # To reduce rank, we need to combine terms.
        
        # Let's try: for each j in range(m), we have n*p = 20 terms.
        # If we can handle each j with fewer than n*p terms, we save.
        # For fixed j, the contribution is sum over i,k of A_{i,j} * B_{j,k} * C_{k,i}
        # which is an outer product of (A_{:,j} in R^n) and (B_{j,:} in R^p) 
        # mapped through the trace structure.
        
        # Actually for fixed j, the sub-tensor is:
        # T_j[(i),(k)] = A_{i,j} * B_{j,k} * delta_{i,k}... no that's not right.
        # The full tensor is T[(i,j),(j,k),(k,i)] = 1 for matching indices.
        # For fixed j, T[(i,j),(j,k),(k,i)] = 1, which is a rank-1 tensor in 
        # (i,k) space: it's the identity matrix I_{n x n}... no, it's delta_{i,k}.
        # Wait, T[(i,j),(j,k),(k,i)] = 1 for all i,j,k. There's no delta.
        # The delta comes from the multiplication structure.
        
        # Let me re-read the problem. The tensor for matrix multiplication C = A*B
        # where A is n x m, B is m x p, C is n x p is:
        # T_{(i,j),(j,k),(i,k)} = 1 for all valid i,j,k.
        # The decomposition is: sum_r U[(i,j),r] * V[(j,k),r] * W[(i,k),r] = 1
        # for all i,j,k.
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] = 1
        # for all i in [0,n), j in [0,m), k in [0,p).
        
        # Note: W is indexed by [i*p+k] in this formulation, but the problem says
        # W[k*n+i, r]. Let me re-check.
        # The problem says W[k*n+i, r]. So W is indexed by (k,i) with k first.
        # So W[k*n + i, r].
        
        # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for all i,j,k.
        
        # Let's try a structured decomposition.
        # For n=2, m=3, p=10:
        # We can try to use the fact that for each j, we need to represent
        # the "identity" structure in (i,k) space.
        
        # Approach: Try rank = 30 + something small.
        # For each j (3 values), we have a 2x10 sub-problem.
        # The sub-tensor for fixed j is: sum_r U[2*j+i, r] * V[10*j+k, r] * W[10*i+k, r] = 1
        # for all i in [0,2), k in [0,10).
        
        # This is a 2x10 matrix of 1s that needs to be represented.
        # If we use rank-1 terms, we need 20 terms per j, total 60.
        # Can we do better by sharing terms across different j values?
        
        # Let's try a direct numerical search with rational arithmetic.
        # We'll set up the system and try to find a sparse solution.
        
        # For now, let's just return the schoolbook if we can't find better.
        # But let's try a few specific structured decompositions.
        
        # Try: Use the fact that 2x10 can be done with 20 terms, but maybe
        # we can share some structure.
        
        # Actually, let's try rank 59, 58, ... down to 30 and see if we can
        # find a valid solution.
        
        # For a quick check, let's try to see if rank 59 works by perturbing
        # the schoolbook solution.
        
        # The schoolbook has 60 terms. If we can merge two terms into one,
        # we get rank 59. Two terms (i1,j1,k1) and (i2,j2,k2) can be merged
        # if their product structure allows it.
        
        # Term (i,j,k) gives: U[i*m+j] = e_{i,j}, V[j*p+k] = e_{j,k}, W[k*n+i] = e_{k,i}
        # where e_{i,j} is a unit vector.
        
        # To merge (i1,j1,k1) and (i2,j2,k2), we'd need:
        # U[i1*m+j1] + U[i2*m+j2] (in the new combined term)
        # V[j1*p+k1] + V[j2*p+k2]
        # W[k1*n+i1] + W[k2*n+i2]
        # such that the product still gives the right values.
        
        # This is complex. Let's try a numerical approach.
        
        # Set up the linear system for rank r.
        # Variables: U (6 x r), V (30 x r), W (20 x r)
        # Equations: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # This is a system of 60 equations, each cubic in the variables.
        # It's not linear! So we can't just solve a linear system.
        
        # We need a different approach. Let's try to fix some variables and
        # solve for others.
        
        # Fix U to be a fixed matrix, then the equations become quadratic in V and W.
        # Or fix U and V, then it's linear in W.
        
        # Strategy: Fix U and V, solve for W.
        # For each (i,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j.
        # Wait, the equation is for each (i,j,k). So for fixed (i,k), we have m equations
        # (one for each j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1.
        
        # If we fix U and V, then for each (i,k), we have a linear equation in W[k*n+i,:]:
        # sum_r (sum_j' U[i*m+j',r] * V[j'*p+k,r]) * W[k*n+i,r] = 1
        # Wait no, the sum is over r, and for each (i,j,k) we have a separate equation.
        
        # Let me re-index. For fixed (i,k), the equations are:
        # For each j: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # So if we define M_{i,k}^{(j)}[r] = U[i*m+j,r] * V[j*p+k,r], then
        # sum_r M_{i,k}^{(j)}[r] * W[k*n+i,r] = 1 for each j.
        
        # This means: for fixed (i,k), the vector W[k*n+i,:] must satisfy
        # m linear equations (one per j). If m > 1, this is generally overdetermined
        # unless the M vectors are linearly dependent.
        
        # For m=3, we have 3 equations in r unknowns (the entries of W[k*n+i,:]).
        # For r >= 3, this is underdetermined and has a solution if the equations
        # are consistent.
        
        # So the strategy is:
        # 1. Choose U and V (6 x r and 30 x r matrices).
        # 2. For each (i,k), check if the system has a solution for W[k*n+i,:].
        # 3. If yes, solve for W and check if all equations are satisfied.
        
        # Let's implement this.
        
        def solve_for_W(U, V, r):
            """Given U (n*m x r) and V (m*p x r), solve for W (p*n x r)."""
            W = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    # For each j, we have: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # This is a linear system in W[k*n+i,:]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve A @ W[k*n+i,:] = ones(m)
                    # Use least squares if overdetermined, or find a particular solution
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if the solution is valid
                        if rank_A < m:
                            # Underdetermined or rank-deficient, check consistency
                            if np.allclose(A @ sol, np.ones(m), atol=1e-10):
                                W[k*n+i, :] = sol
                            else:
                                return None
                        else:
                            if np.allclose(A @ sol, np.ones(m), atol=1e-10):
                                W[k*n+i, :] = sol
                            else:
                                return None
                    except:
                        return None
            return W
        
        # Now we need to choose U and V such that the above works.
        # Let's try a structured choice.
        
        # For the schoolbook, U[i*m+j, c] = 1 if c corresponds to (i,j,*) for some k,
        # but actually in schoolbook each (i,j,k) is a separate term.
        
        # Let's try: U[i*m+j, c] = 1 for c in [j*p, j*p+p) (i.e., for the k values
        # associated with this j), and 0 otherwise. This doesn't quite work.
        
        # Alternative: Let's try U[i*m+j, c] = 1 for c < r in a pattern.
        
        # Let's try a random search with the above structure.
        
        best_result = None
        
        # Try a few random initializations
        np.random.seed(42)
        
        for attempt in range(100):
            if time_left() <= 5:
                break
            
            # Initialize U and V randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            W = solve_for_W(U, V, r)
            if W is None:
                continue
            
            # Verify the solution
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to fractions for exact representation
                U_frac = [[Fraction(x).limit_denominator(10000) for x in U[il]] for il in range(n*m)]
                V_frac = [[Fraction(x).limit_denominator(10000) for x in V[il]] for il in range(m*p)]
                W_frac = [[Fraction(x).limit_denominator(10000) for x in W[il]] for il in range(p*n)]
                best_result = (U_frac, V_frac, W_frac)
                break
        
        return best_result

    # Try ranks from 30 up to 59
    for r in range(30, 60):
        result = try_rank(r, lambda: deadline - time.time())
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
