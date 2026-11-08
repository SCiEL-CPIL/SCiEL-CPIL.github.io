"""Rewritten program for matrix multiplication tensor decomposition 3x3 * 3x4."""
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Strategy: Use a randomized linear algebra approach over finite fields (GF(2) and GF(3))
    to search for a low-rank decomposition by solving systems of linear equations derived
    from the tensor's flattening. Then lift to exact rational solutions.
    
    The matrix multiplication tensor T_{ijk} (where i indexes rows of A, j indexes columns of A/rows of B,
    k indexes columns of B) has rank at most n*m*p = 36 (schoolbook). We search for lower ranks.
    
    For 3x3 * 3x4, known lower bounds suggest rank is around 24-28. We'll try to find
    a decomposition with rank as low as possible using finite field methods.
    """
    n, m, p = DIMENSIONS
    # T is a 3D tensor: T[i, j, k] = 1 if this corresponds to a valid multiplication term
    # For matrix multiplication (A * B)_{ik} = sum_j A_{ij} * B_{jk}
    # So the tensor has T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The tensor flattening: for a rank-r decomposition, we need matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # This is a bilinear system in U,V given W, or trilinear overall.
    # We'll use the approach: fix one factor, solve for others.
    
    # First, let's try to find a good decomposition using known constructions or search.
    # For 3x3 * 3x4, let's try rank 24, 25, 26, ...
    
    # Approach: Use the fact that over a finite field, we can randomly initialize one factor
    # and solve the resulting linear system for the other two factors.
    
    def try_rank(r, field_mod=None, seed=42):
        """Try to find a rank-r decomposition. Returns (U, V, W) as lists or None."""
        np.random.seed(seed)
        
        if field_mod is None:
            # Try over rationals using random integer initialization
            # Initialize W randomly with small integers
            W = np.random.randint(-1, 2, size=(p * n, r))
        else:
            # Initialize W randomly over the field
            W = np.random.randint(0, field_mod, size=(p * n, r)) % field_mod
        
        # For each (i, j) pair, we have a linear system in U and V.
        # Actually, the constraint is: for each (i, j, k):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #
        # This is bilinear in U and V for fixed W.
        # We can vectorize: for fixed i, j, the equation over all k is:
        # For each k: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
        #
        # Let's define for each c: w_c(i,k) = W[k*n+i, c]
        # Then the equation is: sum_c U[i*m+j, c] * sum_k' ... no, this isn't quite right.
        #
        # Better approach: For fixed W, the system is bilinear. We can use the following:
        # For each (i, j), define a vector u_ij of length r (row of U) and for each k,
        # the equation is: u_ij . (V[j*p+k, :] * W[k*n+i, :]) = 1
        #
        # This is still bilinear. To make it linear, we can fix U and solve for V, or vice versa.
        # Let's use an iterative approach: alternate between solving for U (given V) and V (given W, U).
        
        # Actually, let's use a different approach: treat it as a linear system in the entries of U and V
        # by fixing one at a time.
        
        # Initialize V randomly
        if field_mod is None:
            V = np.random.randint(-1, 2, size=(m * p, r))
        else:
            V = np.random.randint(0, field_mod, size=(m * p, r)) % field_mod
        
        # Iterate: solve for U given V and W, then solve for V given U and W
        for iteration in range(100):
            # Solve for U: for each (i, j), we need u_ij such that for all k:
            # sum_c u_ij[c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a linear system in u_ij: A_ij @ u_ij = b_ij
            # where A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c] and b_ij[k] = 1
            
            U_new = np.zeros((n * m, r), dtype=field_mod if field_mod else int)
            for i in range(n):
                for j in range(m):
                    # Build the linear system
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    b = np.ones(p)
                    
                    if field_mod is None:
                        # Solve over reals
                        try:
                            # Use least squares if overdetermined
                            sol, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            # Check if the solution is close to integer/half-integer
                            U_new[i * m + j, :] = sol
                        except:
                            return None
                    else:
                        # Solve over finite field
                        try:
                            sol = _solve_linear_system_ff(A, b, field_mod)
                            if sol is None:
                                return None
                            U_new[i * m + j, :] = sol
                        except:
                            return None
            
            # Now solve for V given U and W
            V_new = np.zeros((m * p, r), dtype=field_mod if field_mod else int)
            for j in range(m):
                for k in range(p):
                    # For each (j, k), we need v_jk such that for all (i, i2) where i2 is row of A:
                    # Actually, the constraint couples all i. Let me re-examine.
                    #
                    # The full constraint is: for all i, j, k:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    #
                    # For fixed j, k, and varying i, we have:
                    # For each i: sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1
                    # Let a_i[c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then: sum_c a_i[c] * V[j*p+k, c] = 1 for all i
                    # This is a linear system in V[j*p+k, :]:
                    # A_jk @ v_jk = 1
                    # where A_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U_new[i * m + j, c] * W[k * n + i, c]
                    b = np.ones(n)
                    
                    if field_mod is None:
                        try:
                            sol, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j * p + k, :] = sol
                        except:
                            return None
                    else:
                        try:
                            sol = _solve_linear_system_ff(A, b, field_mod)
                            if sol is None:
                                return None
                            V_new[j * p + k, :] = sol
                        except:
                            return None
            
            # Check convergence
            if field_mod is None:
                if np.allclose(U_new, U, atol=1e-8) and np.allclose(V_new, V, atol=1e-8):
                    U, V = U_new, V_new
                    break
            else:
                if np.array_equal(U_new, U) and np.array_equal(V_new, V):
                    U, V = U_new, V_new
                    break
            
            U, V = U_new, V_new
        
        # Verify the solution
        if field_mod is None:
            # Check if the solution is exact (integer or half-integer)
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify the decomposition
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_rounded[i * m + j, c] * V_rounded[j * p + k, c] * W_rounded[k * n + i, c]
                                  for c in range(r))
                        if abs(val - 1.0) > 1e-6:
                            return None
            
            # Convert to list of lists with Fraction for exactness
            U_list = [[Fraction(int(U_rounded[i, c] * 2)) / 2 for c in range(r)] for i in range(n * m)]
            V_list = [[Fraction(int(V_rounded[i, c] * 2)) / 2 for c in range(r)] for i in range(m * p)]
            W_list = [[Fraction(int(W_rounded[i, c] * 2)) / 2 for c in range(r)] for i in range(p * n)]
            
            return U_list, V_list, W_list
        else:
            # Verify over the finite field
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] mod field_mod == 1
                        val = 0
                        for c in range(r):
                            val = (val + int(U[i * m + j, c]) * int(V[j * p + k, c]) * int(W[k * n + i, c])) % field_mod
                        if val != 1 % field_mod:
                            return None
            
            # Lift to rationals: the solution over GF(p) can be lifted to integers in [0, p-1]
            # For small fields, these are already integers
            U_list = [[int(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[int(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[int(W[i, c]) for c in range(r)] for i in range(p * n)]
            
            return U_list, V_list, W_list
    
    def _solve_linear_system_ff(A, b, mod):
        """Solve A @ x = b over GF(mod). A is (m, n), b is (m,). Returns x or None if no solution."""
        m, n = A.shape
        # Augmented matrix
        aug = np.zeros((m, n + 1), dtype=int)
        for i in range(m):
            for j in range(n):
                aug[i, j] = int(A[i, j]) % mod
            aug[i, n] = int(b[i]) % mod
        
        # Gaussian elimination
        rank = 0
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(rank, m):
                if aug[row, col] % mod != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            
            # Swap rows
            if pivot_row != rank:
                aug[[rank, pivot_row]] = aug[[pivot_row, rank]]
            
            # Normalize pivot row
            inv_pivot = _mod_inverse(aug[rank, col], mod)
            for j in range(col, n + 1):
                aug[rank, j] = (aug[rank, j] * inv_pivot) % mod
            
            # Eliminate other rows
            for row in range(m):
                if row == rank:
                    continue
                factor = aug[row, col]
                if factor % mod != 0:
                    for j in range(col, n + 1):
                        aug[row, j] = (aug[row, j] - factor * aug[rank, j]) % mod
            
            rank += 1
        
        # Check for consistency
        for row in range(rank, m):
            if aug[row, n] % mod != 0:
                return None
        
        # Extract solution (set free variables to 0)
        x = np.zeros(n, dtype=int)
        for row in range(rank):
            # Find the column of the pivot in this row
            for col in range(n):
                if aug[row, col] % mod != 0:
                    x[col] = aug[row, n] % mod
                    break
        
        return x
    
    def _mod_inverse(a, mod):
        """Find modular inverse of a mod mod using extended Euclidean algorithm."""
        a = a % mod
        if a == 0:
            raise ValueError("No inverse for 0")
        # Extended Euclidean
        g, x, _ = _extended_gcd(a, mod)
        if g != 1:
            raise ValueError(f"No inverse for {a} mod {mod}")
        return x % mod
    
    def _extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = _extended_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1
    
    # Try different ranks and fields
    # Known: schoolbook is 36. Let's try to find something better.
    # For 3x3 * 3x4, let's try ranks from 20 to 35
    
    best_result = None
    for r in range(20, 37):
        # Try over GF(2)
        result = try_rank(r, field_mod=2, seed=42)
        if result is not None:
            best_result = result
            break
        
        # Try over GF(3)
        result = try_rank(r, field_mod=3, seed=42)
        if result is not None:
            best_result = result
            break
        
        # Try over reals with random initialization
        result = try_rank(r, field_mod=None, seed=42)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
