"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    
    # The tensor T for matrix multiplication C = A * B is defined such that
    # for all i in [0, n), j in [0, m), k in [0, p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for any other combination of indices (i, j, k) that doesn't form a valid
    # triple (i, j, k) in the support of the tensor, the sum must be 0.
    # However, the tensor is only non-zero at (i, j, k) where the indices match the
    # multiplication structure. Specifically, the tensor is the identity tensor
    # in the space of (A, B, C) indices.
    # The condition is:
    # For all i, j, k:
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # This is because the only non-zero entries of the multiplication tensor are
    # at (i, j, k) with value 1.
    
    # We will attempt to find a decomposition with rank R.
    # We start with R = 48, which is a reasonable target (half of schoolbook 96).
    # If that fails, we try higher ranks.
    
    # The system of equations is:
    # For each (i, j, k):
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 96 equations.
    # Unknowns: R * (n*m + m*p + p*n) = R * (12 + 32 + 24) = 68 * R.
    
    # We can fix U and V to be sparse or have specific structure to make the problem
    # solvable. A common approach is to use a "basis" for U and V, then solve for W.
    # However, the equation is trilinear.
    
    # A simpler approach for exact integer/half-integer solutions is to use a known
    # algorithm. Since we don't have a specific algorithm for 3x4x8, we will use
    # a heuristic search.
    
    # Let's try to find a rank R decomposition by solving a least-squares problem
    # for real numbers, then rounding to integers/half-integers if close.
    
    # But first, let's try to see if a rank 48 decomposition exists with a specific
    # structure.
    
    # We will use the following strategy:
    # 1. Initialize U, V, W with random values.
    # 2. Use an iterative algorithm to minimize the error.
    # 3. If the error is small, round to integers/half-integers and verify.
    
    # Given the time constraint, we will implement a simple version of this.
    
    # However, a more robust approach for small matrices is to use a known
    # decomposition.
    
    # Let's try to construct a rank 48 decomposition by splitting the problem.
    # C = A * B
    # B is 4x8. Let B = [B1, B2] where B1, B2 are 4x4.
    # C = [A*B1, A*B2].
    # So we need a rank R_344 decomposition for 3x4x4.
    # If R_344 = 24, then total rank is 48.
    
    # Let's try to find a rank 24 decomposition for 3x4x4.
    # The tensor for 3x4x4 has 3*4*4 = 48 equations.
    # Unknowns: 24 * (12 + 16 + 12) = 24 * 40 = 960.
    
    # We will attempt to solve for a rank 24 decomposition of 3x4x4.
    
    def find_decomposition(n, m, p, R, max_iter=1000, tol=1e-6):
        """
        Attempts to find a rank R decomposition for n x m x p matrix multiplication.
        Returns (U, V, W) if found, else None.
        """
        # Initialize U, V, W with small random values
        np.random.seed(42)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # We want to minimize the sum of squares of (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)
        # for all i, j, k.
        
        # This is a non-convex optimization problem. We will use a simple gradient
        # descent or alternating least squares.
        
        # Alternating Least Squares (ALS):
        # Fix U, V -> solve for W (linear system).
        # Fix U, W -> solve for V.
        # Fix V, W -> solve for U.
        
        # For fixed U, V, the equation for W is:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        # For a fixed (i, k), let idx_W = k*n+i.
        # The equation is:
        # sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[idx_W, r] = 1
        # Let M_{j, r} = U[i*m+j, r] V[j*p+k, r].
        # Then sum_j M_{j, r} W[idx_W, r] = 1.
        # This is a linear equation in W[idx_W, :].
        # Specifically, for each (i, k), we have:
        # sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1
        # Let A_{(i,k), r} = sum_j U[i*m+j, r] V[j*p+k, r].
        # Then A_{(i,k), :} dot W[k*n+i, :] = 1.
        
        # So for each (i, k), we have a linear equation:
        # A_{(i,k), :} dot W[k*n+i, :] = 1
        
        # We can solve this for W by setting up a system of equations.
        # There are n*p equations and R unknowns for each row of W.
        # Wait, W has shape (p*n, R).
        # For each row idx_W = k*n+i, we have one equation:
        # A_{(i,k), :} dot W[idx_W, :] = 1.
        # This is an underdetermined system (1 equation, R unknowns) for each row.
        # We can choose the minimum norm solution:
        # W[idx_W, :] = A_{(i,k), :} / ||A_{(i,k), :}||^2
        
        # Similarly for U and V.
        
        # Let's implement ALS.
        
        for _ in range(max_iter):
            # Update W
            for i in range(n):
                for k in range(p):
                    idx_W = k * n + i
                    # Compute A_{(i,k), r} = sum_j U[i*m+j, r] V[j*p+k, r]
                    A_row = np.zeros(R)
                    for j in range(m):
                        A_row += U[i * m + j, :] * V[j * p + k, :]
                    norm_sq = np.dot(A_row, A_row)
                    if norm_sq > tol:
                        W[idx_W, :] = A_row / norm_sq
                    else:
                        # If A_row is zero, we can't satisfy the equation.
                        # Set W to zero or a small value.
                        W[idx_W, :] = 0.0
            
            # Update V
            for j in range(m):
                for k in range(p):
                    idx_V = j * p + k
                    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                    # For fixed (j, k), and for each i:
                    # sum_r U[i*m+j, r] W[k*n+i, r] V[idx_V, r] = 1
                    # Let B_{i, r} = U[i*m+j, r] W[k*n+i, r].
                    # Then B_{i, :} dot V[idx_V, :] = 1 for each i.
                    # This gives n equations for V[idx_V, :].
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    # Solve B dot V[idx_V, :] = ones(n)
                    # Minimum norm solution: V[idx_V, :] = B^T (B B^T)^{-1} ones
                    # Or use least squares.
                    try:
                        V[idx_V, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        V[idx_V, :] = 0.0
            
            # Update U
            for i in range(n):
                for j in range(m):
                    idx_U = i * m + j
                    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                    # For fixed (i, j), and for each k:
                    # sum_r V[j*p+k, r] W[k*n+i, r] U[idx_U, r] = 1
                    # Let C_{k, r} = V[j*p+k, r] W[k*n+i, r].
                    # Then C_{k, :} dot U[idx_U, :] = 1 for each k.
                    C = np.zeros((p, R))
                    for k in range(p):
                        C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[idx_U, :] = np.linalg.lstsq(C, np.ones(p), rcond=None)[0]
                    except:
                        U[idx_U, :] = 0.0
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                        error += (val - 1.0) ** 2
            if error < tol:
                break
        
        # Check if the solution is valid
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                    error += (val - 1.0) ** 2
        
        if error < 1e-4:
            # Try to round to integers/half-integers
            # Check if entries are close to integers or half-integers
            def is_close_to_half_int(x):
                return np.allclose(x, np.round(x * 2) / 2, atol=1e-3)
            
            if is_close_to_half_int(U) and is_close_to_half_int(V) and is_close_to_half_int(W):
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                # Verify the integer/half-integer solution
                error_int = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j, :] * V_int[j * p + k, :], W_int[k * n + i, :])
                            error_int += (val - 1.0) ** 2
                if error_int < 1e-4:
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    # Try to find a rank 48 decomposition for 3x4x8
    # This is equivalent to finding a rank 24 decomposition for 3x4x4 and doubling it.
    # But let's try to find a rank 48 decomposition directly for 3x4x8.
    
    # Actually, let's try to find a rank R decomposition for 3x4x8 for R in [48, 56, 64, 72, 80, 88, 96]
    # Start with R = 48.
    
    for R in [48, 56, 64, 72, 80, 88, 96]:
        result = find_decomposition(n, m, p, R)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If all fail, return the schoolbook decomposition
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
