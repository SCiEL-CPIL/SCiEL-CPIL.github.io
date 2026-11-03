import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def validate_decomposition(U, V, W):
        # Check if sum_r U[:, r] * V[:, r] * W[:, r] == T
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reconstruct T_approx
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor product sum.
        # We can compute this efficiently?
        # For small sizes, just loop or use einsum
        R = U.shape[1]
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if T_approx is close to T
        try:
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        except Exception:
            return False
        if not np.allclose(T_approx, T, atol=1e-9):
            return False
        # Also check if entries are rational/half-integer
        # The problem says "integer/half-integer entries"
        # We will convert to Fraction to check
        def check_rational(matrix):
            for row in matrix:
                for val in row:
                    f = Fraction(val).limit_denominator(100)
                    if abs(f - val) > 1e-9:
                        return False
                    # Check if denominator is 1 or 2
                    if f.denominator not in [1, 2]:
                        return False
            return True
        return check_rational(U) and check_rational(V) and check_rational(W)

    def try_rank(R, max_iters=200, seed=42):
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        # Initialize with random values
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Simple gradient descent or ALS
        # Let's use a few iterations of ALS (Alternating Least Squares)
        # Minimize ||T - sum U V W||^2
        # Fix V, W, update U:
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed a, U[a, :] is a vector.
        # The term is U[a, r] * (V[b, r] * W[c, r]).
        # This is a linear system for U[a, :].
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            # For each a, we want to solve:
            # sum_r U[a, r] * (V[:, r] * W[:, r])_broadcast? No.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Let M_r[b, c] = V[b, r] * W[c, r]
            # T[a, :, :] = sum_r U[a, r] M_r
            # This is a linear regression: T[a] = U[a] @ M_r?
            # T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
            # Flatten V*W: for each r, vector of length m*p*p*n? No, V*W is outer product.
            # Let's construct the matrix A for each a.
            # A is (m*p*p*n) x R.
            # A[b, c, r] = V[b, r] * W[c, r]
            # T_flat[a] = T[a, :, :].flatten()
            # Solve U[a, :] = (A^T A)^-1 A^T T_flat[a]
            
            # Precompute A? It changes when V or W changes.
            # Since m*p*p*n is small (9*6=54? No, 9*6=54 elements? No, b is 0..8, c is 0..5. 9*6=54. R is ~17.
            # A shape: (54, R).
            
            A = np.zeros((m * p * p * n, R))
            # Fill A
            # Index b in 0..m*p-1, c in 0..p*n-1
            # A[b * (p*n) + c, r] = V[b, r] * W[c, r]
            for b in range(m * p):
                for c in range(p * n):
                    for r in range(R):
                        A[b * (p * n) + c, r] = V[b, r] * W[c, r]
            
            A_t_A = A.T @ A
            A_t_A += 1e-6 * np.eye(R) # Regularization
            
            for a in range(n * m):
                T_a = T[a, :, :].flatten()
                A_t_T = A.T @ T_a
                try:
                    U[a, :] = np.linalg.solve(A_t_A, A_t_T)
                except np.linalg.LinAlgError:
                    pass
            
            # Update V
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed b: T[:, b, :] = sum_r V[b, r] * (U[:, r] * W[:, r])
            # A2 shape: (n*m*p*n, R)
            A2 = np.zeros((n * m * p * n, R))
            for a in range(n * m):
                for c in range(p * n):
                    for r in range(R):
                        A2[a * (p * n) + c, r] = U[a, r] * W[c, r]
            
            A2_t_A2 = A2.T @ A2
            A2_t_A2 += 1e-6 * np.eye(R)
            
            for b in range(m * p):
                T_b = T[:, b, :].flatten()
                A2_t_T = A2.T @ T_b
                try:
                    V[b, :] = np.linalg.solve(A2_t_A2, A2_t_T)
                except np.linalg.LinAlgError:
                    pass
            
            # Update W
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed c: T[:, :, c] = sum_r W[c, r] * (U[:, r] * V[:, r])
            A3 = np.zeros((n * m * m * p, R))
            for a in range(n * m):
                for b in range(m * p):
                    for r in range(R):
                        A3[a * (m * p) + b, r] = U[a, r] * V[b, r]
            
            A3_t_A3 = A3.T @ A3
            A3_t_A3 += 1e-6 * np.eye(R)
            
            for c in range(p * n):
                T_c = T[:, :, c].flatten()
                A3_t_T = A3.T @ T_c
                try:
                    W[c, :] = np.linalg.solve(A3_t_A3, A3_t_T)
                except np.linalg.LinAlgError:
                    pass
            
            # Check convergence / validity
            if it % 10 == 0:
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < 1e-6:
                    # Try to rationalize
                    U_rat = np.array([[Fraction(x).limit_denominator(100).numerator / Fraction(x).limit_denominator(100).denominator for x in row] for row in U])
                    V_rat = np.array([[Fraction(x).limit_denominator(100).numerator / Fraction(x).limit_denominator(100).denominator for x in row] for row in V])
                    W_rat = np.array([[Fraction(x).limit_denominator(100).numerator / Fraction(x).limit_denominator(100).denominator for x in row] for row in W])
                    
                    # Verify exactness with rationals
                    if validate_decomposition(U_rat, V_rat, W_rat):
                        return {"U": U_rat.tolist(), "V": V_rat.tolist(), "W": W_rat.tolist()}

        return None

    # Try ranks lower than schoolbook
    for R in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline:
            break
        res = try_rank(R)
        if res:
            return res

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
