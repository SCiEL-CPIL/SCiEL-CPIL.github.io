"""Rewritten program for matrix multiplication tensor decomposition (3,3,5)."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    """Construct a rank-minimal tensor decomposition for (3,3,5) matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (9, rank)
    - V has shape (m*p, rank) = (15, rank)  
    - W has shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    where T[i,j,k] = 1 iff i = a*m+b, j = b*p+c, k = c*n+a for some a,b,c.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    I, J, K = n * m, m * p, p * n  # 9, 15, 15
    
    def build_tensor():
        """Build the (n,m,p) matrix multiplication tensor."""
        T = np.zeros((I, J, K))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1
        return T
    
    T = build_tensor()
    
    def verify(U, V, W, atol=1e-10):
        """Verify that U, V, W form a valid decomposition."""
        rank = U.shape[1]
        recon = np.zeros((I, J, K))
        for c in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(recon, T, atol=atol)
    
    def to_rational_matrix(M, tol=1e-9):
        """Convert a float matrix to rational (Fraction) matrix."""
        rows, cols = M.shape
        R = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                v = M[i, j]
                if abs(v) < tol:
                    R[i][j] = Fraction(0)
                else:
                    # Try to find a simple fraction
                    rounded = round(v * 2) / 2
                    if abs(v - rounded) < tol:
                        R[i][j] = Fraction(rounded).limit_denominator(100)
                    else:
                        R[i][j] = Fraction(v).limit_denominator(1000)
        return R
    
    def schoolbook():
        """Schoolbook decomposition (rank = n*m*p = 45)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(I)]
        V = [[0] * rank for _ in range(J)]
        W = [[0] * rank for _ in range(K)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def als_decompose(r, max_iter=200, n_restarts=5, seed=42):
        """Alternating least squares for tensor rank-r decomposition.
        
        Args:
            r: target rank
            max_iter: max iterations per restart
            n_restarts: number of random restarts
            seed: random seed
            
        Returns:
            Tuple (U, V, W) if success, else None
        """
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(seed + restart)
            U = rng.randn(I, r)
            V = rng.randn(J, r)
            W = rng.randn(K, r)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return best_U, best_V, best_W
                
                # Update U
                for i in range(I):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for j in range(J):
                        for k in range(K):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * V[j, :] * W[k, :]
                                den += V[j, :]**2 * W[k, :]**2
                    mask = den > 1e-12
                    U[i, mask] = num[mask] / den[mask]
                    U[i, ~mask] = 0
                
                # Update V
                for j in range(J):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(I):
                        for k in range(K):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * U[i, :] * W[k, :]
                                den += U[i, :]**2 * W[k, :]**2
                    mask = den > 1e-12
                    V[j, mask] = num[mask] / den[mask]
                    V[j, ~mask] = 0
                
                # Update W
                for k in range(K):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(I):
                        for j in range(J):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * U[i, :] * V[j, :]
                                den += U[i, :]**2 * V[j, :]**2
                    mask = den > 1e-12
                    W[k, mask] = num[mask] / den[mask]
                    W[k, ~mask] = 0
                
                # Check convergence
                recon = np.zeros((I, J, K))
                for c in range(r):
                    recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                err = np.linalg.norm(recon - T) / max(1, np.linalg.norm(T))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    return U, V, W
        
        return best_U, best_V, best_W
    
    def try_polynomial_system(r, max_iterations=200):
        """Try to solve the polynomial system for rank-r decomposition.
        
        Uses a grid search over structured initializations.
        """
        best_result = None
        best_err = float('inf')
        
        # Try structured initializations based on the algebraic structure
        # The tensor has a specific pattern: T[i,j,k] = 1 iff i=a*m+b, j=b*p+c, k=c*n+a
        
        for init_type in range(5):
            if time.time() > deadline:
                break
                
            if init_type == 0:
                # Random
                rng = np.random.RandomState(42)
                U = rng.randn(I, r)
                V = rng.randn(J, r)
                W = rng.randn(K, r)
            elif init_type == 1:
                # Block-diagonal style
                U = np.zeros((I, r))
                V = np.zeros((J, r))
                W = np.zeros((K, r))
                # Assign each (a,b,c) triple to a term
                idx = 0
                for a in range(n):
                    for b in range(m):
                        for c in range(p):
                            if idx < r:
                                i = a * m + b
                                j = b * p + c
                                k = c * n + a
                                U[i, idx] = 1
                                V[j, idx] = 1
                                W[k, idx] = 1
                                idx += 1
            elif init_type == 2:
                # Sparse random
                rng = np.random.RandomState(43)
                U = np.zeros((I, r))
                V = np.zeros((J, r))
                W = np.zeros((K, r))
                for c in range(r):
                    # Each term touches a few entries
                    for _ in range(3):
                        U[rng.randint(I), c] = rng.choice([-1, 1])
                        V[rng.randint(J), c] = rng.choice([-1, 1])
                        W[rng.randint(K), c] = rng.choice([-1, 1])
            elif init_type == 3:
                # Low-rank structure from Strassen-like approach
                # Use 2x2x2 Strassen (rank 7) as building block
                U = np.zeros((I, r))
                V = np.zeros((J, r))
                W = np.zeros((K, r))
                # Place Strassen blocks
                # This is complex; use random for now
                rng = np.random.RandomState(44)
                U = rng.randn(I, r)
                V = rng.randn(J, r)
                W = rng.randn(K, r)
                # Scale
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
            else:
                # Fourier-like initialization
                U = np.zeros((I, r))
                V = np.zeros((J, r))
                W = np.zeros((K, r))
                for c in range(r):
                    for i in range(I):
                        U[i, c] = np.cos(2 * np.pi * i * c / r)
                    for j in range(J):
                        V[j, c] = np.cos(2 * np.pi * j * c / r)
                    for k in range(K):
                        W[k, c] = np.cos(2 * np.pi * k * c / r)
                
                # Run ALS from this initialization
                result = als_decompose(r, max_iter=200, n_restarts=1, 
                                      seed=100 + init_type)
                if result is not None:
                    U_res, V_res, W_res = result
                    recon = np.zeros((I, J, K))
                    for c in range(r):
                        recon += np.einsum('i,j,k->ijk', U_res[:, c], V_res[:, c], W_res[:, c])
                    err = np.linalg.norm(recon - T)
                    if err < best_err:
                        best_err = err
                        best_result = (U_res, V_res, W_res)
                
                if time.time() > deadline:
                    break
        
        if best_result is not None and best_err < 1e-8:
            return best_result
        
        return None
    
    # Main search logic
    # Known bounds for rank(3,3,5):
    # - Lower bound: at least 30 (from various lower bound techniques)
    # - Upper bound: 45 (schoolbook)
    # - Best known: around 30-33
    
    # Try ranks from low to high
    for r in range(28, 46):
        if time.time() > deadline - 20:
            break
        
        # Try ALS-based approach
        result = try_polynomial_system(r)
        if result is not None:
            U_float, V_float, W_float = result
            # Convert to rationals
            U_r = to_rational_matrix(U_float)
            V_r = to_rational_matrix(V_float)
            W_r = to_rational_matrix(W_float)
            
            # Verify
            if verify(U_r, V_r, W_r):
                return {"U": U_r, "V": V_r, "W": W_r}
            
            # Try with the float values directly (as a fallback)
            # But we need exact rational/half-integer entries
            # Check if the float solution is already close to rational
            is_rational = True
            for mat in [U_float, V_float, W_float]:
                for row in mat:
                    for val in row:
                        rounded = round(val * 2) / 2
                        if abs(val - rounded) > 1e-6:
                            is_rational = False
                            break
                    if not is_rational:
                        break
                if not is_rational:
                    break
            
            if is_rational:
                # Convert to exact half-integers
                U_exact = [[round(val * 2) / 2 for val in row] for row in U_float]
                V_exact = [[round(val * 2) / 2 for val in row] for row in V_float]
                W_exact = [[round(val * 2) / 2 for val in row] for row in W_float]
                
                if verify(U_exact, V_exact, W_exact):
                    return {"U": U_exact, "V": V_exact, "W": W_exact}
        
        # Also try direct ALS
        result = als_decompose(r, max_iter=300, n_restarts=5, seed=r)
        if result is not None:
            U_res, V_res, W_res = result
            recon = np.zeros((I, J, K))
            for c in range(r):
                recon += np.einsum('i,j,k->ijk', U_res[:, c], V_res[:, c], W_res[:, c])
            err = np.linalg.norm(recon - T)
            if err < 1e-8:
                # Check if rational
                is_rational = True
                for mat in [U_res, V_res, W_res]:
                    for row in mat:
                        for val in row:
                            rounded = round(val * 2) / 2
                            if abs(val - rounded) > 1e-6:
                                is_rational = False
                                break
                        if not is_rational:
                            break
                    if not is_rational:
                        break
                
                if is_rational:
                    U_exact = [[round(val * 2) / 2 for val in row] for row in U_res]
                    V_exact = [[round(val * 2) / 2 for val in row] for row in V_res]
                    W_exact = [[round(val * 2) / 2 for val in row] for row in W_res]
                    if verify(U_exact, V_exact, W_exact):
                        return {"U": U_exact, "V": V_exact, "W": W_exact}
    
    # Fallback: schoolbook rank 45
    return schoolbook()

# EVOLVE-BLOCK-END
