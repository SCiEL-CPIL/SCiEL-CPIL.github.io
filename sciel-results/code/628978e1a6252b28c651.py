import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank_vandermonde(r, num_attempts=50, seed_base=0):
        """
        Try to find a decomposition of rank r using Vandermonde-like structures.
        This often leads to solutions with small integer coefficients.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            random.seed(seed_base + r * 1000 + attempt)
            
            # Generate V and W using Vandermonde matrices with small integer bases
            # This structure often results in linearly independent rows and
            # solutions for U that are rational with small denominators.
            
            # Choose a base for Vandermonde, e.g., 2 or 3
            base = random.choice([2, 3, 4])
            
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            # Fill V
            for idx in range(mp):
                for c in range(r):
                    # Use a pattern that ensures rows are distinct
                    # V[idx, c] = base ** c is too large. 
                    # Let's use random small integers but with some structure.
                    # Actually, pure random {-1, 0, 1} was tried.
                    # Let's try random integers in [-2, 2].
                    V[idx][c] = random.randint(-2, 2)
                    
            # Fill W
            for idx in range(pn):
                for c in range(r):
                    W[idx][c] = random.randint(-2, 2)
            
            U = [[0.0]*r for _ in range(nm)]
            valid = True
            
            # Solve for U block by block (for each i)
            for i in range(n):
                # System: A * u_i = b
                # u_i is a vector of length m*r (flattened U rows for fixed i)
                # A is m*p x m*r
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                
                try:
                    # Use least squares
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to integers or half-integers
                # Try integers
                sol_int = np.round(sol)
                diff_int = np.max(np.abs(sol - sol_int))
                
                # Try half-integers
                sol_half = np.round(sol * 2.0) / 2.0
                diff_half = np.max(np.abs(sol - sol_half))
                
                if diff_int < diff_half:
                    sol_rounded = sol_int
                    diff = diff_int
                else:
                    sol_rounded = sol_half
                    diff = diff_half
                
                if diff > 1e-8:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Convert to Fractions for exact verification
            U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
            V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
            W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }
        
        return None

    def try_rank_sparse(r, num_attempts=50, seed_base=10000):
        """
        Try sparse random matrices.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            random.seed(seed_base + r * 1000 + attempt)
            
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            # Sparse: only ~30% non-zeros
            density = 0.3
            for idx in range(mp):
                for c in range(r):
                    if random.random() < density:
                        V[idx][c] = random.choice([-1, 1])
                    else:
                        V[idx][c] = 0
                        
            for idx in range(pn):
                for c in range(r):
                    if random.random() < density:
                        W[idx][c] = random.choice([-1, 1])
                    else:
                        W[idx][c] = 0

            U = [[0.0]*r for _ in range(nm)]
            valid = True
            
            for i in range(n):
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                sol_int = np.round(sol)
                diff_int = np.max(np.abs(sol - sol_int))
                
                sol_half = np.round(sol * 2.0) / 2.0
                diff_half = np.max(np.abs(sol - sol_half))
                
                if diff_int < diff_half:
                    sol_rounded = sol_int
                    diff = diff_int
                else:
                    sol_rounded = sol_half
                    diff = diff_half
                
                if diff > 1e-8:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
            V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
            W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }
        
        return None

    best_result = None
    best_rank = n * m * p
    
    # Try ranks from a lower bound upwards
    # Lower bound is 35. Let's try a few values.
    # If we find one, we break.
    for r in range(40, 106):
        if time.time() > deadline - 5:
            break
            
        # Try sparse first, it's faster to solve
        result = try_rank_sparse(r, num_attempts=20, seed_base=10000)
        if result is not None:
            best_result = result
            best_rank = r
            break
            
        # Try dense random
        if time.time() < deadline - 10:
            result = try_rank_vandermonde(r, num_attempts=20, seed_base=0)
            if result is not None:
                best_result = result
                best_rank = r
                break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
