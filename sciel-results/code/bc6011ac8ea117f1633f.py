"""Matrix multiplication 3x5x7 - rank minimization via structured search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
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
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_ladder_rank(r, num_seeds=50):
        """
        Ladder method: Start from schoolbook and try to reduce rank by 1.
        Fix V and W to have structure that allows solving for U with half-integers.
        
        Strategy: Use V and W with entries in {-1, 0, 1} and solve for U.
        The key is to create a system that has a unique half-integer solution.
        """
        if time.time() > deadline:
            return None
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                return None
            
            random.seed(99999 + r * 1000 + seed)
            
            # Generate sparse V and W to make the system more likely to have half-integer solutions
            # Use a structured pattern: each row has exactly 2 non-zero entries
            V = [[0] * r for _ in range(m*p)]
            W = [[0] * r for _ in range(p*n)]
            
            # For V: each row (j*p+k) gets 2 random columns with values +/-1
            for row in range(m*p):
                cols = random.sample(range(r), 2)
                V[row][cols[0]] = random.choice([-1, 1])
                V[row][cols[1]] = random.choice([-1, 1])
            
            # For W: each row (k*n+i) gets 2 random columns with values +/-1
            for row in range(p*n):
                cols = random.sample(range(r), 2)
                W[row][cols[0]] = random.choice([-1, 1])
                W[row][cols[1]] = random.choice([-1, 1])
            
            # Now solve for U
            U = [[0]*r for _ in range(n*m)]
            valid = True
            
            for i in range(n):
                # Build system for this i
                # Unknowns: U[i*m+j, c] for j in 0..m-1, c in 0..r-1
                # Equations: for each (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                
                # Check if system is consistent
                try:
                    # First check rank
                    rank_A = np.linalg.matrix_rank(A_mat)
                    if rank_A < m*p:
                        # System may be underdetermined; use least squares
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)[0]
                    else:
                        # Try to solve exactly
                        sol = np.linalg.solve(A_mat, b_vec)
                except:
                    valid = False
                    break
                
                # Check for half-integer solution
                sol_rounded = np.round(sol * 2) / 2
                max_err = np.max(np.abs(sol - sol_rounded))
                
                if max_err > 1e-8:
                    # Try a different approach: use integer linear programming-like search
                    # Round to nearest half-integer and verify
                    candidate = sol_rounded
                    # Verify this candidate
                    for j in range(m):
                        for k in range(p):
                            eq_idx = j*p + k
                            val = 0.0
                            for c in range(r):
                                val += candidate[j*r + c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Full verification
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions
                U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(p*n)]
                
                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None
    
    def try_structured_rank(r):
        """
        Use a more structured approach: exploit the tensor structure.
        For matrix multiplication, we can use block structure.
        """
        if time.time() > deadline:
            return None
        
        # Try different block structures
        # For 3x5x7, we can think of it as blocks
        # Let's try a rank-104 construction by modifying the schoolbook
        
        # Start with schoolbook and try to merge two terms
        # This is equivalent to finding a linear dependence
        
        # Alternative: use the fact that for small matrices, we can do exhaustive search
        # over a subset of parameters
        
        # For now, try the ladder method with different sparsity patterns
        for sparsity in [1, 2, 3]:
            if time.time() > deadline:
                return None
            
            result = _try_sparse_ladder(r, sparsity)
            if result is not None:
                return result
        
        return None
    
    def _try_sparse_ladder(r, nnz_per_row):
        """Try ladder method with specific sparsity."""
        if time.time() > deadline:
            return None
        
        num_seeds = 30
        for seed in range(num_seeds):
            if time.time() > deadline:
                return None
            
            random.seed(77777 + r * 1000 + nnz_per_row * 100 + seed)
            
            V = [[0] * r for _ in range(m*p)]
            W = [[0] * r for _ in range(p*n)]
            
            for row in range(m*p):
                cols = random.sample(range(r), min(nnz_per_row, r))
                for c in cols:
                    V[row][c] = random.choice([-1, 1])
            
            for row in range(p*n):
                cols = random.sample(range(r), min(nnz_per_row, r))
                for c in cols:
                    W[row][c] = random.choice([-1, 1])
            
            U = [[0]*r for _ in range(n*m)]
            valid = True
            
            for i in range(n):
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)[0]
                except:
                    valid = False
                    break
                
                sol_rounded = np.round(sol * 2) / 2
                if np.max(np.abs(sol - sol_rounded)) > 1e-8:
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            if _verify_np(U, V, W, n, m, p):
                U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(p*n)]
                
                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None
    
    # Try to find rank < 105
    for r in range(104, 99, -1):
        if time.time() > deadline:
            break
        
        # Try ladder method
        result = try_ladder_rank(r, num_seeds=20)
        if result is not None:
            return result
        
        # Try structured method
        result = try_structured_rank(r)
        if result is not None:
            return result
    
    return schoolbook()

# EVOLVE-BLOCK-END
