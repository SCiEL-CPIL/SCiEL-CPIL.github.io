"""Matrix multiplication 3x5x7 - rank minimization via algebraic search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - target) > 1e-12:
                    return False
    return True

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    n, m, p = 3, 5, 7
    rank = U.shape[1]
    # Build the multiplication tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > 1e-10:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    
    # Schoolbook baseline: rank 105
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
    
    # Known lower bound: Strassen-like or other algorithms
    # For 3x5x7, the minimum rank is known to be less than 105.
    # Let's try to find it via randomized algebraic methods.
    
    # Strategy: Use the fact that the multiplication tensor has structure.
    # Try ranks below 105 and see if we can find a valid decomposition.
    
    # First, let's establish what ranks are feasible.
    # The tensor is (n*m) x (m*p) x (p*n) = 15 x 35 x 21
    # We need U (15 x r), V (35 x r), W (21 x r) such that
    # for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Approach: Try to find solutions using the structure of the problem.
    # Key insight: The multiplication tensor for C = A*B where A is 3x5, B is 5x7, C is 3x7
    # has the property that C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's try a structured search using the known properties.
    # One approach: use the fact that we can "factor" the middle dimension.
    
    # Try rank values from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n, ...) but more carefully
    # For matrix multiplication, known bounds exist.
    # Let's just search from reasonable values.
    
    # Actually, let me think about this differently.
    # The standard result: M(n,m,p) for matrix multiplication.
    # M(3,5,7) - let's see if we can exploit structure.
    
    # One key idea: if we can find a decomposition where some factors are zero,
    # we can reduce the effective rank.
    
    # Let's try a direct algebraic approach with random initialization and solving.
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # We need to solve for U, V, W with integer/half-integer entries.
        # This is a system of n*m*p = 105 equations with 3*r unknowns (approximately).
        
        # Randomized approach: fix two matrices, solve for the third.
        # But the equations are cubic (products of three terms), so it's nonlinear.
        
        # Alternative: use the bilinear structure.
        # For each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        # So sum_j U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] over c = 1 for each (i,j,k)
        
        # Let's try: fix V and W randomly, then for each (i,j,k), we need
        # sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
        # This is a linear system in U if V and W are fixed!
        
        # For fixed V, W:
        # For each i (0..n-1), we have m*p = 35 equations in m = 5 unknowns (U[i*m+0..i*m+4, :])
        # Wait, U has n*m = 15 rows, each of length r.
        # For fixed i, the rows are U[i*m+j, :] for j=0..m-1.
        # The equations for fixed i are: for each (j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # That's p*m = 35 equations for m=5 rows of U (each row has r entries).
        # So 35 equations, 5*r unknowns. For this to be solvable, we need enough degrees of freedom.
        
        # Actually, let me reconsider. For fixed V and W, the constraint is:
        # For all (i,j,k): sum_c U[i*m+j,c] * M[i,j,k,c] = 1
        # where M[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        
        # Grouping by i: for each i, we have m*p = 35 equations.
        # Unknowns for this i: U[i*m+0, :], U[i*m+1, :], ..., U[i*m+(m-1), :] - that's m vectors of length r.
        # So m*r = 5r unknowns, 35 equations.
        # For a solution to exist, we need the system to be consistent.
        
        # Let's try: pick random integer V and W, then solve for U via least squares and check.
        
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Generate random V and W with small integer entries
            V = [[random.choice([-2,-1,0,1,2]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-2,-1,0,1,2]) for _ in range(r)] for _ in range(p*n)]
            
            # For each i, solve for U[i*m+j, :] for j=0..m-1
            # The system: for each (j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is linear in U[i*m+j, :]
            
            U = [[0]*r for _ in range(n*m)]
            
            valid = True
            for i in range(n):
                # Build the system for this i
                # Unknowns: U[i*m+j, c] for j in range(m), c in range(r) -> m*r unknowns
                # Equations: for j in range(m), k in range(p):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                
                # Solve least squares
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    valid = False
                    break
                
                # Check if solution is close to integer/half-integer
                sol_rounded = np.round(sol * 2) / 2
                if np.max(np.abs(sol - sol_rounded)) > 1e-6:
                    # Try to find nearby half-integers
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Verify the full solution
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions
                U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(p*n)]
                
                # Double check with fractions
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(r))
                            if val != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But matrix multiplication requires more. Known: M(2,2,2)=7, M(3,3,3) >= 19
    # For 3x5x7, let's try from ~40 upward
    
    # Actually, let's be smarter. Try specific ranks.
    for r in range(35, 106):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
