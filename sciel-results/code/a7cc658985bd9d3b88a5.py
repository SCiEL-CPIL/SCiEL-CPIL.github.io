"""Program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        num_u = len(U)
        num_v = len(V)
        num_w = len(W)
        if num_u != n * m or num_v != m * p or num_w != p * n:
            return False
        
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 1:
                        return False
        return True
    
    def try_rank_randomized(r, n, m, p, max_attempts=100):
        """
        Try to find a rank-r decomposition using randomized approach.
        Fix U and V randomly, then solve linear system for W.
        """
        for attempt in range(max_attempts):
            # Initialize U and V with small random rational values
            # Use values from {-1, 0, 1} to keep things simple
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            
            # Build linear system for W
            # For each output index c_idx = k*n + i (0 to p*n-1),
            # we need: sum_{j=0}^{m-1} sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[c_idx, r] = 1
            # This is a system of (p*n) equations in (p*n * r) unknowns (W entries)
            # But actually, each equation involves only r unknowns (one row of W)
            # Wait, no: for fixed (i,k), the equation is:
            # sum_{j=0}^{m-1} sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # So for each (i,k) pair, we have one equation in r unknowns W[k*n+i, 0..r-1]
            
            # Number of (i,k) pairs: n * p = 14
            # Unknowns: p * n * r = 14 * r
            # So the system is underdetermined if r > 1
            
            # Let's build the system properly
            # Unknowns: W[c, r] for c in 0..p*n-1, r in 0..r-1
            # Total unknowns: p*n*r
            # Equations: for each (i,j,k) triple, one equation
            # But wait, the equation for (i,j,k) is:
            # sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This couples U[i*m+j], V[j*p+k], and W[k*n+i]
            
            # Actually, let me reconsider. For fixed U and V, the equations are linear in W.
            # For each triple (i,j,k), we have:
            # sum_{r=0}^{r-1} (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # Let A_{(i,j,k), (c,r)} = U[i*m+j, r] * V[j*p+k, r] if c = k*n+i, else 0
            # Then A * W_vec = 1_vec
            
            # This is a system of n*m*p = 42 equations in p*n*r unknowns.
            
            # Let's use numpy to solve this least-squares problem
            num_eq = n * m * p
            num_unk = p * n * r
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c_idx = k * n + i
                        for r_idx in range(r):
                            col_idx = c_idx * r + r_idx
                            A[eq_idx, col_idx] = U[i * m + j][r_idx] * V[j * p + k][r_idx]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is exact
            if len(residuals) > 0 and residuals[0] > 1e-10:
                continue
            
            # Reshape W
            W = [[0] * r for _ in range(p * n)]
            for c_idx in range(p * n):
                for r_idx in range(r):
                    W[c_idx][r_idx] = W_flat[c_idx * r + r_idx]
            
            # Try to round to rationals (half-integers)
            # First check if floating point solution is close to half-integers
            valid = True
            for c_idx in range(p * n):
                for r_idx in range(r):
                    val = W[c_idx][r_idx]
                    # Check if close to a half-integer
                    nearest_half = round(val * 2) / 2
                    if abs(val - nearest_half) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Round to half-integers
            for c_idx in range(p * n):
                for r_idx in range(r):
                    W[c_idx][r_idx] = round(W[c_idx][r_idx] * 2) / 2
            
            # Verify the decomposition
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from schoolbook down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = 21
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few ranks below schoolbook
    for r in range(schoolbook_rank, lower_bound - 1, -1):
        result = try_rank_randomized(r, n, m, p, max_attempts=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
