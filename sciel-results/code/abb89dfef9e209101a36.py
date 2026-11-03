import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try to find a decomposition with rank r
    # Using a structured approach: try to find U, V, W such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    def try_rank(r, num_attempts=20):
        """Try to find a rank-r decomposition."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize U and V with random integers in {-1, 0, 1}
            # Use structured patterns to increase success probability
            if attempt < 5:
                # Try structured patterns
                U_np = np.zeros((n * m, r), dtype=int)
                V_np = np.zeros((m * p, r), dtype=int)
                # Fill with a pattern
                for c in range(r):
                    U_np[:, c] = (np.arange(n * m) % (c + 1)) - ((c + 1) // 2)
                    V_np[:, c] = (np.arange(m * p) % (c + 2)) - ((c + 2) // 2)
            else:
                # Random initialization
                U_np = np.random.choice([-1, 0, 1], size=(n * m, r))
                V_np = np.random.choice([-1, 0, 1], size=(m * p, r))
            
            # Build the linear system for W
            # Variables: W[k*n + i, c] for k in range(p), i in range(n), c in range(r)
            # Total variables: p * n * r
            num_vars = p * n * r
            num_eqs = n * m * p
            
            # Build A matrix where A[eq_idx, var_idx] = U[i*m+j, c] * V[j*p+k, c]
            # eq_idx = i * m * p + j * p + k
            # var_idx = (k * n + i) * r + c
            
            A = np.zeros((num_eqs, num_vars), dtype=float)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(r):
                            coeff = U_np[i * m + j, c] * V_np[j * p + k, c]
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx, var_idx] = coeff
            
            b = np.ones(num_eqs)
            
            # Solve using least squares (since system is underdetermined)
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is close to exact
            residual = np.abs(A @ x - b).max()
            if residual > 1e-6:
                continue
            
            # Round to nearest integer/half-integer
            W_vals = np.round(x * 2) / 2
            
            # Verify with rounded values
            W_np = W_vals.reshape(p * n, r)
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                        error = max(error, abs(s - 1.0))
            
            if error > 1e-9:
                continue
            
            # Convert to Fractions for exact representation
            U = [[Fraction(int(U_np[i, c])) for c in range(r)] for i in range(n * m)]
            V = [[Fraction(int(V_np[i, c])) for c in range(r)] for i in range(m * p)]
            W = [[Fraction(W_np[i, c]) for c in range(r)] for i in range(p * n)]
            
            # Final exact verification
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if s != 1:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 18, but likely higher
    # Start from 20 and go up to schoolbook rank
    schoolbook_rank = n * m * p
    
    for r in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
