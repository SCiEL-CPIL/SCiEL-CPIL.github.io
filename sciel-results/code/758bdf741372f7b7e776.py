import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def try_rank(r):
        if time.time() > deadline - 10.0:
            return None
            
        # Try random U, V
        # We allow up to 2000 attempts, but stop if time is low
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10.0:
                return None
                
            # Generate random sparse U, V
            # Use 0, 1, -1
            U_rand = np.random.randint(-1, 2, size=(n * m, r))
            V_rand = np.random.randint(-1, 2, size=(m * p, r))
            
            # Construct matrix M for W
            # M shape: (n*m*p, p*n*r)
            M = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(r):
                            coeff = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                M[eq_idx, var_idx] = coeff
            
            # Solve M w = b
            try:
                w, residuals, rank_M, s = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
                
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            # Check if w is close to half-integers
            w = w.reshape(p * n, r)
            w_2 = 2 * w
            if np.allclose(w_2, np.round(w_2), atol=1e-9):
                # Verify exactly
                U_frac = [[Fraction(int(x)) for x in row] for row in U_rand]
                V_frac = [[Fraction(int(x)) for x in row] for row in V_rand]
                W_frac = [[Fraction(int(round(2*x))) / 2 for x in row] for row in w]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return {
                        "U": [[int(x) for x in row] for row in U_rand],
                        "V": [[int(x) for x in row] for row in V_rand],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        return None

    # Try ranks from 59 down to 40
    # We can prioritize lower ranks if we have time, but usually lower rank is harder to find
    # So start from 59 and go down
    for r in range(59, 39, -1):
        res = try_rank(r)
        if res is not None:
            return res
            
    return {"U": U_sb, "V": V_sb, "W": W_sb}
