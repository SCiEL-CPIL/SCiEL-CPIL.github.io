import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21

    num_eqs = n * m * p  # 84
    min_rank = max(dim_U, dim_V, dim_W)  # 28

    # Fallback: Schoolbook
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verification function using exact Fraction arithmetic
    def verify(U, V, W2, r):
        # U, V, W2 are integer/half-integer arrays (W2 may be half-integers)
        # We'll use Fractions for exact verification
        # But for speed, let's first check with floats, then verify with Fractions only if close
        # Actually, let's just use Fractions directly since r is small and num_eqs is small
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(r):
                        total += Fraction(int(U[i*m+j, c]), 1) * Fraction(int(V[j*p+k, c]), 1) * W2[k*n+i, c]
                    if total != 1:
                        return False
        return True

    # Search for better rank
    best_result = None
    best_rank = 84

    for r in range(min_rank, 84):
        if time.time() > deadline - 5:
            break

        # Determine number of tries based on rank
        if r < 40:
            num_tries = 100
        elif r < 60:
            num_tries = 50
        else:
            num_tries = 20

        for attempt in range(num_tries):
            if time.time() > deadline - 2:
                break

            # Random initialization using {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(dim_U, r))
            V = np.random.choice([-1, 0, 1], size=(dim_V, r))

            # Decoupled system: for each (i,k), solve m x r system
            # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # Solve M @ w = ones(m)
            W_sol = np.zeros((dim_W, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build M: m x r
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve M @ w = ones(m)
                    try:
                        w, residuals, rank_sol, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except Exception:
                        valid = False
                        break
                    
                    if not valid:
                        break
                    
                    # Check residual
                    if residuals.size > 0 and residuals[0] > 1e-10:
                        valid = False
                        break
                    
                    # Check half-integer
                    W_scaled = w * 2
                    W_rounded = np.round(W_scaled)
                    if np.max(np.abs(W_scaled - W_rounded)) > 1e-6:
                        valid = False
                        break
                    
                    W_sol[k*n+i, :] = W_rounded / 2.0
            
            if not valid:
                continue
            
            # Now verify with exact arithmetic
            # Convert W_sol to Fractions
            W_frac = np.zeros((dim_W, r), dtype=object)
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        W_frac[k*n+i, c] = Fraction(int(W_sol[k*n+i, c] * 2), 2)
            
            if verify(U, V, W_frac, r):
                # Valid!
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_sol.tolist()
                }
                return result

    # If no better rank found, return schoolbook
    return get_schoolbook()
