import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0 # 5s margin for 150s limit
    
    # Schoolbook as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fb

    # Try to find lower rank
    # We search from rank 104 down to 50
    # Using smaller entries {-1, 0, 1} increases chance of small denominator solutions
    for r in range(104, 49, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        # Increase attempts if r is larger (easier to satisfy)
        attempts = 50 if r > 80 else 20
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V with small integer entries
            # U shape: n*m x r
            # V shape: m*p x r
            # Using -1, 0, 1
            U_rand = np.random.randint(-1, 2, (n * m, r))
            V_rand = np.random.randint(-1, 2, (m * p, r))
            
            # Reshape to 3D for easier indexing
            U_3d = U_rand.reshape(n, m, r)
            V_3d = V_rand.reshape(m, p, r)
            
            W_3d = np.zeros((p, n, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A matrix of size m x r
                    # A[j, c] = U_3d[i, j, c] * V_3d[j, k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U_3d[i, j, c] * V_3d[j, k, c]
                    
                    # Solve A @ w = 1
                    # w is W_3d[k, i, :]
                    try:
                        # Use least squares to find a solution
                        w, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Check if it's an exact solution
                        if not np.allclose(A @ w, np.ones(m), atol=1e-9):
                            success = False
                            break
                        
                        # Check if w is close to half-integers
                        # 2*w should be close to integers
                        w_scaled = 2 * w
                        w_int = np.round(w_scaled)
                        
                        if not np.allclose(w_scaled, w_int, atol=1e-6):
                            success = False
                            break
                            
                        # Store the exact half-integer values as Fractions
                        # w_int / 2
                        for c in range(r):
                            W_3d[k, i, c] = w_int[c] / 2.0
                            
                    except:
                        success = False
                        break
                if not success:
                    break
            
            if success:
                # We have a candidate solution with half-integer W.
                # Convert to Fractions for exact verification
                U_frac = [[Fraction(U_rand[idx, c]) for c in range(r)] for idx in range(n * m)]
                V_frac = [[Fraction(V_rand[idx, c]) for c in range(r)] for idx in range(m * p)]
                W_frac = [[Fraction(int(W_3d[k, i, c] * 2)) / 2 for c in range(r)] for k in range(p) for i in range(n)]
                
                # Verify the decomposition with Fractions
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(r):
                                s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = r
                    return best_result
                else:
                    # Verification failed, continue searching
                    pass

    return best_result
