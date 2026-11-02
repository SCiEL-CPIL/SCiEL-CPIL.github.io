import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds margin
    
    # Try ranks from 13 to max_rank
    # We will try a few attempts for each rank
    # If we find a valid one, return it.
    # If not, return schoolbook.
    
    for r in range(13, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random attempts
        # For rank 18, we know schoolbook works, so we can skip random search for 18 and just return it?
        # Or try to find a non-trivial one?
        # Let's just try to find any valid one.
        
        attempts = 20 if r < max_rank else 1
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + r)
            
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            success = True
            for k in range(p):
                for i in range(n):
                    row_w = k * n + i
                    B = [[0] * r for _ in range(m)]
                    for j in range(m):
                        for c in range(r):
                            B[j][c] = U[i * m + j][c] * V[j * p + k][c]
                    
                    # Find 3 independent columns
                    cols_to_use = []
                    found = False
                    for c1 in range(r):
                        if found: break
                        for c2 in range(c1 + 1, r):
                            if found: break
                            for c3 in range(c2 + 1, r):
                                det = (B[0][c1] * (B[1][c2] * B[2][c3] - B[1][c3] * B[2][c2]) -
                                       B[0][c2] * (B[1][c1] * B[2][c3] - B[1][c3] * B[2][c1]) +
                                       B[0][c3] * (B[1][c1] * B[2][c2] - B[1][c2] * B[2][c1]))
                                if det != 0:
                                    cols_to_use = [c1, c2, c3]
                                    found = True
                                    break
                            if found: break
                        if found: break
                    
                    if not found:
                        success = False
                        break
                    
                    # Solve with Fraction
                    a00 = Fraction(B[0][cols_to_use[0]])
                    a01 = Fraction(B[0][cols_to_use[1]])
                    a02 = Fraction(B[0][cols_to_use[2]])
                    a10 = Fraction(B[1][cols_to_use[0]])
                    a11 = Fraction(B[1][cols_to_use[1]])
                    a12 = Fraction(B[1][cols_to_use[2]])
                    a20 = Fraction(B[2][cols_to_use[0]])
                    a21 = Fraction(B[2][cols_to_use[1]])
                    a22 = Fraction(B[2][cols_to_use[2]])
                    
                    det_f = (a00 * (a11 * a22 - a12 * a21) -
                             a01 * (a10 * a22 - a12 * a20) +
                             a02 * (a10 * a21 - a11 * a20))
                    
                    if det_f == 0:
                        success = False
                        break

                    w0_f = (Fraction(1) * (a11 * a22 - a12 * a21) - a01 * (Fraction(1) * a22 - a12 * Fraction(1)) + a02 * (Fraction(1) * a21 - a11 * Fraction(1))) / det_f
                    w1_f = (a00 * (Fraction(1) * a22 - a12 * Fraction(1)) - Fraction(1) * (a10 * a22 - a12 * a20) + a02 * (a10 * Fraction(1) - Fraction(1) * a20)) / det_f
                    w2_f = (a00 * (a11 * Fraction(1) - Fraction(1) * a21) - a01 * (a10 * Fraction(1) - Fraction(1) * a20) + Fraction(1) * (a10 * a21 - a11 * a20)) / det_f
                    
                    def is_half_int(f):
                        return f.denominator in [1, 2]
                        
                    if not (is_half_int(w0_f) and is_half_int(w1_f) and is_half_int(w2_f)):
                        success = False
                        break
                        
                    W[row_w][cols_to_use[0]] = float(w0_f)
                    W[row_w][cols_to_use[1]] = float(w1_f)
                    W[row_w][cols_to_use[2]] = float(w2_f)

            if not success:
                continue
                
            # Exact verification
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
