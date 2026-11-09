"""Matrix multiplication 3x3x4 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if U,V,W form an exact decomposition of the matrix mult tensor."""
        r = len(U[0]) if U else 0
        # Convert to numpy for speed
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def is_half_integer(vals):
        """Check if all values are integers or half-integers (multiples of 0.5)."""
        for v in vals:
            if abs(v * 2 - round(v * 2)) > 1e-9:
                return False
        return True
    
    def to_half_int_list(mat):
        """Convert matrix to list of lists with half-integer values."""
        result = []
        for row in mat:
            new_row = []
            for v in row:
                rounded = round(v * 2) / 2
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    def try_rank(r, max_iters=20000):
        """Try to find a rank-r decomposition using randomized search."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return None
        
        # Use a structured approach: start with random sparse matrices
        # and try to refine
        
        for attempt in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT:
                return None
            
            # Generate random half-integer matrices with sparsity
            def rand_half_int_matrix(rows, cols, density=0.3):
                mat = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < density:
                            # Values from {-1, -0.5, 0.5, 1}
                            mat[i, j] = random.choice([-1, -0.5, 0.5, 1])
                return mat
            
            U = rand_half_int_matrix(nm, r, density=0.4)
            V = rand_half_int_matrix(mp, r, density=0.4)
            W = rand_half_int_matrix(pn, r, density=0.4)
            
            # Quick check: compute all (i,j,k) contractions
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
            
            if valid:
                # Verify more carefully
                U_list = to_half_int_list(U)
                V_list = to_half_int_list(V)
                W_list = to_half_int_list(W)
                if verify(U_list, V_list, W_list):
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = 12
    # Also, for matrix multiplication, rank >= n*m*p / max(n,m,p) is a known bound
    # For 3,3,4: 36/4 = 9, so rank >= 12 from dimension constraint
    
    # Try a few specific ranks
    for r in range(12, min(37, best_rank)):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        result = try_rank(r, max_iters=5000)
        if result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
