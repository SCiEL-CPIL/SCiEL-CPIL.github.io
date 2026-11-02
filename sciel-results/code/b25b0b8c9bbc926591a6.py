import numpy as np
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor and check
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = T[i*m+j, j*p+k, k*n+i]
                err = abs(actual - 1.0)
                if err > max_err:
                    max_err = err
    
    # Also check that all other entries are 0
    for idx_i in range(n*m):
        i = idx_i // m
        j = idx_i % m
        for idx_j in range(m*p):
            j2 = idx_j // p
            k = idx_j % p
            for idx_k in range(p*n):
                k2 = idx_k // n
                i2 = idx_k % n
                if j == j2 and i == i2 and k == k2:
                    continue
                err = abs(T[idx_i, idx_j, idx_k])
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-10

def _build_base_decomposition():
    """Build the rank-44 Strassen+trivial decomposition."""
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m2 = (c+d)e
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            W[(b_c*2+1)*n + 1][col] = -1
            col += 1
            
            # m3 = a(f-h)
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m4 = d(g-e)
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            col += 1
            
            # m5 = (a+b)h
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            col += 1
            
            # m6 = (c-a)(e+f)
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m7 = (b-d)(g+h)
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return U, V, W

def _try_reduce_rank(base_rank, max_attempts=200):
    """Try to find a lower-rank decomposition by zeroing out columns."""
    n, m, p = 3, 4, 4
    
    U, V, W = _build_base_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    best_U, best_V, best_W = U, V, W
    best_rank = base_rank
    
    # Try removing one column at a time
    for col_to_remove in range(base_rank):
        if _time_left() < 5:
            break
        
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Remove the column
        U_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_test]
        V_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_test]
        W_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_test]
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = base_rank - 1
            # Reset base for next iteration
            U, V, W = U_test, V_test, W_test
            base_rank = best_rank
            break
    
    # Try removing two columns at a time from the reduced decomposition
    for i in range(best_rank):
        if _time_left() < 5:
            break
        for j in range(i+1, best_rank):
            if _time_left() < 3:
                break
            
            U_test = [row[:] for row in U]
            V_test = [row[:] for row in V]
            W_test = [row[:] for row in W]
            
            # Remove columns i and j (j first if j > i to keep indices valid)
            for row in U_test:
                del row[j]
                del row[i]
            for row in V_test:
                del row[j]
                del row[i]
            for row in W_test:
                del row[j]
                del row[i]
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = best_rank - 2
                U, V, W = U_test, V_test, W_test
                break
    
    if best_rank < base_rank:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = 3, 4, 4
    
    # Build the rank-44 decomposition
    U, V, W = _build_base_decomposition()
    
    # Try to reduce rank
    reduced = _try_reduce_rank(44)
    if reduced is not None:
        U, V, W = reduced
    
    return {"U": U, "V": V, "W": W}
