"""Strassen-based rank decomposition for 2x4x4 matrix multiplication.

Uses 4 independent 2x2x2 Strassen subproblems (rank 28) then attempts
greedy rank reduction by removing linearly dependent terms.
"""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    r = U.shape[1]
    if r == 0:
        return False
    T = np.ones((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T_computed, T, atol=tol)

def _strassen_28():
    """
    Build Strassen-based decomposition for C = A * B where A is 2x4, B is 4x4, C is 2x4.
    
    Split into 4 subproblems of size 2x2x2:
    - A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    - B_top = B[0:2, :] (2x4), B_bot = B[2:4, :] (2x4)
    - B_top_left = B[0:2, 0:2], B_top_right = B[0:2, 2:4]
    - B_bot_left = B[2:4, 0:2], B_bot_right = B[2:4, 2:4]
    
    C = A_left * B_top + A_right * B_bot
    C_left (2x2) = A_left * B_top_left + A_right * B_bot_left
    C_right (2x2) = A_left * B_top_right + A_right * B_bot_right
    
    Apply Strassen (7 terms) to each of the 4 2x2x2 subproblems.
    Total rank: 4 * 7 = 28
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen 7 terms for 2x2x2:
    # M0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M1 = (A[1,0]+A[1,1])*B[0,0]
    # M2 = A[0,0]*(B[0,1]-B[1,1])
    # M3 = A[1,1]*(B[1,0]-B[0,0])
    # M4 = (A[0,0]+A[0,1])*B[1,1]
    # M5 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # 
    # C[0,0] = M0+M3-M4+M6
    # C[0,1] = M2+M4
    # C[1,0] = M1+M3
    # C[1,1] = M0-M1+M2+M5
    
    # U coefficients (A factor): for each term, list of (i_local, j_local, coeff)
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    
    # V coefficients (B factor): for each term, list of (j_local, k_local, coeff)
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (1,1,1)],      # M6
    ]
    
    # W coefficients (C factor): for each term, list of (i_local, k_local, coeff)
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],    # M0: C00+=1, C11+=1
        1: [(1,0,1), (1,1,-1)],   # M1: C10+=1, C11+=-1
        2: [(0,1,1), (1,1,1)],    # M2: C01+=1, C11+=1
        3: [(0,0,1), (1,0,1)],    # M3: C00+=1, C10+=1
        4: [(0,0,-1), (0,1,1)],   # M4: C00+=-1, C01+=1
        5: [(1,1,1)],             # M5: C11+=1
        6: [(0,0,1)],             # M6: C00+=1
    }
    
    col = 0
    for jp in range(2):  # j_pair: 0={0,1}, 1={2,3}
        for kp in range(2):  # k_pair: 0={0,1}, 1={2,3}
            for t in range(7):
                # U coefficients
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                # V coefficients
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                # W coefficients
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
    """Try to reduce rank by removing terms that are linear combinations of others."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for iteration in range(max_iterations):
        if _remaining() < 15.0:
            break
        if best_rank <= 14:
            break
        
        removed_any = False
        r = best_U.shape[1]
        
        for c in range(r):
            if _remaining() < 15.0:
                break
            
            # Try removing column c
            mask = [i for i in range(r) if i != c]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            if _verify(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = len(mask)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return best_U, best_V, best_W, best_rank

def _to_list_of_lists(mat):
    return [[float(x) for x in row] for row in mat]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 2, 4, 4
    
    if _remaining() < 10.0:
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    # Build Strassen decomposition
    U, V, W = _strassen_28()
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    # Try to reduce rank
    U, V, W, final_rank = _try_reduce_rank(U, V, W, n, m, p)
    
    # Final verification
    if _verify(U, V, W, n, m, p):
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    else:
        # Fallback to original 28-term
        U, V, W = _strassen_28()
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
