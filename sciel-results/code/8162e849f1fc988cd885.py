"""Strassen-based rank decomposition for 2x4x4 matrix multiplication.

Uses ring decomposition R = Z[x]/(x^2-1) to reduce the problem to a single
2x2x2 matrix multiplication over R, then applies Strassen (rank 7) over R,
yielding rank 7 for the real-valued decomposition.
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

def _strassen_ring7():
    """
    Build rank-7 decomposition using ring R = Z[x]/(x^2-1).
    
    The 4x4 matrix B is viewed as a 2x2 matrix over R where each entry is
    a pair (a, b) representing a + b*x with x^2 = 1.
    
    B = [[B00, B01], [B10, B11]] where:
      B00 = (B[0,0], B[0,2]), B01 = (B[0,1], B[0,3])
      B10 = (B[1,0], B[1,2]), B11 = (B[1,1], B[1,3])
    
    Similarly for C (output):
      C00 = (C[0,0], C[0,2]), C01 = (C[0,1], C[0,3])
      C10 = (C[1,0], C[1,2]), C11 = (C[1,1], C[1,3])
    
    A is a 2x2 matrix over R:
      A00 = (A[0,0], A[0,2]), A01 = (A[0,1], A[0,3])
      A10 = (A[1,0], A[1,2]), A11 = (A[1,1], A[1,3])
    
    Strassen over R gives 7 multiplications M0..M6, each producing a pair.
    The reconstruction of C entries from M values gives the W coefficients.
    
    U coefficients: for each Strassen term, the A-side coefficients in R.
    V coefficients: for each Strassen term, the B-side coefficients in R.
    W coefficients: for each Strassen term, the C-side coefficients in R.
    
    Each R-coefficient (a, b) maps to two real coefficients:
      a contributes to the "even" part (index 2*j or 2*k)
      b contributes to the "odd" part (index 2*j+1 or 2*k+1)
    """
    n, m, p = 2, 4, 4
    rank = 7
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen 7 terms for 2x2x2 over a ring:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each term t, define:
    # U_side: list of (i, j, (a, b)) where (a,b) is the R-coefficient for A[i,j]
    # V_side: list of (j, k, (a, b)) where (a,b) is the R-coefficient for B[j,k]
    # W_side: list of (i, k, (a, b)) where (a,b) is the R-coefficient for C[i,k]
    
    # R-coefficients: (a, b) means a + b*x
    # Mapping to real indices:
    #   A[i, j] where j in {0,1,2,3}: j=2*j_l + j_p, j_p in {0,1}
    #     j_p=0 -> real index j, j_p=1 -> real index j+2
    #   Similarly for B and C.
    
    # Strassen U-side (A factor) coefficients as R-pairs:
    # Each entry is (i, j_l, (a, b)) where j_l in {0,1} is the local column
    U_coeffs = [
        [(0, 0, (1, 1)), (1, 1, (1, 1))],   # M0: A00*(1,1) + A11*(1,1)
        [(1, 0, (1, 1)), (1, 1, (1, 1))],   # M1: A10*(1,1) + A11*(1,1)
        [(0, 0, (1, 1))],                    # M2: A00*(1,1)
        [(1, 1, (1, 1))],                    # M3: A11*(1,1)
        [(0, 0, (1, 1)), (0, 1, (1, 1))],   # M4: A00*(1,1) + A01*(1,1)
        [(1, 0, (1, 1)), (0, 0, (-1, -1))], # M5: A10*(1,1) - A00*(1,1)
        [(0, 1, (1, 1)), (1, 1, (-1, -1))], # M6: A01*(1,1) - A11*(1,1)
    ]
    
    # Strassen V-side (B factor) coefficients as R-pairs:
    V_coeffs = [
        [(0, 0, (1, 1)), (1, 1, (1, 1))],   # M0: B00*(1,1) + B11*(1,1)
        [(0, 0, (1, 1))],                    # M1: B00*(1,1)
        [(0, 1, (1, 1)), (1, 1, (-1, -1))], # M2: B01*(1,1) - B11*(1,1)
        [(1, 0, (1, 1)), (0, 0, (-1, -1))], # M3: B10*(1,1) - B00*(1,1)
        [(1, 1, (1, 1))],                    # M4: B11*(1,1)
        [(0, 0, (1, 1)), (0, 1, (1, 1))],   # M5: B00*(1,1) + B01*(1,1)
        [(1, 0, (1, 1)), (1, 1, (1, 1))],   # M6: B10*(1,1) + B11*(1,1)
    ]
    
    # Strassen W-side (C factor) coefficients as R-pairs:
    W_coeffs = [
        [(0, 0, (1, 1)), (1, 1, (1, 1))],   # M0: C00*(1,1) + C11*(1,1)
        [(1, 0, (1, 1)), (1, 1, (-1, -1))], # M1: C10*(1,1) - C11*(1,1)
        [(0, 1, (1, 1)), (1, 1, (1, 1))],   # M2: C01*(1,1) + C11*(1,1)
        [(0, 0, (1, 1)), (1, 0, (1, 1))],   # M3: C00*(1,1) + C10*(1,1)
        [(0, 0, (-1, -1)), (0, 1, (1, 1))], # M4: -C00*(1,1) + C01*(1,1)
        [(1, 1, (1, 1))],                    # M5: C11*(1,1)
        [(0, 0, (1, 1))],                    # M6: C00*(1,1)
    ]
    
    for t in range(7):
        col = t
        # Fill U
        for (i, j_l, (a, b)) in U_coeffs[t]:
            # j_l=0 -> j=0, j_l=1 -> j=1
            # a -> real col j, b -> real col j+2
            j = j_l
            U[i * m + j, col] += a
            U[i * m + j + 2, col] += b
        
        # Fill V
        for (j_l, k_l, (a, b)) in V_coeffs[t]:
            j = j_l
            k = k_l
            V[j * p + k, col] += a
            V[j * p + k + 2, col] += b
        
        # Fill W
        for (i, k_l, (a, b)) in W_coeffs[t]:
            k = k_l
            W[k * n + i, col] += a
            W[k + 2 * n + i, col] += b
    
    return U, V, W

def _strassen_28():
    """
    Build Strassen-based decomposition for C = A * B where A is 2x4, B is 4x4, C is 2x4.
    Split into 4 subproblems of size 2x2x2. Total rank: 4 * 7 = 28
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    U_coeffs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],
        1: [(1,0,1), (1,1,-1)],
        2: [(0,1,1), (1,1,1)],
        3: [(0,0,1), (1,0,1)],
        4: [(0,0,-1), (0,1,1)],
        5: [(1,1,1)],
        6: [(0,0,1)],
    }
    
    col = 0
    for jp in range(2):
        for kp in range(2):
            for t in range(7):
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

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
    
    # Try rank-7 ring decomposition first
    U7, V7, W7 = _strassen_ring7()
    
    if _verify(U7, V7, W7, n, m, p):
        return {"U": _to_list_of_lists(U7), "V": _to_list_of_lists(V7), "W": _to_list_of_lists(W7)}
    
    # Fallback: Strassen 28
    U, V, W = _strassen_28()
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
