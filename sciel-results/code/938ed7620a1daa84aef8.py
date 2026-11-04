"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Strassen decomposition of 2x2x2, rank 7. Returns (U, V, W) as lists."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A11+A22)(B21+B22)
    U[0][0] = 1; U[3][0] = 1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, rank

def _block_decomposition():
    """
    Decompose 3x5x6 using blocks:
    i: 3 = 1 + 2
    j: 5 = 2 + 3
    k: 6 = 2 + 2 + 2
    Use Strassen for 2x2x2 blocks, schoolbook otherwise.
    """
    n, m, p = 3, 5, 6
    i_sizes = [1, 2]
    j_sizes = [2, 3]
    k_sizes = [2, 2, 2]
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    col = 0
    
    i_off = 0
    for i_size in i_sizes:
        j_off = 0
        for j_size in j_sizes:
            k_off = 0
            for k_size in k_sizes:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    Ub, Vb, Wb, r = _strassen_222()
                else:
                    # Schoolbook block
                    r = i_size * j_size * k_size
                    Ub = [[0]*r for _ in range(i_size*j_size)]
                    Vb = [[0]*r for _ in range(j_size*k_size)]
                    Wb = [[0]*r for _ in range(k_size*i_size)]
                    c = 0
                    for ii in range(i_size):
                        for jj in range(j_size):
                            for kk in range(k_size):
                                Ub[ii*j_size+jj][c] = 1
                                Vb[jj*k_size+kk][c] = 1
                                Wb[kk*i_size+ii][c] = 1
                                c += 1
                
                for rr in range(r):
                    for u_row in range(i_size * j_size):
                        il = u_row // j_size
                        jl = u_row % j_size
                        ig = i_off + il
                        jg = j_off + jl
                        val = Ub[u_row][rr]
                        if val != 0:
                            key = (ig * m + jg, col + rr)
                            U_dict[key] = U_dict.get(key, 0) + val
                    for v_row in range(j_size * k_size):
                        jl = v_row // k_size
                        kl = v_row % k_size
                        jg = j_off + jl
                        kg = k_off + kl
                        val = Vb[v_row][rr]
                        if val != 0:
                            key = (jg * p + kg, col + rr)
                            V_dict[key] = V_dict.get(key, 0) + val
                    for w_row in range(k_size * i_size):
                        kl = w_row // i_size
                        il = w_row % i_size
                        kg = k_off + kl
                        ig = i_off + il
                        val = Wb[w_row][rr]
                        if val != 0:
                            key = (kg * n + ig, col + rr)
                            W_dict[key] = W_dict.get(key, 0) + val
                col += r
                k_off += k_size
            j_off += j_size
        i_off += i_size
    
    total_rank = col
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    for (row, c), val in U_dict.items():
        U[row][c] = val
    for (row, c), val in V_dict.items():
        V[row][c] = val
    for (row, c), val in W_dict.items():
        W[row][c] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Try block decomposition with Strassen (rank 87)
    try:
        result = _block_decomposition()
        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 90)
    return _schoolbook(n, m, p)
