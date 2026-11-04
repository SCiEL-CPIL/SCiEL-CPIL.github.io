"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

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

def _schoolbook_block(n, m, p):
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
    return U, V, W, rank

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

def _get_block_decomp(n, m, p):
    """Get the best known decomposition for a block of size n x m x p."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    else:
        return _schoolbook_block(n, m, p)

def _block_decomposition(n, m, p, i_parts, j_parts, k_parts):
    """
    Decompose n x m x p using blocks defined by partitions.
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    col = 0
    
    i_off = 0
    for i_size in i_parts:
        j_off = 0
        for j_size in j_parts:
            k_off = 0
            for k_size in k_parts:
                Ub, Vb, Wb, r = _get_block_decomp(i_size, j_size, k_size)
                
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

def _find_best_partition(n, m, p):
    """Try a few partitions to find the one with the lowest rank."""
    # We want to maximize the number of 2x2x2 blocks.
    # Partitions of n, m, p into 1s and 2s.
    
    def gen_parts(total, max_size=2):
        parts = []
        if total == 0:
            parts.append([])
        elif total == 1:
            parts.append([1])
        elif total == 2:
            parts.append([2])
            parts.append([1, 1])
        elif total > 2:
            # Recursively generate
            if total % 2 == 0:
                # Try [2] + parts(total-2)
                for p in gen_parts(total - 2, max_size):
                    parts.append([2] + p)
            if total >= 2:
                # Try [1] + parts(total-1)
                for p in gen_parts(total - 1, max_size):
                    parts.append([1] + p)
        return parts

    best_result = None
    best_rank = float('inf')
    
    # To keep runtime low, we only try a subset of partitions
    # Specifically, we prioritize partitions with more 2s.
    
    i_parts_list = gen_parts(n)
    j_parts_list = gen_parts(m)
    k_parts_list = gen_parts(p)
    
    # Sort partitions by number of 2s (descending)
    def count_twos(parts):
        return parts.count(2)
        
    i_parts_list.sort(key=count_twos, reverse=True)
    j_parts_list.sort(key=count_twos, reverse=True)
    k_parts_list.sort(key=count_twos, reverse=True)
    
    # Try top partitions
    for i_p in i_parts_list[:5]:
        for j_p in j_parts_list[:5]:
            for k_p in k_parts_list[:5]:
                if _remaining() < 10:
                    return best_result
                try:
                    result = _block_decomposition(n, m, p, i_p, j_p, k_p)
                    rank = len(result["U"][0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
                        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
                            # We can stop early if we find a very good one, 
                            # but let's check a few more to be sure
                            pass
                except Exception:
                    pass
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank block decomposition
    result = _find_best_partition(n, m, p)
    
    if result and _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U, "V": V, "W": W}
