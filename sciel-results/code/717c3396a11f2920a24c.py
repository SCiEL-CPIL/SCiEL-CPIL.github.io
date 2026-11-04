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

def _verify_strassen_222():
    """Verify the Strassen 2x2x2 decomposition is correct."""
    U, V, W, rank = _strassen_222()
    n, m, p = 2, 2, 2
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _block_decomposition(i_sizes=None, j_sizes=None, k_sizes=None):
    """
    Decompose 3x5x6 using blocks with Strassen for 2x2x2 blocks.
    """
    n, m, p = 3, 5, 6
    if i_sizes is None:
        i_sizes = [1, 2]
    if j_sizes is None:
        j_sizes = [2, 3]
    if k_sizes is None:
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
    
    # Quick self-test of Strassen 2x2x2
    strassen_ok = _verify_strassen_222()
    
    # Try various block decompositions with Strassen for 2x2x2 blocks
    # and pick the one with lowest rank that verifies
    
    # Generate all valid partitions
    def _partitions(total, num_parts):
        """Generate all compositions of total into num_parts positive parts."""
        if num_parts == 1:
            if total >= 1:
                yield [total]
            return
        for first in range(1, total - num_parts + 2):
            for rest in _partitions(total - first, num_parts - 1):
                yield [first] + rest
    
    i_parts = list(_partitions(3, 2))  # 3 into 2 parts: [1,2],[2,1]
    j_parts = list(_partitions(5, 2))  # 5 into 2 parts: [1,4],[2,3],[3,2],[4,1]
    k_parts = list(_partitions(6, 3))  # 6 into 3 parts: various
    
    best_result = None
    best_rank = float('inf')
    
    # Sort candidates by expected rank (fewer 2x2x2 blocks = higher rank, but we want low)
    # Actually, more 2x2x2 blocks = lower rank. Prioritize partitions with more 2s.
    
    all_candidates = []
    for i_s in i_parts:
        for j_s in j_parts:
            for k_s in k_parts:
                # Count how many 2x2x2 blocks
                count_222 = 0
                for a, b, c in zip(i_s, j_s, k_s):
                    if a == 2 and b == 2 and c == 2:
                        count_222 += 1
                # Expected rank
                expected_rank = 0
                i_off = 0
                for i_size in i_s:
                    j_off = 0
                    for j_size in j_s:
                        k_off = 0
                        for k_size in k_s:
                            if i_size == 2 and j_size == 2 and k_size == 2 and strassen_ok:
                                expected_rank += 7
                            else:
                                expected_rank += i_size * j_size * k_size
                        k_off += k_size
                    j_off += j_size
                all_candidates.append((expected_rank, i_s, j_s, k_s))
    
    # Sort by expected rank
    all_candidates.sort(key=lambda x: x[0])
    
    for expected_rank, i_s, j_s, k_s in all_candidates:
        if _remaining() < 2.0:
            break
        if expected_rank >= best_rank:
            break
        try:
            result = _block_decomposition(i_s, j_s, k_s)
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook (rank 90)
    return _schoolbook(n, m, p)
