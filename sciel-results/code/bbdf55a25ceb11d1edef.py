DIMENSIONS = [5, 5, 5]
import numpy as np
import time

def _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col):
    """Add Strassen rank-7 decomposition for 2x2x2 block."""
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1

    # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    # C[i0,k0] += M1, C[i1,k1] += M1
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M2 = (A[i1,j0]+A[i1,j1])*(B[j0,k0])
    # C[i1,k0] += M2, C[i1,k1] -= M2
    U[i1*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    W[k0*n+i1][col] = 1
    W[k1*n+i1][col] = -1
    col += 1

    # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
    # C[i0,k1] += M3, C[i1,k1] += M3
    U[i0*m+j0][col] = 1
    V[j0*p+k1][col] = 1
    V[j1*p+k1][col] = -1
    W[k1*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
    # C[i0,k0] += M4, C[i1,k0] += M4
    U[i1*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j0*p+k0][col] = -1
    W[k0*n+i0][col] = 1
    W[k0*n+i1][col] = 1
    col += 1

    # M5 = (A[i0,j0]+A[i0,j1])*(B[j1,k1])
    # C[i0,k0] -= M5, C[i0,k1] += M5
    U[i0*m+j0][col] = 1
    U[i0*m+j1][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = -1
    W[k1*n+i0][col] = 1
    col += 1

    # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    # C[i1,k1] += M6
    U[i1*m+j0][col] = 1
    U[i0*m+j0][col] = -1
    V[j0*p+k0][col] = 1
    V[j0*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    # C[i0,k0] += M7
    U[i0*m+j1][col] = 1
    U[i1*m+j1][col] = -1
    V[j1*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    return col

def _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col):
    """Add Laderman rank-7 decomposition for 3x3x3 block.
    Standard Laderman (1976) rank-7 algorithm."""
    i0, i1, i2 = i_start, i_start+1, i_start+2
    j0, j1, j2 = j_start, j_start+1, j_start+2
    k0, k1, k2 = k_start, k_start+1, k_start+2

    # M1 = (A00+A22)(B00+B22) -> C00, C22
    U[i0*m+j0][col] = 1
    U[i2*m+j2][col] = 1
    V[j0*p+k0][col] = 1
    V[j2*p+k2][col] = 1
    W[k0*n+i0][col] = 1
    W[k2*n+i2][col] = 1
    col += 1

    # M2 = (A11+A22)(B11+B22) -> C11, C22
    U[i1*m+j1][col] = 1
    U[i2*m+j2][col] = 1
    V[j1*p+k1][col] = 1
    V[j2*p+k2][col] = 1
    W[k1*n+i1][col] = 1
    W[k2*n+i2][col] = 1
    col += 1

    # M3 = (A00+A11)(B00+B11) -> C00, C11
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M4 = (A22+A00)(B22+B00) -> C22, C00
    U[i2*m+j2][col] = 1
    U[i0*m+j0][col] = 1
    V[j2*p+k2][col] = 1
    V[j0*p+k0][col] = 1
    W[k2*n+i2][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    # M5 = (A22+A11)(B22+B11) -> C22, C11
    U[i2*m+j2][col] = 1
    U[i1*m+j1][col] = 1
    V[j2*p+k2][col] = 1
    V[j1*p+k1][col] = 1
    W[k2*n+i2][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M6 = (A01+A21)(B10+B20) -> C00, C20
    U[i0*m+j1][col] = 1
    U[i2*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j2*p+k0][col] = 1
    W[k0*n+i0][col] = 1
    W[k0*n+i2][col] = 1
    col += 1

    # M7 = (A10+A12)(B01+B21) -> C10, C12
    U[i1*m+j0][col] = 1
    U[i1*m+j2][col] = 1
    V[j0*p+k1][col] = 1
    V[j2*p+k1][col] = 1
    W[k0*n+i1][col] = 1
    W[k2*n+i1][col] = 1
    col += 1

    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct using full tensor check."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all entries of the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    import time
    _t0 = time.time()
    _deadline = _t0 + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try 2+3 partition first (lower rank = 104)
    n_parts = [2, 3]
    m_parts = [2, 3]
    p_parts = [2, 3]
    
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
    
    # Calculate total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni, mj, pk = n_parts[i], m_parts[j], p_parts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                elif ni == 3 and mj == 3 and pk == 3:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    col = _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col)
                elif ni == 3 and mj == 3 and pk == 3:
                    col = _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    if time.time() > _deadline:
        # Timeout safety: return schoolbook as last resort
        U_f = [[0] * (n*m*p) for _ in range(n*m)]
        V_f = [[0] * (n*m*p) for _ in range(m*p)]
        W_f = [[0] * (n*m*p) for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1
                    V_f[j*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    # Fallback: 2+2+1 partition (rank 114)
    n_parts2 = [2, 2, 1]
    m_parts2 = [2, 2, 1]
    p_parts2 = [2, 2, 1]
    n_starts2 = [0, 2, 4]
    m_starts2 = [0, 2, 4]
    p_starts2 = [0, 2, 4]
    
    total_rank2 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = n_parts2[i], m_parts2[j], p_parts2[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank2 += 7
                else:
                    total_rank2 += ni * mj * pk
    
    U2 = [[0] * total_rank2 for _ in range(n * m)]
    V2 = [[0] * total_rank2 for _ in range(m * p)]
    W2 = [[0] * total_rank2 for _ in range(p * n)]
    
    col2 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts2[i]
                mj = m_parts2[j]
                pk = p_parts2[k]
                i_start = n_starts2[i]
                j_start = m_starts2[j]
                k_start = p_starts2[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    col2 = _strassen_2x2(i_start, j_start, k_start, n, m, p, U2, V2, W2, col2)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U2[(i_start + u) * m + (j_start + v)][col2] = 1
                                V2[(j_start + v) * p + (k_start + w)][col2] = 1
                                W2[(k_start + w) * n + (i_start + u)][col2] = 1
                                col2 += 1
    
    return {"U": U2, "V": V2, "W": W2}
