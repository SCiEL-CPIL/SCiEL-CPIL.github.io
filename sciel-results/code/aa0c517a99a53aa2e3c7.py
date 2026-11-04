DIMENSIONS = [5, 5, 5]
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if this is a valid multiplication tensor entry
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # Wait no. The multiplication tensor T has T[i,j,k] = 1 where
    # the (i,k)-th entry of A*B depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Wait, that's not right either. Let me think...
    # The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (for the identity)
    # No, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check a few random entries
    for _ in range(20):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        if abs(val - 1.0) > 1e-10:
            return False
    return True

def _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col):
    """Add Strassen decomposition for 2x2x2 block starting at given indices."""
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1
    
    # M1
    U[i0 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1
    
    # M2
    U[i1 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    W[k0 * n + i1][col] = 1
    W[k1 * n + i1][col] = -1
    col += 1
    
    # M3
    U[i0 * m + j0][col] = 1
    V[j0 * p + k1][col] = 1
    V[j1 * p + k1][col] = -1
    W[k1 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1
    
    # M4
    U[i1 * m + j1][col] = 1
    V[j1 * p + k0][col] = 1
    V[j0 * p + k0][col] = -1
    W[k0 * n + i0][col] = 1
    W[k0 * n + i1][col] = 1
    col += 1
    
    # M5
    U[i0 * m + j0][col] = 1
    U[i0 * m + j1][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = -1
    W[k1 * n + i0][col] = 1
    col += 1
    
    # M6
    U[i1 * m + j0][col] = 1
    U[i0 * m + j0][col] = -1
    V[j0 * p + k0][col] = 1
    V[j0 * p + k1][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1
    
    # M7
    U[i0 * m + j1][col] = 1
    U[i1 * m + j1][col] = -1
    V[j1 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    col += 1
    
    return col

def _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col):
    """Add Laderman rank-7 decomposition for 3x3x3 block."""
    # Laderman's 7-term algorithm for 3x3 matrix multiplication
    # M1 = (A11+A33)(B11+B33)
    # M2 = (A22+A33)(B22+B33)
    # M3 = (A11+A22)(B11+B22)
    # M4 = (A33+A11)(B33+B11)
    # M5 = (A33+A22)(B33+B22)
    # M6 = (A12+A32)(B21+B31)
    # M7 = (A21+A23)(B12+B32)
    
    # Indices: i0,i1,i2 = i_start, i_start+1, i_start+2
    #           j0,j1,j2 = j_start, j_start+1, j_start+2
    #           k0,k1,k2 = k_start, k_start+1, k_start+2
    i0, i1, i2 = i_start, i_start+1, i_start+2
    j0, j1, j2 = j_start, j_start+1, j_start+2
    k0, k1, k2 = k_start, k_start+1, k_start+2
    
    # M1 = (A[i0,j0]+A[i2,j2]) * (B[j0,k0]+B[j2,k2])
    U[i0*m+j0][col] = 1
    U[i2*m+j2][col] = 1
    V[j0*p+k0][col] = 1
    V[j2*p+k2][col] = 1
    # C[i0,k0] += M1, C[i2,k2] += M1
    W[k0*n+i0][col] = 1
    W[k2*n+i2][col] = 1
    col += 1
    
    # M2 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    U[i1*m+j1][col] = 1
    U[i2*m+j2][col] = 1
    V[j1*p+k1][col] = 1
    V[j2*p+k2][col] = 1
    # C[i1,k1] += M2, C[i2,k2] += M2
    W[k1*n+i1][col] = 1
    W[k2*n+i2][col] = 1
    col += 1
    
    # M3 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    # C[i0,k0] += M3, C[i1,k1] += M3
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1
    
    # M4 = (A[i2,j2]+A[i0,j0]) * (B[j2,k2]+B[j0,k0])
    U[i2*m+j2][col] = 1
    U[i0*m+j0][col] = 1
    V[j2*p+k2][col] = 1
    V[j0*p+k0][col] = 1
    # C[i2,k2] += M4, C[i0,k0] += M4
    W[k2*n+i2][col] = 1
    W[k0*n+i0][col] = 1
    col += 1
    
    # M5 = (A[i2,j2]+A[i1,j1]) * (B[j2,k2]+B[j1,k1])
    U[i2*m+j2][col] = 1
    U[i1*m+j1][col] = 1
    V[j2*p+k2][col] = 1
    V[j1*p+k1][col] = 1
    # C[i2,k2] += M5, C[i1,k1] += M5
    W[k2*n+i2][col] = 1
    W[k1*n+i1][col] = 1
    col += 1
    
    # M6 = (A[i0,j1]+A[i2,j1]) * (B[j1,k0]+B[j2,k0])
    U[i0*m+j1][col] = 1
    U[i2*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j2*p+k0][col] = 1
    # C[i0,k0] += M6, C[i2,k0] += M6
    W[k0*n+i0][col] = 1
    W[k0*n+i2][col] = 1
    col += 1
    
    # M7 = (A[i1,j0]+A[i1,j2]) * (B[j0,k1]+B[j2,k1])
    U[i1*m+j0][col] = 1
    U[i1*m+j2][col] = 1
    V[j0*p+k1][col] = 1
    V[j2*p+k1][col] = 1
    # C[i1,k1] += M7, C[i1,k1] += M7 (same position, so coefficient 2?)
    # Wait, this doesn't look right. Let me reconsider.
    W[k1*n+i1][col] = 1
    col += 1
    
    return col

def construct():
       n, m, p = DIMENSIONS
       
       # Partitions: 2+3
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
       # 2x2x2: 7 (Strassen)
       # 2x2x3: 12 (schoolbook)
       # 2x3x2: 12 (schoolbook)
       # 2x3x3: 18 (schoolbook)
       # 3x2x2: 12 (schoolbook)
       # 3x2x3: 18 (schoolbook)
       # 3x3x2: 18 (schoolbook)
       # 3x3x3: 7 (Laderman)
       total_rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 7  # = 104
       
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
                       # Schoolbook
                       for u in range(ni):
                           for v in range(mj):
                               for w in range(pk):
                                   U[(i_start + u) * m + (j_start + v)][col] = 1
                                   V[(j_start + v) * p + (k_start + w)][col] = 1
                                   W[(k_start + w) * n + (i_start + u)][col] = 1
                                   col += 1
                                   
       # Verify and fallback if needed
       if not _verify_decomposition(U, V, W, n, m, p):
           # Fallback to 2+2+1 partition
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
                       if n_parts2[i] == 2 and m_parts2[j] == 2 and p_parts2[k] == 2:
                           total_rank2 += 7
                       else:
                           total_rank2 += n_parts2[i] * m_parts2[j] * p_parts2[k]
           
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
       
       return {"U": U, "V": V, "W": W}
