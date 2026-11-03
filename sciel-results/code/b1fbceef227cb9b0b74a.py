"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses block-Strassen: split into 2x2 blocks, use Strassen rank-7 per 2x2 multiplication.
    A is 2x2 blocks of 2x2, B is 2x3 blocks of 2x2.
    Each output block C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c]
    Using Strassen: 7 per 2x2*2x2, so 14 per output block, 6 blocks = 84 total.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    bn = 2
    n_blocks = n // bn  # 2
    m_blocks = m // bn  # 2
    p_blocks = p // bn  # 3
    
    # Strassen for 2x2*2x2:
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11)
    # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    strassen_terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):-1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    max_rank = n_blocks * p_blocks * m_blocks * 7 + 10
    U_data = np.zeros((n*m, max_rank))
    V_data = np.zeros((m*p, max_rank))
    W_data = np.zeros((p*n, max_rank))
    
    col = 0
    for a in range(n_blocks):
        for c in range(p_blocks):
            for jb in range(m_blocks):
                for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                    for (ar, ac), val in a_coeffs.items():
                        U_data[(2*a+ar)*m + (2*jb+ac), col] += val
                    for (br, bc), val in b_coeffs.items():
                        V_data[(2*jb+br)*p + (2*c+bc), col] += val
                    for (cr, cc), val in c_coeffs.items():
                        W_data[(2*c+cc)*n + (2*a+cr), col] += val
                    col += 1
    
    rank = col
    U = U_data[:, :rank].tolist()
    V = V_data[:, :rank].tolist()
    W = W_data[:, :rank].tolist()
    
    # Verify
    U_np, V_np, W_np = np.array(U), np.array(V), np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j] * V_np[j*p+k], W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    r2 = n*m*p
                    U2 = np.zeros((n*m, r2)); V2 = np.zeros((m*p, r2)); W2 = np.zeros((p*n, r2))
                    cc = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U2[ii*m+jj, cc] = 1; V2[jj*p+kk, cc] = 1; W2[kk*n+ii, cc] = 1; cc += 1
                    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
