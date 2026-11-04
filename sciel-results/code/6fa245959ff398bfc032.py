"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """Block decomposition with Strassen on 2x2x2 sub-blocks for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Split dimensions: rows 2+1, middle 2+2+1, cols 2+2+2+1
    row_splits = [2, 1]
    mid_splits = [2, 2, 1]
    col_splits = [2, 2, 2, 1]
    
    row_starts = [0]
    for s in row_splits:
        row_starts.append(row_starts[-1] + s)
    mid_starts = [0]
    for s in mid_splits:
        mid_starts.append(mid_starts[-1] + s)
    col_starts = [0]
    for s in col_splits:
        col_starts.append(col_starts[-1] + s)
    
    # Collect all terms
    terms = []  # list of (U_row, V_col, W_row, coefficient)
    
    for ri, (r_start, r_size) in enumerate(zip(row_starts[:-1], row_splits)):
        for mi, (m_start, m_size) in enumerate(zip(mid_starts[:-1], mid_splits)):
            for pi, (p_start, p_size) in enumerate(zip(col_starts[:-1], col_splits)):
                if r_size == 2 and m_size == 2 and p_size == 2:
                    # Strassen decomposition for 2x2x2
                    # A[2x2] * B[2x2] = C[2x2]
                    # Strassen:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    # C11 = M1+M4-M5+M7
                    # C12 = M3+M5
                    # C21 = M2+M4
                    # C22 = M1-M2+M3+M6
                    
                    # Each Strassen term contributes to specific (i,j,k) positions
                    # i is in {r_start, r_start+1}, j in {m_start, m_start+1}, k in {p_start, p_start+1}
                    a00, a01, a10, a11 = (r_start, m_start), (r_start, m_start+1), (r_start+1, m_start), (r_start+1, m_start+1)
                    b00, b01, b10, b11 = (m_start, p_start), (m_start, p_start+1), (m_start+1, p_start), (m_start+1, p_start+1)
                    c00, c01, c10, c11 = (r_start, p_start), (r_start, p_start+1), (r_start+1, p_start), (r_start+1, p_start+1)
                    
                    # M1 = (A00+A11)(B00+B11)
                    terms.append((a00, b00, c00, 1))
                    terms.append((a00, b11, c00, 1))
                    terms.append((a11, b00, c00, 1))
                    terms.append((a11, b11, c00, 1))
                    terms.append((a00, b00, c01, 1))
                    terms.append((a00, b11, c01, 1))
                    terms.append((a11, b00, c01, 1))
                    terms.append((a11, b11, c01, 1))
                    terms.append((a00, b00, c10, 1))
                    terms.append((a00, b11, c10, 1))
                    terms.append((a11, b00, c10, 1))
                    terms.append((a11, b11, c10, 1))
                    terms.append((a00, b00, c11, 1))
                    terms.append((a00, b11, c11, 1))
                    terms.append((a11, b00, c11, 1))
                    terms.append((a11, b11, c11, 1))
                    
                    # M2 = (A10+A11)B00
                    terms.append((a10, b00, c10, 1))
                    terms.append((a11, b00, c10, 1))
                    terms.append((a10, b00, c11, -1))
                    terms.append((a11, b00, c11, -1))
                    
                    # M3 = A00(B10-B11)
                    terms.append((a00, b10, c01, 1))
                    terms.append((a00, b11, c01, -1))
                    terms.append((a00, b10, c11, 1))
                    terms.append((a00, b11, c11, -1))
                    
                    # M4 = A11(B01-B00)
                    terms.append((a11, b01, c00, 1))
                    terms.append((a11, b00, c00, -1))
                    terms.append((a11, b01, c10, 1))
                    terms.append((a11, b00, c10, 1))
                    terms.append((a11, b01, c11, -1))
                    terms.append((a11, b00, c11, -1))
                    
                    # M5 = (A00+A01)B11
                    terms.append((a00, b11, c01, 1))
                    terms.append((a01, b11, c01, 1))
                    terms.append((a00, b11, c11, -1))
                    terms.append((a01, b11, c11, -1))
                    
                    # M6 = (A10-A00)(B00+B01)
                    terms.append((a10, b00, c11, 1))
                    terms.append((a10, b01, c11, 1))
                    terms.append((a00, b00, c11, -1))
                    terms.append((a00, b01, c11, -1))
                    
                    # M7 = (A01-A11)(B10+B11)
                    terms.append((a01, b10, c00, 1))
                    terms.append((a01, b11, c00, 1))
                    terms.append((a11, b10, c00, -1))
                    terms.append((a11, b11, c00, -1))
                    terms.append((a01, b10, c01, 1))
                    terms.append((a01, b11, c01, 1))
                    terms.append((a11, b10, c01, -1))
                    terms.append((a11, b11, c01, -1))
                    
                else:
                    # Schoolbook for this block
                    for i in range(r_start, r_start + r_size):
                        for j in range(m_start, m_start + m_size):
                            for k in range(p_start, p_start + p_size):
                                terms.append(((i, j), (j, k), (k, i), 1))
    
    # Now combine terms: group by (U_index, V_index, W_index) and sum coefficients
    # U_index = i*m + j, V_index = j*p + k, W_index = k*n + i
    from collections import defaultdict
    combined = defaultdict(int)
    for (u_idx, v_idx, w_idx, coeff) in terms:
        u_row = u_idx[0] * m + u_idx[1]
        v_row = v_idx[0] * p + v_idx[1]
        w_row = w_idx[0] * n + w_idx[1]
        combined[(u_row, v_row, w_row)] += coeff
    
    # Remove zero coefficients
    combined = {k: v for k, v in combined.items() if v != 0}
    
    rank = len(combined)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, ((u_row, v_row, w_row), coeff) in enumerate(combined.items()):
        U[u_row][col] = coeff
        V[v_row][col] = coeff
        W[w_row][col] = coeff
    
    # Verify
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if i,j,k valid, 0 otherwise
        # Actually the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix mult: T[i,j,k] = 1 if all indices valid, else 0
        # Wait, no. The tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all valid i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k]
        
        # Let's verify by checking that the decomposition is exact
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
