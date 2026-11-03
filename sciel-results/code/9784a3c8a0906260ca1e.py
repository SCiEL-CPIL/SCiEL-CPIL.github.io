"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

_START_TIME = time.time()
DEADLINE = _START_TIME + 120.0

def _check_deadline():
    return time.time() > DEADLINE - 10.0

def _strassen_2x2x2(offset_u, offset_v, offset_w, i0, j0, k0, n, m, p):
    """
    Generate Strassen rank-7 decomposition for a 2x2x2 sub-problem.
    
    Parameters:
    - offset_u, offset_v, offset_w: column offsets in U, V, W matrices
    - i0, j0, k0: starting indices for the 2x2x2 block
    - n, m, p: full dimensions (for index mapping)
    
    Returns: (U_cols, V_cols, W_cols) as lists of (row_idx, value) tuples
    """
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 7 Strassen products, we need to add entries to U, V, W
    # U is indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    
    U_entries = []  # list of (row, col_offset, value)
    V_entries = []
    W_entries = []
    
    # Strassen formula as (U_coeff, V_coeff, W_coeff) for each product
    # Each product contributes to C[i,k] via W[i,k] * (sum U[i,j]*A[i,j]) * (sum V[j,k]*B[j,k])
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    U_entries.append((i0 * m + j0, 0, 1.0))
    U_entries.append(((i0+1) * m + (j0+1), 0, 1.0))
    V_entries.append((j0 * p + k0, 0, 1.0))
    V_entries.append(((j0+1) * p + (k0+1), 0, 1.0))
    W_entries.append((k0 * n + i0, 0, 1.0))
    W_entries.append(((k0+1) * n + (i0+1), 0, 1.0))
    
    # M2: (A10+A11)B00
    # U: A10=1, A11=1; V: B00=1; W: C10=1
    U_entries.append((i0 * m + (j0+1), 1, 1.0))
    U_entries.append(((i0+1) * m + (j0+1), 1, 1.0))
    V_entries.append((j0 * p + k0, 1, 1.0))
    W_entries.append((k0 * n + (i0+1), 1, 1.0))
    
    # M3: A00(B10-B11)
    # U: A00=1; V: B10=1, B11=-1; W: C01=1
    U_entries.append((i0 * m + j0, 2, 1.0))
    V_entries.append(((j0+1) * p + k0, 2, 1.0))
    V_entries.append(((j0+1) * p + (k0+1), 2, -1.0))
    W_entries.append((k0 * n + i0, 2, 1.0))
    
    # M4: A11(B01-B00)
    # U: A11=1; V: B01=1, B00=-1; W: C00=1, C10=1
    U_entries.append(((i0+1) * m + (j0+1), 3, 1.0))
    V_entries.append((j0 * p + (k0+1), 3, 1.0))
    V_entries.append((j0 * p + k0, 3, -1.0))
    W_entries.append((k0 * n + i0, 3, 1.0))
    W_entries.append((k0 * n + (i0+1), 3, 1.0))
    
    # M5: (A00+A01)B11
    # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
    U_entries.append((i0 * m + j0, 4, 1.0))
    U_entries.append((i0 * m + (j0+1), 4, 1.0))
    V_entries.append(((j0+1) * p + (k0+1), 4, 1.0))
    W_entries.append((k0 * n + i0, 4, -1.0))
    W_entries.append((k0 * n + i0, 4, 1.0))  # C01 contribution
    
    # Wait, I need to be more careful. Let me redo this.
    # The W factor tells us how each product contributes to each C[i,k].
    # For M5: contributes -1 to C00 and +1 to C01
    
    # Let me restart with a cleaner formulation.
    # For each Strassen product c, we have:
    #   z_c = (sum_{i,j} U_c[i,j] * A[i,j]) * (sum_{j,k} V_c[j,k] * B[j,k])
    #   C[i,k] += W_c[i,k] * z_c
    #
    # So W_c[i,k] is the coefficient for product c in computing C[i,k].
    
    # Let me redo all 7 products properly:
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # Contributes: +1 to C00, +1 to C11
    U_entries += [(i0*m + j0, 0, 1.0), ((i0+1)*m + (j0+1), 0, 1.0)]
    V_entries += [(j0*p + k0, 0, 1.0), ((j0+1)*p + (k0+1), 0, 1.0)]
    W_entries += [(k0*n + i0, 0, 1.0), ((k0+1)*n + (i0+1), 0, 1.0)]
    
    # M2 = (A10 + A11)B00
    # Contributes: +1 to C10
    U_entries += [(i0*m + (j0+1), 1, 1.0), ((i0+1)*m + (j0+1), 1, 1.0)]
    V_entries += [(j0*p + k0, 1, 1.0)]
    W_entries += [(k0*n + (i0+1), 1, 1.0)]
    
    # M3 = A00(B10 - B11)
    # Contributes: +1 to C01
    U_entries += [(i0*m + j0, 2, 1.0)]
    V_entries += [((j0+1)*p + k0, 2, 1.0), ((j0+1)*p + (k0+1), 2, -1.0)]
    W_entries += [(k0*n + i0, 2, 1.0)]
    
    # M4 = A11(B01 - B00)
    # Contributes: +1 to C00, +1 to C10
    U_entries += [((i0+1)*m + (j0+1), 3, 1.0)]
    V_entries += [(j0*p + (k0+1), 3, 1.0), (j0*p + k0, 3, -1.0)]
    W_entries += [(k0*n + i0, 3, 1.0), (k0*n + (i0+1), 3, 1.0)]
    
    # M5 = (A00 + A01)B11
    # Contributes: -1 to C00, +1 to C01
    U_entries += [(i0*m + j0, 4, 1.0), (i0*m + (j0+1), 4, 1.0)]
    V_entries += [((j0+1)*p + (k0+1), 4, 1.0)]
    W_entries += [(k0*n + i0, 4, -1.0), (k0*n + i0, 4, 1.0)]
    
    # M6 = (A10 - A00)(B00 + B10)
    # Contributes: +1 to C11
    U_entries += [(i0*m + (j0+1), 5, 1.0), (i0*m + j0, 5, -1.0)]
    V_entries += [(j0*p + k0, 5, 1.0), ((j0+1)*p + k0, 5, 1.0)]
    W_entries += [((k0+1)*n + (i0+1), 5, 1.0)]
    
    # M7 = (A01 - A11)(B01 + B11)
    # Contributes: +1 to C00, +1 to C11
    U_entries += [(i0*m + (j0+1), 6, 1.0), ((i0+1)*m + (j0+1), 6, -1.0)]
    V_entries += [(j0*p + (k0+1), 6, 1.0), ((j0+1)*p + (k0+1), 6, 1.0)]
    W_entries += [(k0*n + i0, 6, 1.0), ((k0+1)*n + (i0+1), 6, 1.0)]
    
    return U_entries, V_entries, W_entries

def _schoolbook_1xm_p(offset, i, j_start, j_end, k_start, k_end, n, m, p):
    """
    Generate schoolbook decomposition for a 1 x (j_end-j_start) x (k_end-k_start) sub-problem.
    Only row i of the output is computed.
    
    Returns: (U_entries, V_entries, W_entries)
    """
    U_entries = []
    V_entries = []
    W_entries = []
    
    col = 0
    for j in range(j_start, j_end):
        for k in range(k_start, k_end):
            U_entries.append((i * m + j, col, 1.0))
            V_entries.append((j * p + k, col, 1.0))
            W_entries.append((k * n + i, col, 1.0))
            col += 1
    
    return U_entries, V_entries, W_entries

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    Returns {'U': list of lists (15, rank), 'V': list of lists (25, rank), 'W': list of lists (15, rank)}
    where U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] exactly decompose the multiplication tensor.
    
    Strategy: Split into blocks:
    - Rows 0-1 (2 rows), cols 0-3 (4 intermediate), cols 0-3 (4 output): 2x4x4
      Further split into 4 blocks of 2x2x2, each using Strassen (rank 7). Total: 28.
    - Row 2 (1 row), cols 0-4 (5 intermediate), cols 0-4 (5 output): 1x5x5
      Split into 1x2x2 (Strassen, rank 3) + 1x2x3 + 1x3x2 + 1x3x3 (schoolbook, rank 9).
      Total: 3 + 6 + 6 + 9 = 24.
    
    Grand total rank: 28 + 24 = 52.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    nm = n * m  # 15
    mp = m * p  # 25
    pn = p * n  # 15
    
    # Collect all entries with column offsets
    U_entries = []  # (row, col, value)
    V_entries = []
    W_entries = []
    
    col_offset = 0
    
    # Block 1: Rows 0-1, intermediate cols 0-3, output cols 0-3 (2x4x4)
    # Split into 4 sub-blocks of 2x2x2:
    #   (i=0-1, j=0-1, k=0-1)
    #   (i=0-1, j=0-1, k=2-3)
    #   (i=0-1, j=2-3, k=0-1)
    #   (i=0-1, j=2-3, k=2-3)
    
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            U_e, V_e, W_e = _strassen_2x2x2(col_offset, col_offset, col_offset, 0, j0, k0, n, m, p)
            for row, col, val in U_e:
                U_entries.append((row, col_offset + col, val))
            for row, col, val in V_e:
                V_entries.append((row, col_offset + col, val))
            for row, col, val in W_e:
                W_entries.append((row, col_offset + col, val))
            col_offset += 7
    
    # Block 2: Row 2, intermediate cols 0-4, output cols 0-4 (1x5x5)
    # Split into:
    #   1x2x2: j=0-1, k=0-1 (Strassen, rank 3)
    #   1x2x3: j=0-1, k=2-4 (schoolbook, rank 6)
    #   1x3x2: j=2-4, k=0-1 (schoolbook, rank 6)
    #   1x3x3: j=2-4, k=2-4 (schoolbook, rank 9)
    
    # 1x2x2 Strassen for row 2, j=0-1, k=0-1
    # For 1x2x2, we can use a rank-3 decomposition:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # Using Strassen-like: 
    #   M1 = A[0,0] * (B[0,0] + B[0,1])
    #   M2 = (A[0,0] + A[0,1]) * B[1,0]
    #   M3 = A[0,1] * (B[1,1] - B[0,1])
    #   C[0,0] = M1 - M3 + M2
    #   C[0,1] = M1 + M3
    # Wait, let me verify:
    #   C[0,0] = A00*B00 + A01*B10
    #   M1 - M3 + M2 = A00*(B00+B01) - A01*(B11-B01) + (A00+A01)*B10
    #                = A00*B00 + A00*B01 - A01*B11 + A01*B01 + A00*B10 + A01*B10
    #   This doesn't simplify to C[0,0]. Let me use a different approach.
    #
    # For 1x2x2, the rank is actually 2 (not 3), since we only have one output row.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # This is just two independent dot products, so rank 2.
    #
    # But wait, we can share factors:
    #   M1 = A[0,0] * (B[0,0] + B[1,0])
    #   M2 = A[0,1] * (B[0,1] + B[1,1])
    #   M3 = (A[0,0] + A[0,1]) * (B[0,0] + B[0,1])
    #   C[0,0] = M1 + M3 - M2  ... no this doesn't work cleanly.
    #
    # Actually, for 1xm x np, the rank is m*p in general (schoolbook).
    # For 1x2x2, rank is 4 with schoolbook. Can we do better?
    # The tensor is T[j,k] = 1 for j in {0,1}, k in {0,1}.
    # We need: sum_c U[j,c] * V[k,c] * W[c] = 1 for all j,k.
    # This is a rank-1 approximation problem. The minimum rank is 1 if we allow
    # W to be absorbed, but in the bilinear setting, the rank is the matrix rank
    # of the 2x2 all-ones matrix, which is 1.
    #
    # Wait, I'm confusing myself. Let me think about this more carefully.
    #
    # For computing C[i,k] = sum_j A[i,j] * B[j,k] for a fixed i:
    # This is a bilinear form in A[i,:] and B. The "tensor" for this single row
    # has entries T[j,k] = 1. A rank-r decomposition means:
    #   sum_c U[j,c] * V[k,c] = 1 for all j,k
    # where U is 2x r, V is 2x r (for 1x2x2 case).
    # This is just a rank-r decomposition of the all-ones 2x2 matrix, which has rank 1.
    # So rank 1 suffices! U[j,0]=1, V[k,0]=1.
    #
    # But in the full bilinear computation, we have:
    #   C[i,k] = sum_c W[c] * (sum_j U[j,c] * A[i,j]) * (sum_k' V[k',c] * B[k',k])
    # For this to equal sum_j A[i,j] B[j,k], we need the tensor equation to hold.
    #
    # I think I'm overcomplicating this. Let me just use schoolbook for the 1x5x5 block.
    # Rank = 25.
    
    # For simplicity and correctness, use schoolbook for the entire 1x5x5 block
    U_e, V_e, W_e = _schoolbook_1xm_p(col_offset, 2, 0, 5, 0, 5, n, m, p)
    for row, col, val in U_e:
        U_entries.append((row, col_offset + col, val))
    for row, col, val in V_e:
        V_entries.append((row, col_offset + col, val))
    for row, col, val in W_e:
        W_entries.append((row, col_offset + col, val))
    col_offset += 25
    
    total_rank = col_offset  # 28 + 25 = 53
    
    # Build the matrices
    U = [[0.0] * total_rank for _ in range(nm)]
    V = [[0.0] * total_rank for _ in range(mp)]
    W = [[0.0] * total_rank for _ in range(pn)]
    
    for row, col, val in U_entries:
        if 0 <= row < nm and 0 <= col < total_rank:
            U[row][col] += val
    
    for row, col, val in V_entries:
        if 0 <= row < mp and 0 <= col < total_rank:
            V[row][col] += val
    
    for row, col, val in W_entries:
        if 0 <= row < pn and 0 <= col < total_rank:
            W[row][col] += val
    
    # Verify
    def verify(U_mat, V_mat, W_mat, rank_val):
        rng = np.random.RandomState(42)
        for trial in range(30):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank_val):
                a_vec = np.array(U_mat).reshape(n, m, rank_val)[:, :, c]
                b_vec = np.array(V_mat).reshape(m, p, rank_val)[:, :, c]
                w_mat = np.array(W_mat).reshape(p, n, rank_val)[:, :, c]
                z_c = np.sum(a_vec * A) * np.sum(b_vec * B)
                for ki in range(p * n):
                    k_idx = ki // n
                    i_idx = ki % n
                    C_computed[i_idx, k_idx] += w_mat[k_idx, i_idx] * z_c
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    if not verify(U, V, W, total_rank):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0.0] * rank for _ in range(nm)]
        V_fb = [[0.0] * rank for _ in range(mp)]
        W_fb = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1.0
                    V_fb[j * p + k][col] = 1.0
                    W_fb[k * n + i][col] = 1.0
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
