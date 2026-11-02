import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """Return 7 (u,v,w) triples for 2x2x2 matrix multiplication (Strassen).
    Convention: sum_t u[i*2+j]*v[j*2+k]*w[i*2+k] == 1 for all i,j,k in {0,1}
    u is 4-element flat (2x2 row-major), v is 4-element flat, w is 4-element flat.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1,0,0,1], [1,0,1,1], [1,0,0,0]))
    # M2 = (A10+A11)B00
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    # M3 = A00(B10-B11)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))
    # M4 = A11(B01-B00)
    terms.append(([0,0,0,1], [1,1,0,0], [0,0,0,1]))
    # M5 = (A00+A01)B11
    terms.append(([1,1,0,0], [0,0,1,1], [1,0,0,0]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([1,-1,1,0], [1,1,0,0], [0,1,0,0]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0,1,-1,0], [0,1,1,1], [0,0,1,0]))
    return terms

def _verify_tensor_decomposition(terms, n, m, p):
    """Verify that the list of (u,v,w) terms correctly decomposes the n x m x p matmul tensor.
    Each u has length n*m, v has length m*p, w has length p*n.
    Convention: for each (i,j,k), sum_t u[i*m+j]*v[j*p+k]*w[k*n+i] == 1
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_decomp = np.zeros((n, m, p))
    for u, v, w in terms:
        u_mat = np.array(u, dtype=float).reshape(n, m)
        v_mat = np.array(v, dtype=float).reshape(m, p)
        w_mat = np.array(w, dtype=float).reshape(p, n)
        # T_decomp[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        T_decomp += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
    
    return np.allclose(T_decomp, T, atol=1e-10)

def _build_block_decomposition(n, m, p):
    """Build a block decomposition using Strassen for 2x2x2 and schoolbook for the rest.
    
    Strategy: Split n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2.
    Blocks of size 2x2x2 use Strassen (rank 7), others use schoolbook.
    
    Block grid:
      A rows: [0:2], [2:3]
      A cols / B rows: [0:2], [2:4], [4:5]
      B cols: [0:2], [2:4], [4:6]
    
    2x2x2 blocks (Strassen, rank 7):
      (A[0:2, 0:2], B[0:2, 0:2]) -> C[0:2, 0:2]
      (A[0:2, 0:2], B[0:2, 2:4]) -> C[0:2, 2:4]
      (A[0:2, 0:2], B[0:2, 4:6]) -> C[0:2, 4:6]
      (A[0:2, 2:4], B[2:4, 0:2]) -> C[0:2, 0:2]
      (A[0:2, 2:4], B[2:4, 2:4]) -> C[0:2, 2:4]
      (A[0:2, 2:4], B[2:4, 4:6]) -> C[0:2, 4:6]
      (A[2:3, 0:2], B[0:2, 0:2]) -> C[2:3, 0:2]  -- this is 1x2x2, not 2x2x2
    """
    # Let me reconsider. With n=3, m=5, p=6:
    # Split n: 2+1
    # Split m: 2+2+1  
    # Split p: 2+2+2
    #
    # 2x2x2 blocks (need A block 2x2, B block 2x2, C block 2x2):
    # A rows [0:2], A cols [0:2], B rows [0:2], B cols [0:2] -> C[0:2, 0:2]  (2x2x2)
    # A rows [0:2], A cols [0:2], B rows [0:2], B cols [2:4] -> C[0:2, 2:4]  (2x2x2)
    # A rows [0:2], A cols [0:2], B rows [0:2], B cols [4:6] -> C[0:2, 4:6]  (2x2x2)
    # A rows [0:2], A cols [2:4], B rows [2:4], B cols [0:2] -> C[0:2, 0:2]  (2x2x2)
    # A rows [0:2], A cols [2:4], B rows [2:4], B cols [2:4] -> C[0:2, 2:4]  (2x2x2)
    # A rows [0:2], A cols [2:4], B rows [2:4], B cols [4:6] -> C[0:2, 4:6]  (2x2x2)
    #
    # That's 6 blocks of 2x2x2, each rank 7 = 42
    #
    # Remaining blocks (schoolbook):
    # A rows [0:2], A cols [4:5], B rows [4:5], B cols [0:2] -> C[0:2, 0:2]  (2x1x2, rank 4)
    # A rows [0:2], A cols [4:5], B rows [4:5], B cols [2:4] -> C[0:2, 2:4]  (2x1x2, rank 4)
    # A rows [0:2], A cols [4:5], B rows [4:5], B cols [4:6] -> C[0:2, 4:6]  (2x1x2, rank 4)
    # A rows [2:3], A cols [0:2], B rows [0:2], B cols [0:2] -> C[2:3, 0:2]  (1x2x2, rank 4)
    # A rows [2:3], A cols [0:2], B rows [0:2], B cols [2:4] -> C[2:3, 2:4]  (1x2x2, rank 4)
    # A rows [2:3], A cols [0:2], B rows [0:2], B cols [4:6] -> C[2:3, 4:6]  (1x2x2, rank 4)
    # A rows [2:3], A cols [2:4], B rows [2:4], B cols [0:2] -> C[2:3, 0:2]  (1x2x2, rank 4)
    # A rows [2:3], A cols [2:4], B rows [2:4], B cols [2:4] -> C[2:3, 2:4]  (1x2x2, rank 4)
    # A rows [2:3], A cols [2:4], B rows [2:4], B cols [4:6] -> C[2:3, 4:6]  (1x2x2, rank 4)
    # A rows [2:3], A cols [4:5], B rows [4:5], B cols [0:2] -> C[2:3, 0:2]  (1x1x2, rank 2)
    # A rows [2:3], A cols [4:5], B rows [4:5], B cols [2:4] -> C[2:3, 2:4]  (1x1x2, rank 2)
    # A rows [2:3], A cols [4:5], B rows [4:5], B cols [4:6] -> C[2:3, 4:6]  (1x1x2, rank 2)
    #
    # Total: 42 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    
    strassen_terms = _strassen_2x2x2()
    
    # Define block specifications: (a_row_start, a_col_start, b_col_start, a_rows, a_cols, b_cols, is_strassen)
    # For Strassen blocks: a_rows=2, a_cols=2, b_cols=2
    # For schoolbook: various sizes
    
    blocks = []
    
    # 6 Strassen 2x2x2 blocks
    # A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    blocks.append((0, 0, 0, 2, 2, 2, True))
    # A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4]
    blocks.append((0, 0, 2, 2, 2, 2, True))
    # A[0:2, 0:2] x B[0:2, 4:6] -> C[0:2, 4:6]
    blocks.append((0, 0, 4, 2, 2, 2, True))
    # A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2]
    blocks.append((0, 2, 0, 2, 2, 2, True))
    # A[0:2, 2:4] x B[2:4, 2:4] -> C[0:2, 2:4]
    blocks.append((0, 2, 2, 2, 2, 2, True))
    # A[0:2, 2:4] x B[2:4, 4:6] -> C[0:2, 4:6]
    blocks.append((0, 2, 4, 2, 2, 2, True))
    
    # Schoolbook blocks
    # A[0:2, 4:5] x B[4:5, 0:2] -> C[0:2, 0:2]  (2x1x2)
    blocks.append((0, 4, 0, 2, 1, 2, False))
    # A[0:2, 4:5] x B[4:5, 2:4] -> C[0:2, 2:4]  (2x1x2)
    blocks.append((0, 4, 2, 2, 1, 2, False))
    # A[0:2, 4:5] x B[4:5, 4:6] -> C[0:2, 4:6]  (2x1x2)
    blocks.append((0, 4, 4, 2, 1, 2, False))
    
    # A[2:3, 0:2] x B[0:2, 0:2] -> C[2:3, 0:2]  (1x2x2)
    blocks.append((2, 0, 0, 1, 2, 2, False))
    # A[2:3, 0:2] x B[0:2, 2:4] -> C[2:3, 2:4]  (1x2x2)
    blocks.append((2, 0, 2, 1, 2, 2, False))
    # A[2:3, 0:2] x B[0:2, 4:6] -> C[2:3, 4:6]  (1x2x2)
    blocks.append((2, 0, 4, 1, 2, 2, False))
    # A[2:3, 2:4] x B[2:4, 0:2] -> C[2:3, 0:2]  (1x2x2)
    blocks.append((2, 2, 0, 1, 2, 2, False))
    # A[2:3, 2:4] x B[2:4, 2:4] -> C[2:3, 2:4]  (1x2x2)
    blocks.append((2, 2, 2, 1, 2, 2, False))
    # A[2:3, 2:4] x B[2:4, 4:6] -> C[2:3, 4:6]  (1x2x2)
    blocks.append((2, 2, 4, 1, 2, 2, False))
    
    # A[2:3, 4:5] x B[4:5, 0:2] -> C[2:3, 0:2]  (1x1x2)
    blocks.append((2, 4, 0, 1, 1, 2, False))
    # A[2:3, 4:5] x B[4:5, 2:4] -> C[2:3, 2:4]  (1x1x2)
    blocks.append((2, 4, 2, 1, 1, 2, False))
    # A[2:3, 4:5] x B[4:5, 4:6] -> C[2:3, 4:6]  (1x1x2)
    blocks.append((2, 4, 4, 1, 1, 2, False))
    
    # Compute total rank
    total_rank = 0
    for (ar, ac, bc, a_r, a_c, b_c, is_str) in blocks:
        if is_str:
            total_rank += 7
        else:
            total_rank += a_r * a_c * b_c
    
    # Build U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    deadline = time.time() + 140.0
    
    for (ar, ac, bc, a_r, a_c, b_c, is_str) in blocks:
        if is_str:
            # Strassen 2x2x2
            for t in range(7):
                u_small, v_small, w_small = strassen_terms[t]
                # u_small is 4 elements (2x2), maps to A[ar:ar+2, ac:ac+2]
                for ii in range(2):
                    for jj in range(2):
                        U[(ar+ii) * m + (ac+jj)][col] += u_small[ii*2+jj]
                # v_small is 4 elements (2x2), maps to B[ac:ac+2, bc:bc+2]
                for jj in range(2):
                    for kk in range(2):
                        V[(ac+jj) * p + (bc+kk)][col] += v_small[jj*2+kk]
                # w_small is 4 elements (2x2), maps to C[ar:ar+2, bc:bc+2]
                # W index: k*n + i, so W[(bc+kk)*n + (ar+ii)]
                for ii in range(2):
                    for kk in range(2):
                        W[(bc+kk) * n + (ar+ii)][col] += w_small[ii*2+kk]
                col += 1
        else:
            # Schoolbook for a_r x a_c x b_c
            for ii in range(a_r):
                for jj in range(a_c):
                    for kk in range(b_c):
                        U[(ar+ii) * m + (ac+jj)][col] = 1
                        V[(ac+jj) * p + (bc+kk)][col] = 1
                        W[(bc+kk) * n + (ar+ii)][col] = 1
                        col += 1
                        if time.time() > deadline:
                            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the full decomposition with random matrices."""
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        a_u = u_2d * A
        b_v = v_2d * B
        contrib = a_u @ b_v
        C_computed += w_2d.T * contrib
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Try the block decomposition with Strassen
    result = _build_block_decomposition(n, m, p)
    
    if time.time() > deadline - 5:
        return result
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If verification fails, try a simpler approach: split m=5 as 3+2, p=6 as 3+3
    # Use Bini rank-23 for 3x3x3 blocks if available, otherwise schoolbook
    # For now, fall back to schoolbook
    if time.time() > deadline - 3:
        return result
    
    return _schoolbook_decomposition(n, m, p)
