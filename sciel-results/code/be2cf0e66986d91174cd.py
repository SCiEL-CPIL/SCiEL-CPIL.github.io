"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _block_split_schoolbook(n, m, p, block_sizes):
    """
    Split the p columns into blocks and use schoolbook for each block.
    This is equivalent to full schoolbook but structured.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for k_start, k_end in block_sizes:
        for i in range(n):
            for j in range(m):
                for k in range(k_start, k_end):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    return {"U": U, "V": V, "W": W}


def _bini_2x2x2():
    """
    Return the 7 rank-1 tensors for 2x2 * 2x2 -> 2x2 multiplication.
    Using Bini-Motta-Valtrano decomposition.
    Convention: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
    U indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    V indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    W indices: 0=C[0,0], 1=C[1,0], 2=C[0,1], 3=C[1,1]
    """
    # The correct 7-term Bini decomposition:
    # We need: C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Term 0: A[0,0]*B[0,0] -> C[0,0]: U[0]=1, V[0]=1, W[0]=1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]: U[1]=1, V[2]=1, W[0]=1
    # Term 2: A[1,0]*B[0,0] -> C[1,0]: U[2]=1, V[0]=1, W[1]=1
    # Term 3: A[1,1]*B[1,0] -> C[1,0]: U[3]=1, V[2]=1, W[1]=1
    # Term 4: A[0,0]*B[0,1] -> C[0,1]: U[0]=1, V[1]=1, W[2]=1
    # Term 5: A[0,1]*B[1,1] -> C[0,1]: U[1]=1, V[3]=1, W[2]=1
    # Term 6: A[1,0]*B[0,1] -> C[1,1]: U[2]=1, V[1]=1, W[3]=1
    # Term 7: A[1,1]*B[1,1] -> C[1,1]: U[3]=1, V[3]=1, W[3]=1
    # That's 8 terms. The Bini trick combines some.
    
    # Actually, the standard Bini rank-7 uses a different construction.
    # Let me use the known correct 7-term decomposition:
    
    # For 2x2 matrix multiplication, the tensor rank is 7.
    # One valid decomposition:
    terms = []
    # t0: A00*B00 -> C00
    terms.append(([1,0,0,0], [1,0,0,0], [1,0,0,0]))
    # t1: A01*B10 -> C00  (wait, this should go to C00? No)
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # 8 schoolbook terms. To get 7, we use the Bini identity:
    # (A00+A01+A10+A11)(B00+B10+B01+B11) - ... no, that's not it.
    
    # The correct Bini-Motta-Valtrano 7-term decomposition:
    # f1: u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]  # A00*B00 -> C00
    # f2: u=[0,1,0,0], v=[0,0,1,0], w=[0,1,0,0]  # A01*B10 -> C10? No, W[1]=C[1,0]
    #     But we need A01*B10 -> C00, not C10.
    
    # I think the issue is that the standard Bini decomposition doesn't simply
    # map to this indexing convention in an obvious way. Let me just use the
    # schoolbook rank 8 for 2x2x2 and see if the overall structure helps.
    
    # For now, return the 8-term schoolbook (which is valid, just not minimal):
    terms = []
    # C[0,0]:
    terms.append(([1,0,0,0], [1,0,0,0], [1,0,0,0]))  # A00*B00
    terms.append(([0,1,0,0], [0,0,1,0], [1,0,0,0]))  # A01*B10
    # C[1,0]:
    terms.append(([0,0,1,0], [1,0,0,0], [0,1,0,0]))  # A10*B00
    terms.append(([0,0,0,1], [0,0,1,0], [0,1,0,0]))  # A11*B10
    # C[0,1]:
    terms.append(([1,0,0,0], [0,0,0,1], [0,0,1,0]))  # A00*B01
    terms.append(([0,1,0,0], [0,0,0,1], [0,0,1,0]))  # A01*B11
    # C[1,1]:
    terms.append(([0,0,1,0], [0,0,0,1], [0,0,0,1]))  # A10*B01
    terms.append(([0,0,0,1], [0,0,0,1], [0,0,0,1]))  # A11*B11
    return terms


def _construct_blockwise_bini(n, m, p):
    """
    Construct a decomposition by splitting 4x4*4x9 into:
    - Two 4x4*4x4 blocks (columns 0-3 and 4-7)
    - One 4x4*4x1 block (column 8)
    
    For each 4x4*4x4 block, split into four 2x2*2x2*2x2 sub-blocks.
    Use Bini rank-7 (or schoolbook rank-8) for each.
    For the 4x4*4x1 block, use schoolbook rank 16.
    
    Total with Bini: 2*4*7 + 16 = 56 + 16 = 72
    Total with schoolbook: 2*4*8 + 16 = 64 + 16 = 80
    """
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # For 4x4*4x4 block, we have 4 sub-blocks of 2x2*2x2->2x2
    # Sub-block (bi, bj) where bi, bj in {0,1}:
    #   A sub-block rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    #   B sub-block rows 2*bj..2*bj+1, cols 2*bi..2*bi+1 (for the output)
    #   C sub-block rows 2*bi..2*bi+1, cols 2*bi..2*bi+1
    
    # Actually, let me think about this more carefully.
    # A is 4x4, B is 4x4, C = A*B is 4x4.
    # Partition A into 2x2 blocks: A[bi, bj] for bi,bj in {0,1}
    # Partition B into 2x2 blocks: B[bj, bk] for bj,bk in {0,1}
    # C[bi, bk] = sum_j A[bi, j] * B[j, bk]
    
    # For each (bi, bk), C[bi, bk] = A[bi,0]*B[0,bk] + A[bi,1]*B[1,bk]
    # Each term is a 2x2 * 2x2 -> 2x2 multiplication.
    
    # So we have 4 (bi,bk) pairs, each requiring 2 multiplications of 2x2 blocks.
    # Total: 8 multiplications of 2x2*2x2->2x2.
    # With Bini rank-7: 8*7 = 56
    # With schoolbook rank-8: 8*8 = 64
    
    bini_terms = _bini_2x2x2()
    
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                for u, v, w in bini_terms:
                    # Create full-length vectors
                    u_full = [0] * (n * m)  # 16
                    v_full = [0] * (m * p)  # 16 for 4x4 block
                    w_full = [0] * (p * n)  # 16 for 4x4 block
                    
                    # Map 2x2 indices to full indices
                    # A sub-block: rows 2*bi + i, cols 2*bj + j -> index (2*bi+i)*4 + (2*bj+j)
                    # B sub-block: rows 2*bj + j, cols 2*bk + k -> index (2*bj+j)*4 + (2*bk+k)
                    # C sub-block: rows 2*bi + i, cols 2*bk + k -> index (2*bk+k)*4 + (2*bi+i)
                    
                    for idx in range(4):
                        i2 = idx // 2
                        j2 = idx % 2
                        if u[idx] != 0:
                            u_full[(2*bi + i2) * m + (2*bj + j2)] = u[idx]
                        if v[idx] != 0:
                            v_full[(2*bj + i2) * p + (2*bk + j2)] = v[idx]
                        if w[idx] != 0:
                            w_full[(2*bk + i2) * n + (2*bi + j2)] = w[idx]
                    
                    add_term(u_full, v_full, w_full)
    
    # Now for the 4x4*4x1 block (column 8 of B)
    # C[i, 8] = sum_j A[i,j] * B[j, 8]
    # Schoolbook: 4*4 = 16 terms
    for i in range(n):
        for j in range(m):
            u_full = [0] * (n * m)
            v_full = [0] * (m * p)
            w_full = [0] * (p * n)
            u_full[i * m + j] = 1
            v_full[j * p + 8] = 1
            w_full[8 * n + i] = 1
            add_term(u_full, v_full, w_full)
    
    rank = len(U[0])
    # Transpose to correct shape
    U_out = [[0] * rank for _ in range(n * m)]
    V_out = [[0] * rank for _ in range(m * p)]
    W_out = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for u in range(n * m):
            U_out[u][c] = U[c][u]
        for v in range(m * p):
            V_out[v][c] = V[c][v]
        for w in range(p * n):
            W_out[w][c] = W[c][w]
    
    return {"U": U_out, "V": V_out, "W": W_out}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Try blockwise Bini approach
    try:
        result = _construct_blockwise_bini(n, m, p)
        rank = len(result["U"][0])
        
        # Verify correctness
        T = _build_tensor(n, m, p)
        Ua = np.array(result["U"], dtype=object)
        Va = np.array(result["V"], dtype=object)
        Wa = np.array(result["W"], dtype=object)
        
        valid = True
        for io in range(p * n):
            for u in range(n * m):
                for v in range(m * p):
                    s = 0
                    for c in range(rank):
                        s += Ua[u, c] * Va[v, c] * Wa[io, c]
                    if s != T[io, u, v]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return result
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
