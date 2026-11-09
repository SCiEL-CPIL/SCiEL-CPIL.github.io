"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    rank = 7
    U = [[0]*rank for _ in range(4)]  # 2x2 = 4 entries, indexed [i*2+j]
    V = [[0]*rank for _ in range(4)]  # 2x2 = 4 entries, indexed [j*2+k]
    W = [[0]*rank for _ in range(4)]  # 2x2 = 4 entries, indexed [k*2+i]
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = U[0], A22 = U[3], B11 = V[0], B22 = V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    W[0*2+0][0] = 1  # C11 (i=0,k=0) -> W[0]
    W[1*2+1][0] = 1  # C22 (i=1,k=1) -> W[3]
    
    # M2 = (A21+A22)B11
    # A21 = U[2], A22 = U[3], B11 = V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2, C22 -= M2
    W[0*2+1][1] = 1  # C21 (i=1,k=0) -> W[2]
    W[1*2+1][1] = -1 # C22 (i=1,k=1) -> W[3]
    
    # M3 = A11(B12-B22)
    # A11 = U[0], B12 = V[1], B22 = V[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 += M3, C22 += M3
    W[1*2+0][2] = 1  # C12 (i=0,k=1) -> W[1]
    W[1*2+1][2] = 1  # C22 (i=1,k=1) -> W[3]
    
    # M4 = A22(B21-B11)
    # A22 = U[3], B21 = V[2], B11 = V[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    W[0*2+0][3] = 1  # C11 (i=0,k=0) -> W[0]
    W[0*2+1][3] = 1  # C21 (i=1,k=0) -> W[2]
    
    # M5 = (A11+A12)B22
    # A11 = U[0], A12 = U[1], B22 = V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    W[0*2+0][4] = -1 # C11 (i=0,k=0) -> W[0]
    W[1*2+0][4] = 1  # C12 (i=0,k=1) -> W[1]
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = U[2], A11 = U[0], B11 = V[0], B12 = V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[1*2+1][5] = 1  # C22 (i=1,k=1) -> W[3]
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = U[1], A22 = U[3], B21 = V[2], B22 = V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0*2+0][6] = 1  # C11 (i=0,k=0) -> W[0]
    
    return U, V, W

def _mul_2x2_by_2x1():
    """Multiply 2x2 A by 2x1 B to get 2x1 C. Rank 4.
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indexing: U[i*2+j, r], V[j*1+k, r]=V[j+k, r], W[k*1+i, r]=W[k+i, r]
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    """
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    # C[0] = A00*B0 + A01*B1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1  # A00*B0 -> C0
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1  # A01*B1 -> C0
    # C[1] = A10*B0 + A11*B1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1  # A10*B0 -> C1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1  # A11*B1 -> C1
    
    return U, V, W

def _mul_1x2_by_2x2():
    """Multiply 1x2 A by 2x2 B to get 1x2 C. Rank 4.
    U shape: (2, 4), V shape: (4, 4), W shape: (2, 4)
    Indexing: U[i*2+j, r]=U[j, r], V[j*2+k, r], W[k*1+i, r]=W[k, r]
    C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(2)]
    
    # C[0] = A0*B00 + A1*B10
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1  # A0*B00 -> C0
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1  # A1*B10 -> C0
    # C[1] = A0*B01 + A1*B11
    U[0][2] = 1; V[1][2] = 1; W[1][2] = 1  # A0*B01 -> C1
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1  # A1*B11 -> C1
    
    return U, V, W

def _mul_2x1_by_1x2():
    """Multiply 2x1 A by 1x2 B to get 2x2 C. Rank 1 (outer product).
    U shape: (2, 1), V shape: (2, 1), W shape: (4, 1)
    Indexing: U[i*1+j, r]=U[i, r], V[j*1+k, r]=V[k, r], W[k*2+i, r]
    C[i,k] = A[i]*B[k]
    """
    rank = 1
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(4)]
    
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    # C[i,k] = A[i]*B[k]
    for i in range(2):
        for k in range(2):
            W[k*2+i][0] = 1
    
    return U, V, W

def _mul_2x1_by_1x1():
    """Multiply 2x1 A by 1x1 B to get 2x1 C. Rank 2.
    U shape: (2, 2), V shape: (1, 2), W shape: (2, 2)
    C[i] = A[i]*B[0]
    """
    rank = 2
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(1)]
    W = [[0]*rank for _ in range(2)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[0][1] = 1; W[1][1] = 1
    
    return U, V, W

def _mul_1x1_by_1x2():
    """Multiply 1x1 A by 1x2 B to get 1x2 C. Rank 2.
    U shape: (1, 2), V shape: (2, 2), W shape: (2, 2)
    C[k] = A[0]*B[k]
    """
    rank = 2
    U = [[0]*rank for _ in range(1)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    
    return U, V, W

def _mul_1x2_by_2x1():
    """Multiply 1x2 A by 2x1 B to get 1x1 C. Rank 2 (dot product).
    U shape: (2, 2), V shape: (2, 2), W shape: (1, 2)
    C[0] = A[0]*B[0] + A[1]*B[1]
    """
    rank = 2
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(1)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    
    return U, V, W

def _mul_1x1_by_1x1():
    """Multiply 1x1 A by 1x1 B to get 1x1 C. Rank 1.
    """
    rank = 1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block sizes: 2, 2, 1
    block_sizes = [2, 2, 1]
    
    # Precompute all sub-multiplication decompositions
    # Types: (a, b, c) where A is a x b, B is b x c, C is a x c
    decomp_cache = {}
    
    def get_decomp(a, b, c):
        key = (a, b, c)
        if key not in decomp_cache:
            if a == 2 and b == 2 and c == 2:
                decomp_cache[key] = _strassen_2x2()
            elif a == 2 and b == 2 and c == 1:
                decomp_cache[key] = _mul_2x2_by_2x1()
            elif a == 1 and b == 2 and c == 2:
                decomp_cache[key] = _mul_1x2_by_2x2()
            elif a == 2 and b == 1 and c == 2:
                decomp_cache[key] = _mul_2x1_by_1x2()
            elif a == 2 and b == 1 and c == 1:
                decomp_cache[key] = _mul_2x1_by_1x1()
            elif a == 1 and b == 1 and c == 2:
                decomp_cache[key] = _mul_1x1_by_1x2()
            elif a == 1 and b == 2 and c == 1:
                decomp_cache[key] = _mul_1x2_by_2x1()
            elif a == 1 and b == 1 and c == 1:
                decomp_cache[key] = _mul_1x1_by_1x1()
            else:
                # Fallback to schoolbook
                rank = a * b * c
                U = [[0]*rank for _ in range(a*b)]
                V = [[0]*rank for _ in range(b*c)]
                W = [[0]*rank for _ in range(c*a)]
                col = 0
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            U[i*b+j][col] = 1
                            V[j*c+k][col] = 1
                            W[k*a+i][col] = 1
                            col += 1
                decomp_cache[key] = (U, V, W)
        return decomp_cache[key]
    
    # Accumulate U, V, W matrices
    # U: (n*m, rank_total) = (25, rank_total)
    # V: (m*p, rank_total) = (25, rank_total)
    # W: (p*n, rank_total) = (25, rank_total)
    U_entries = []  # list of (row, col_offset, value)
    V_entries = []
    W_entries = []
    
    current_rank = 0
    
    for i in range(3):  # block row of A / block row of C
        for j in range(3):  # block col of A / block row of B
            for k in range(3):  # block col of B / block col of C
                a_size = block_sizes[i]  # rows of A_ij
                b_size = block_sizes[j]  # cols of A_ij = rows of B_jk
                c_size = block_sizes[k]  # cols of B_jk
                
                U_sub, V_sub, W_sub = get_decomp(a_size, b_size, c_size)
                rank_sub = len(U_sub[0]) if U_sub else 0
                
                # Row offsets in the full matrices
                # A is n x m = 5 x 5, blocks are block_sizes[i] x block_sizes[j]
                a_row_offset = sum(block_sizes[:i])  # starting row of block i in A
                a_col_offset = sum(block_sizes[:j])  # starting col of block j in A
                # B is m x p = 5 x 5, blocks are block_sizes[j] x block_sizes[k]
                b_row_offset = sum(block_sizes[:j])  # starting row of block j in B
                b_col_offset = sum(block_sizes[:k])  # starting col of block k in B
                # C is n x p = 5 x 5, blocks are block_sizes[i] x block_sizes[k]
                c_row_offset = sum(block_sizes[:k])  # starting col of block k in C (for W indexing)
                c_col_offset = sum(block_sizes[:i])  # starting row of block i in C (for W indexing)
                
                # Add entries
                for r in range(rank_sub):
                    for u_row in range(len(U_sub)):
                        val = U_sub[u_row][r]
                        if val != 0:
                            full_row = a_row_offset * m + a_col_offset + u_row
                            U_entries.append((full_row, current_rank + r, val))
                    
                    for v_row in range(len(V_sub)):
                        val = V_sub[v_row][r]
                        if val != 0:
                            full_row = b_row_offset * p + b_col_offset + v_row
                            V_entries.append((full_row, current_rank + r, val))
                    
                    for w_row in range(len(W_sub)):
                        val = W_sub[w_row][r]
                        if val != 0:
                            # W is indexed as W[k*n + i] for C[i,k]
                            # In sub-problem, W_sub is indexed as W_sub[k_sub * a_size + i_sub]
                            # k_sub is the column index in C_sub (0..c_size-1)
                            # i_sub is the row index in C_sub (0..a_size-1)
                            k_sub = w_row // a_size
                            i_sub = w_row % a_size
                            full_row = (c_row_offset + k_sub) * n + (c_col_offset + i_sub)
                            W_entries.append((full_row, current_rank + r, val))
                
                current_rank += rank_sub
    
    total_rank = current_rank
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for row, col, val in U_entries:
        U[row][col] = val
    for row, col, val in V_entries:
        V[row][col] = val
    for row, col, val in W_entries:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
