"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly for integer/half-integer entries."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    if np.max(np.abs(T_computed - target)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A22(B12-B11) -> C21 += M4, C11 += M4
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A11+A12)B22 -> C12 += M5, C11 -= M5
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _block_decompose(n, m, p, sizes_a, sizes_b, sizes_c):
    """
    Block decomposition for n×m×p matrix multiplication.
    sizes_a: partition of n, sizes_b: partition of m, sizes_c: partition of p.
    Uses Strassen for 2×2×2 blocks, schoolbook otherwise.
    """
    U_terms = []
    V_terms = []
    W_terms = []
    rank_offset = 0

    for i in range(len(sizes_a)):
        for j in range(len(sizes_b)):
            for k in range(len(sizes_c)):
                sa, sb, sc = sizes_a[i], sizes_b[j], sizes_c[k]
                if sa == 0 or sb == 0 or sc == 0:
                    continue

                if sa == 2 and sb == 2 and sc == 2:
                    U_blk, V_blk, W_blk = _strassen_2x2()
                    r = 7
                else:
                    r = sa * sb * sc
                    U_blk = np.zeros((sa*sb, r), dtype=np.int64)
                    V_blk = np.zeros((sb*sc, r), dtype=np.int64)
                    W_blk = np.zeros((sc*sa, r), dtype=np.int64)
                    col = 0
                    for a in range(sa):
                        for b in range(sb):
                            for c in range(sc):
                                U_blk[a*sb+b, col] = 1
                                V_blk[b*sc+c, col] = 1
                                W_blk[c*sa+a, col] = 1
                                col += 1

                row_start_A = sum(sizes_a[:i])
                row_start_B = sum(sizes_b[:j])
                row_start_C = sum(sizes_c[:k])

                for rr in range(r):
                    for a_idx in range(sa*sb):
                        val = U_blk[a_idx, rr]
                        if val != 0:
                            U_terms.append((row_start_A + a_idx, rank_offset + rr, val))
                    for b_idx in range(sb*sc):
                        val = V_blk[b_idx, rr]
                        if val != 0:
                            V_terms.append((row_start_B + b_idx, rank_offset + rr, val))
                    for c_idx in range(sc*sa):
                        val = W_blk[c_idx, rr]
                        if val != 0:
                            W_terms.append((row_start_C + c_idx, rank_offset + rr, val))
                rank_offset += r

    total_rank = rank_offset
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    for row, col, val in U_terms:
        U[row, col] += val
    for row, col, val in V_terms:
        V[row, col] += val
    for row, col, val in W_terms:
        W[row, col] += val
    return U.tolist(), V.tolist(), W.tolist(), total_rank

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try various block decompositions
    partitions = [
        ([2, 2, 1], [2, 2, 1], [2, 2, 1]),
        ([2, 3], [2, 3], [2, 3]),
        ([3, 2], [3, 2], [3, 2]),
        ([1, 2, 2], [1, 2, 2], [1, 2, 2]),
        ([2, 1, 2], [2, 1, 2], [2, 1, 2]),
        ([1, 4], [1, 4], [1, 4]),
        ([4, 1], [4, 1], [4, 1]),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
    ]
    
    for sa, sb, sc in partitions:
        if time.time() > deadline - 30:
            break
        U_blk, V_blk, W_blk, blk_rank = _block_decompose(n, m, p, sa, sb, sc)
        if blk_rank < best_rank:
            if _verify(U_blk, V_blk, W_blk, n, m, p, blk_rank):
                best_rank = blk_rank
                best_result = {"U": U_blk, "V": V_blk, "W": W_blk}
                print(f"Block decomposition {sa}|{sb}|{sc} rank: {best_rank}")
            else:
                print(f"Block decomposition {sa}|{sb}|{sc} rank {blk_rank} FAILED")
    
    return best_result
# EVOLVE-BLOCK-END
