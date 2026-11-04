"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses a 2+2+1 block partition with Strassen for 2x2x2 blocks and schoolbook for others.
    Expected rank: 117.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partition sizes for n, m, p dimensions
    # 5 = 2 + 2 + 1
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Total rank calculation
    # 8 blocks of 2x2x2 use Strassen (rank 7)
    # 19 blocks use Schoolbook
    rank_strassen_blocks = 8
    rank_schoolbook_blocks = 0
    for pi in range(3):
        for ri in range(3):
            for qi in range(3):
                if sizes[pi] == 2 and sizes[ri] == 2 and sizes[qi] == 2:
                    pass
                else:
                    rank_schoolbook_blocks += sizes[pi] * sizes[ri] * sizes[qi]
    
    total_rank = rank_strassen_blocks * 7 + rank_schoolbook_blocks
    # total_rank should be 117
    
    # Initialize U, V, W
    # U shape: (n*m, rank) = (25, 117)
    # V shape: (m*p, rank) = (25, 117)
    # W shape: (p*n, rank) = (25, 117)
    U = np.zeros((25, total_rank), dtype=np.int64)
    V = np.zeros((25, total_rank), dtype=np.int64)
    W = np.zeros((25, total_rank), dtype=np.int64)
    
    col = 0
    
    # Iterate over all block combinations
    for pi in range(3):
        for ri in range(3):
            for qi in range(3):
                if time.time() > deadline:
                    break
                sp = sizes[pi]
                sr = sizes[ri]
                sq = sizes[qi]
                op = offsets[pi]
                orr = offsets[ri]
                oq = offsets[qi]
                
                if sp == 2 and sr == 2 and sq == 2:
                    # Strassen block
                    col = _add_strassen_block(U, V, W, op, orr, oq, col)
                else:
                    # Schoolbook block
                    col = _add_schoolbook_block(U, V, W, op, orr, oq, sp, sr, sq, col)
    
    # Convert to lists for compatibility if needed, but numpy arrays are usually fine
    # The seed program returned lists, so let's convert to be safe.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _add_strassen_block(U, V, W, op, orr, oq, col):
    """
    Adds Strassen decomposition for a 2x2x2 block.
    op, orr, oq are offsets for i, j, k dimensions.
    Returns new column index.
    """
    # Strassen formulas for 2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Local indices 0,1 for i,j,k
    
    # m1
    # a=A[0,0], d=A[1,1] -> U[op+0, orr+0], U[op+1, orr+1]
    # e=B[0,0], h=B[1,1] -> V[orr+0, oq+0], V[orr+1, oq+1]
    # C[0,0] += m1, C[1,1] += m1 -> W[oq+0, op+0], W[oq+1, op+1]
    
    U[op*5 + orr+0, col] = 1
    U[(op+1)*5 + orr+1, col] = 1
    V[(orr+0)*5 + oq+0, col] = 1
    V[(orr+1)*5 + oq+1, col] = 1
    W[(oq+0)*5 + op+0, col] = 1
    W[(oq+1)*5 + op+1, col] = 1
    col += 1
    
    # m2 = (c+d)e
    # c=A[1,0], d=A[1,1] -> U[(op+1)*5 + orr+0], U[(op+1)*5 + orr+1]
    # e=B[0,0] -> V[(orr+0)*5 + oq+0]
    # C[1,0] += m2 -> W[(oq+0)*5 + op+1]
    
    U[(op+1)*5 + orr+0, col] = 1
    U[(op+1)*5 + orr+1, col] = 1
    V[(orr+0)*5 + oq+0, col] = 1
    W[(oq+0)*5 + op+1, col] = 1
    col += 1
    
    # m3 = a(f-h)
    # a=A[0,0] -> U[op*5 + orr+0]
    # f=B[0,1], h=B[1,1] -> V[(orr+0)*5 + oq+1], V[(orr+1)*5 + oq+1]
    # C[0,1] += m3 -> W[(oq+1)*5 + op+0]
    
    U[op*5 + orr+0, col] = 1
    V[(orr+0)*5 + oq+1, col] = 1
    V[(orr+1)*5 + oq+1, col] = -1
    W[(oq+1)*5 + op+0, col] = 1
    col += 1
    
    # m4 = d(g-e)
    # d=A[1,1] -> U[(op+1)*5 + orr+1]
    # g=B[1,0], e=B[0,0] -> V[(orr+1)*5 + oq+0], V[(orr+0)*5 + oq+0]
    # C[0,0] += m4, C[1,0] += m4 -> W[(oq+0)*5 + op+0], W[(oq+0)*5 + op+1]
    
    U[(op+1)*5 + orr+1, col] = 1
    V[(orr+1)*5 + oq+0, col] = 1
    V[(orr+0)*5 + oq+0, col] = -1
    W[(oq+0)*5 + op+0, col] = 1
    W[(oq+0)*5 + op+1, col] = 1
    col += 1
    
    # m5 = (a+b)h
    # a=A[0,0], b=A[0,1] -> U[op*5 + orr+0], U[op*5 + orr+1]
    # h=B[1,1] -> V[(orr+1)*5 + oq+1]
    # C[0,0] -= m5, C[0,1] += m5 -> W[(oq+0)*5 + op+0] -= 1, W[(oq+1)*5 + op+0] += 1
    
    U[op*5 + orr+0, col] = 1
    U[op*5 + orr+1, col] = 1
    V[(orr+1)*5 + oq+1, col] = 1
    W[(oq+0)*5 + op+0, col] = -1
    W[(oq+1)*5 + op+0, col] = 1
    col += 1
    
    # m6 = (c-a)(e+f)
    # c=A[1,0], a=A[0,0] -> U[(op+1)*5 + orr+0], U[op*5 + orr+0]
    # e=B[0,0], f=B[0,1] -> V[(orr+0)*5 + oq+0], V[(orr+0)*5 + oq+1]
    # C[1,1] += m6 -> W[(oq+1)*5 + op+1]
    
    U[(op+1)*5 + orr+0, col] = 1
    U[op*5 + orr+0, col] = -1
    V[(orr+0)*5 + oq+0, col] = 1
    V[(orr+0)*5 + oq+1, col] = 1
    W[(oq+1)*5 + op+1, col] = 1
    col += 1
    
    # m7 = (b-d)(g+h)
    # b=A[0,1], d=A[1,1] -> U[op*5 + orr+1], U[(op+1)*5 + orr+1]
    # g=B[1,0], h=B[1,1] -> V[(orr+1)*5 + oq+0], V[(orr+1)*5 + oq+1]
    # C[0,0] += m7 -> W[(oq+0)*5 + op+0]
    
    U[op*5 + orr+1, col] = 1
    U[(op+1)*5 + orr+1, col] = -1
    V[(orr+1)*5 + oq+0, col] = 1
    V[(orr+1)*5 + oq+1, col] = 1
    W[(oq+0)*5 + op+0, col] = 1
    col += 1
    
    return col

def _add_schoolbook_block(U, V, W, op, orr, oq, sp, sr, sq, col):
    """
    Adds schoolbook decomposition for a sp x sr x sq block.
    Returns new column index.
    """
    for i in range(sp):
        for j in range(sr):
            for k in range(sq):
                # Global indices
                i_g = op + i
                j_g = orr + j
                k_g = oq + k
                
                # U index: i_g * 5 + j_g
                # V index: j_g * 5 + k_g
                # W index: k_g * 5 + i_g
                
                U[i_g * 5 + j_g, col] = 1
                V[j_g * 5 + k_g, col] = 1
                W[k_g * 5 + i_g, col] = 1
                col += 1
    return col

# EVOLVE-BLOCK-END
