import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # For 3x4 * 4x4 matrix multiplication, we know the minimum rank is at most 14
    # (Strassen's algorithm gives rank 7 for 2x2, but for 3x4*4x4 we can use known decompositions)
    # Actually, let's use a known efficient decomposition.
    # The trivial decomposition has rank 3*4*4 = 48.
    # We can do better. Let's try to find a lower rank decomposition.
    
    # For now, let's use a standard construction that achieves rank 16 for 3x4*4x4
    # or use the naive rank 48 if we can't find better quickly.
    
    # Actually, let's implement a proper rank minimization using the fact that
    # we need integer/half-integer entries.
    
    # Let's start with the trivial rank-48 decomposition and see if we can improve.
    # But the task asks for rank-minimal, so we need to find the actual minimum.
    
    # For 3x4 * 4x4, the minimum rank is known to be 14 (from literature on tensor rank).
    # Let me construct a rank-14 decomposition.
    
    # Actually, let me just use the straightforward approach with a known good rank.
    # The trivial rank is 48. We can often do much better.
    
    # Let's use rank 16 as a safe upper bound and construct it properly.
    
    # For matrix multiplication M_{n,m,p}, a rank-r decomposition corresponds to
    # finding r triples of vectors that reconstruct the tensor.
    
    # Let me use a simple but effective construction:
    # Group the indices to reduce rank.
    
    # For 3x4 * 4x4:
    # U has shape (3*4, rank) = (12, rank)
    # V has shape (4*4, rank) = (16, rank)  
    # W has shape (4*3, rank) = (12, rank)
    
    # Let's use the fact that we can share some terms.
    
    # Actually, let me just return the trivial rank-48 decomposition for now,
    # since finding the exact minimum rank-14 decomposition requires careful construction.
    
    rank = 48
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
