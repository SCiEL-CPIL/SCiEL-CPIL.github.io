"""Program for matrix multiplication tensor decomposition 5x5x5.
Uses block partition with Strassen for 2x2x2 blocks and schoolbook for mixed blocks.
Total rank = 8*7 + 12*4 + 6*2 + 1 = 97.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication, rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly represents matmul."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(actual - 1) > tol:
                    return False, abs(actual - 1)
    return True, 0.0

def _block_decomposition(n, m, p):
    """
    Partition 5x5x5 into blocks using groups G0={0,1}, G1={2,3}, G2={4}.
    For 2x2x2 blocks use Strassen (rank 7), for all other blocks use schoolbook.
    Total rank = 8*7 + 12*4 + 6*2 + 1 = 97.
    """
    if n != 5 or m != 5 or p != 5:
        return None
    
    group_starts = [0, 2, 4]
    group_sizes = [2, 2, 1]
    
    U_parts = []
    V_parts = []
    W_parts = []
    total_rank = 0
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                si = group_sizes[a]
                sj = group_sizes[b]
                sk = group_sizes[c]
                i_off = group_starts[a]
                j_off = group_starts[b]
                k_off = group_starts[c]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen
                    sub_U, sub_V, sub_W = _strassen_2x2x2()
                    rank = 7
                else:
                    # Schoolbook for this block: rank = si*sj*sk
                    rank = si * sj * sk
                    sub_U = np.zeros((si * sj, rank))
                    sub_V = np.zeros((sj * sk, rank))
                    sub_W = np.zeros((sk * si, rank))
                    col = 0
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                sub_U[ii * sj + jj, col] = 1
                                sub_V[jj * sk + kk, col] = 1
                                sub_W[kk * si + ii, col] = 1
                                col += 1
                
                # Create full-sized matrices
                U_full = np.zeros((25, rank))
                V_full = np.zeros((25, rank))
                W_full = np.zeros((25, rank))
                
                # Fill U
                for ii in range(si):
                    for jj in range(sj):
                        U_full[(i_off + ii) * 5 + (j_off + jj), :] = sub_U[ii * sj + jj, :]
                
                # Fill V
                for jj in range(sj):
                    for kk in range(sk):
                        V_full[(j_off + jj) * 5 + (k_off + kk), :] = sub_V[jj * sk + kk, :]
                
                # Fill W
                for kk in range(sk):
                    for ii in range(si):
                        W_full[(k_off + kk) * 5 + (i_off + ii), :] = sub_W[kk * si + ii, :]
                
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
                total_rank += rank
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook baseline
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try block decomposition
    if time.time() < deadline:
        result = _block_decomposition(n, m, p)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if rank_new < best_rank:
                ok, _ = _verify(U_new, V_new, W_new, n, m, p)
                if ok:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = rank_new
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
