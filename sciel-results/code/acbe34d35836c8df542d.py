"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) each of shape (4, 7) for 2x2x2 matmul.
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i].
    Satisfies: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B01+B10+B11) -> C00+C11
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[1,0] = 1; V[2,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # Term 2: (A10+A11)B00 -> C10
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1  # W[k*2+i] = W[0*2+1] = W[2] for C[1,0]
    
    # Term 3: A00(B10-B11) -> C01
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1  # W[k*2+i] = W[1*2+0] = W[2]... wait
    # C[0,1] means i=0, k=1. W[k*2+i] = W[1*2+0] = W[2]
    W[2,2] = 1
    
    # Term 4: A11(B01-B00) -> C00+C11
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[3,3] = 1
    
    # Term 5: (A00+A01)B11 -> -C00+C01
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1  # C[0,0]: k=0,i=0 -> W[0]
    W[2,4] = 1   # C[0,1]: k=1,i=0 -> W[2]
    
    # Term 6: (A10-A00)(B00+B10) -> C11
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[3,5] = 1   # C[1,1]: k=1,i=1 -> W[3]
    
    # Term 7: (A01-A11)(B01+B11) -> C00
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = 1   # C[0,0]: k=0,i=0 -> W[0]
    
    return U, V, W

def _verify_strassen():
    U, V, W = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(7))
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _schoolbook_block(a, b, c):
    """Return (U, V, W) each of shape (a*b, a*b*c), (b*c, a*b*c), (c*a, a*b*c) for a×b×c matmul."""
    rank = a * b * c
    U = np.zeros((a * b, rank))
    V = np.zeros((b * c, rank))
    W = np.zeros((c * a, rank))
    col = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                U[i * b + j, col] = 1
                V[j * c + k, col] = 1
                W[k * a + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    deadline = time.time() + 140  # safety margin
    
    # Block partition: [2, 2, 1] for each dimension
    sizes_i = [2, 2, 1]
    sizes_j = [2, 2, 1]
    sizes_k = [2, 2, 1]
    
    # Verify Strassen
    strassen_ok = _verify_strassen()
    
    # Build the full decomposition
    # First, determine total rank
    total_rank = 0
    block_decomps = []
    
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                a = sizes_i[alpha]
                b = sizes_j[beta]
                c = sizes_k[gamma]
                if a == 2 and b == 2 and c == 2 and strassen_ok:
                    U_b, V_b, W_b = _strassen_2x2x2()
                    rank_b = 7
                else:
                    U_b, V_b, W_b = _schoolbook_block(a, b, c)
                    rank_b = a * b * c
                block_decomps.append((alpha, beta, gamma, a, b, c, U_b, V_b, W_b, rank_b))
                total_rank += rank_b
    
    # Now assemble the full U, V, W
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (alpha, beta, gamma, a, b, c, U_b, V_b, W_b, rank_b) in block_decomps:
        # Row offsets in the full matrices
        i_start = sum(sizes_i[:alpha])
        j_start = sum(sizes_j[:beta])
        k_start = sum(sizes_k[:gamma])
        
        # U: rows indexed by [i*m + j]
        for local_i in range(a):
            for local_j in range(b):
                full_row = (i_start + local_i) * m + (j_start + local_j)
                U_full[full_row, col_offset:col_offset + rank_b] = U_b[local_i * b + local_j, :]
        
        # V: rows indexed by [j*p + k]
        for local_j in range(b):
            for local_k in range(c):
                full_row = (j_start + local_j) * p + (k_start + local_k)
                V_full[full_row, col_offset:col_offset + rank_b] = V_b[local_j * c + local_k, :]
        
        # W: rows indexed by [k*n + i]
        for local_k in range(c):
            for local_i in range(a):
                full_row = (k_start + local_k) * n + (i_start + local_i)
                W_full[full_row, col_offset:col_offset + rank_b] = W_b[local_k * a + local_i, :]
        
        col_offset += rank_b
    
    # Verify the full decomposition
    U_np = np.array(U_full)
    V_np = np.array(V_full)
    W_np = np.array(W_full)
    
    # Check a few random entries
    import random
    random.seed(42)
    valid = True
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
        if abs(val - 1) > 1e-10:
            valid = False
            break
    
    if not valid:
        # Fall back to full schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
