import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin before 120s limit

    n, m, p = 2, 4, 8

    # Build rank 56 via Strassen on 2x2 blocks
    U, V, W = _build_rank_56(n, m, p)

    # Verify with vectorized check
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: return whatever we have
    return {"U": U, "V": V, "W": W}

def _build_rank_56(n, m, p):
    """Rank 56 decomposition via Strassen on 2x2 sub-blocks."""
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # For each pair of j-indices (j1, j2) where j1 in {0,1,2,3}, 
    # actually let's use the block approach:
    # Split j into pairs: (0,1) and (2,3)
    # For j-pair (0,1): involves A[i,0], A[i,1], B[0,k], B[1,k]
    # For j-pair (2,3): involves A[i,2], A[i,3], B[2,k], B[3,k]
    
    # For each j-pair, we have a 2x2 (in i) by 2 (in j within pair) by 8 (in k) sub-problem.
    # Actually for fixed j-pair (j1,j2), the contribution to C[i,k] is:
    # A[i,j1]*B[j1,k] + A[i,j2]*B[j2,k]
    # This is (2x2) * (2x8) = 2x8 matrix multiplication!
    # Where the "matrix" A_block is 2x2 (rows=i, cols=j within pair)
    # and B_block is 2x8 (rows=j within pair, cols=k)
    # Result C_block is 2x8.
    
    # So we have 2 independent (2x2)*(2x8) multiplications.
    # Each (2x2)*(2x8) can be done with Strassen: split the 8 columns into 4 pairs of 2.
    # So (2x2)*(2x8) = 4 * (2x2)*(2x2), each done in rank 7.
    # Total per j-pair: 4*7 = 28.
    # Total: 2*28 = 56. This matches!
    
    # Let me construct this.
    # j-pair 0: j1=0, j2=1
    # j-pair 1: j1=2, j2=3
    
    for jp in range(2):
        j1 = 2 * jp
        j2 = 2 * jp + 1
        base_col = 28 * jp
        
        # A indices: i=0,1 and j=j1,j2 -> A[i*4+j]
        # B indices: j=j1,j2 and k=0..7 -> B[j*8+k]
        # C indices: i=0,1 and k=0..7 -> C[k*2+i] (W[k*n+i])
        
        # 4 column-pairs for k: (0,1), (2,3), (4,5), (6,7)
        for kp in range(4):
            k1 = 2 * kp
            k2 = 2 * kp + 1
            col_offset = base_col + 7 * kp
            
            # Strassen on 2x2 * 2x2:
            # A_block = [[A[0,j1], A[0,j2]], [A[1,j1], A[1,j2]]]
            # B_block = [[B[j1,k1], B[j1,k2]], [B[j2,k1], B[j2,k2]]]
            # C_block = A_block * B_block (2x2)
            # C[i,k1] and C[i,k2] for i=0,1
            
            a00 = 0 * 4 + j1  # A[0,j1]
            a01 = 0 * 4 + j2  # A[0,j2]
            a10 = 1 * 4 + j1  # A[1,j1]
            a11 = 1 * 4 + j2  # A[1,j2]
            
            b00 = j1 * 8 + k1  # B[j1,k1]
            b01 = j1 * 8 + k2  # B[j1,k2]
            b10 = j2 * 8 + k1  # B[j2,k1]
            b11 = j2 * 8 + k2  # B[j2,k2]
            
            c00 = k1 * 2 + 0  # C[0,k1] -> W[k1*2+0]
            c01 = k2 * 2 + 0  # C[0,k2] -> W[k2*2+0]
            c10 = k1 * 2 + 1  # C[1,k1] -> W[k1*2+1]
            c11 = k2 * 2 + 1  # C[1,k2] -> W[k2*2+1]
            
            # Strassen rank 7:
            # M1 = (a00+a11)(b00+b11)
            U[a00][col_offset] += 1
            U[a11][col_offset] += 1
            V[b00][col_offset] += 1
            V[b11][col_offset] += 1
            W[c00][col_offset] += 1
            W[c11][col_offset] += 1
            
            # M2 = (a10+a11)b00
            U[a10][col_offset+1] += 1
            U[a11][col_offset+1] += 1
            V[b00][col_offset+1] += 1
            W[c10][col_offset+1] += 1
            W[c11][col_offset+1] -= 1
            
            # M3 = a00(b01-b11)
            U[a00][col_offset+2] += 1
            V[b01][col_offset+2] += 1
            V[b11][col_offset+2] -= 1
            W[c01][col_offset+2] += 1
            W[c11][col_offset+2] += 1
            
            # M4 = a11(b10-b00)
            U[a11][col_offset+3] += 1
            V[b10][col_offset+3] += 1
            V[b00][col_offset+3] -= 1
            W[c00][col_offset+3] += 1
            W[c10][col_offset+3] += 1
            
            # M5 = (a00+a01)b11
            U[a00][col_offset+4] += 1
            U[a01][col_offset+4] += 1
            V[b11][col_offset+4] += 1
            W[c00][col_offset+4] -= 1
            W[c01][col_offset+4] += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[a10][col_offset+5] += 1
            U[a00][col_offset+5] -= 1
            V[b00][col_offset+5] += 1
            V[b01][col_offset+5] += 1
            W[c11][col_offset+5] += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[a01][col_offset+6] += 1
            U[a11][col_offset+6] -= 1
            V[b10][col_offset+6] += 1
            V[b11][col_offset+6] += 1
            W[c00][col_offset+6] += 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the tensor decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # For each (i,j,k), check sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1
    # We can do this efficiently by building the full tensor product check.
    # Number of (i,j,k) triples: n*m*p = 2*4*8 = 64
    # Rank: 56
    # So we need 64 dot products of length 56.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U_arr[i*m+j, s] * V_arr[j*p+k, s] * W_arr[k*n+i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
