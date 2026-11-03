"""Strassen-based rank-54 decomposition for 2x5x6 matrix multiplication."""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Partition j (size 5) into blocks: [0,1], [2,3], [4]
    # Partition k (size 6) into blocks: [0,1], [2,3], [4,5]
    # i is size 2, one block
    
    j_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end)
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    rank = 0
    U = []  # will be list of lists, U[i*m+j][c]
    V = []  # V[j*p+k][c]
    W = []  # W[k*n+i][c]
    
    # Initialize
    total_rank = 54  # 6*7 + 3*4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for jb in j_blocks:
        j_start, j_end = jb
        j_size = j_end - j_start
        for kb in k_blocks:
            k_start, k_end = kb
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Strassen 2x2x2 block, 7 terms
                # Local indices: i in {0,1}, jl in {0,1}, kl in {0,1}
                # Global: i_global = i, j_global = j_start + jl, k_global = k_start + kl
                
                # Strassen's 7 terms (U, V, W) for local 2x2x2
                # Each is a tuple of 4 values for the 4 positions
                # U indexed by (i,j) -> (i*2+j)
                # V indexed by (j,k) -> (j*2+k)  
                # W indexed by (k,i) -> (k*2+i)
                
                strassen_terms = [
                    # (U_vals, V_vals, W_vals)
                    # P1 = (a00+a11)(b00+b11)
                    ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
                    # P2 = (a10+a11)b00
                    ([0,0,1,1], [1,0,0,0], [0,0,0,1]),
                    # P3 = a00(b01-b11)
                    ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
                    # P4 = a11(b10-b00)
                    ([0,0,0,1], [0,0,1,-1], [0,0,1,0]),
                    # P5 = (a00+a01)b11
                    ([1,1,0,0], [0,0,0,1], [0,0,0,1]),
                    # P6 = (a10-a00)(b00+b01)
                    ([-1,0,1,0], [1,1,0,0], [0,0,1,0]),
                    # P7 = (a01-a11)(b10+b11)
                    ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
                ]
                
                for u_vals, v_vals, w_vals in strassen_terms:
                    # u_vals indexed by (i, jl) -> i*2+jl
                    for i in range(2):
                        for jl in range(2):
                            U[i * m + (j_start + jl)][col] = u_vals[i * 2 + jl]
                    for jl in range(2):
                        for kl in range(2):
                            V[(j_start + jl) * p + (k_start + kl)][col] = v_vals[jl * 2 + kl]
                    for kl in range(2):
                        for i in range(2):
                            W[(k_start + kl) * n + i][col] = w_vals[kl * 2 + i]
                    col += 1
                    
            elif j_size == 1 and k_size == 2:
                # 2x1x2 block, schoolbook 4 terms
                # Local: i in {0,1}, jl=0, kl in {0,1}
                # j_global = j_start, k_global = k_start + kl
                for i in range(2):
                    for kl in range(2):
                        U[i * m + j_start][col] = 1
                        V[j_start * p + (k_start + kl)][col] = 1
                        W[(k_start + kl) * n + i][col] = 1
                        col += 1
            else:
                raise ValueError(f"Unexpected block size j={j_size}, k={k_size}")
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
