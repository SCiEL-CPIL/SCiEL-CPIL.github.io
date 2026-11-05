"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0

def construct():
    n, m, p = DIMENSIONS
    # Block sizes: 2, 2, 1
    block_sizes = [2, 2, 1]
    num_blocks = len(block_sizes)
    
    # Precompute Strassen factors for 2x2x2
    U_strassen = [[1,0,1,0,1,-1,0],
                  [0,0,0,0,1,0,1],
                  [0,1,0,0,0,1,0],
                  [1,1,0,1,0,0,-1]]
    V_strassen = [[1,1,0,-1,0,1,0],
                  [0,0,1,0,0,1,0],
                  [0,0,0,1,0,0,1],
                  [1,0,-1,0,1,0,1]]
    W_strassen = [[1,0,0,1,-1,0,1],
                  [0,0,1,0,1,0,0],
                  [0,1,0,1,0,0,0],
                  [1,-1,1,0,0,1,1]]
    
    # Compute total rank
    total_rank = 0
    for i_b in range(num_blocks):
        for k_b in range(num_blocks):
            for j_b in range(num_blocks):
                si, sk, sj = block_sizes[i_b], block_sizes[k_b], block_sizes[j_b]
                if si == 2 and sk == 2 and sj == 2:
                    total_rank += 7
                else:
                    total_rank += si * sk * sj
    
    # Build U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in range(num_blocks):
        for k_b in range(num_blocks):
            for j_b in range(num_blocks):
                si, sk, sj = block_sizes[i_b], block_sizes[k_b], block_sizes[j_b]
                
                i_start = sum(block_sizes[:i_b])
                k_start = sum(block_sizes[:k_b])
                j_start = sum(block_sizes[:j_b])
                
                if si == 2 and sk == 2 and sj == 2:
                    # Use Strassen for this 2x2x2 block
                    for c in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                i_global = i_start + ii
                                j_global = k_start + jj
                                U[i_global * m + j_global][col + c] += U_strassen[ii * 2 + jj][c]
                        for jj in range(2):
                            for kk in range(2):
                                j_global = k_start + jj
                                k_global = j_start + kk
                                V[j_global * p + k_global][col + c] += V_strassen[jj * 2 + kk][c]
                        for kk in range(2):
                            for ii in range(2):
                                k_global = j_start + kk
                                i_global = i_start + ii
                                W[k_global * n + i_global][col + c] += W_strassen[kk * 2 + ii][c]
                    col += 7
                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sk):
                            for kk in range(sj):
                                i_global = i_start + ii
                                j_global = k_start + jj
                                k_global = j_start + kk
                                U[i_global * m + j_global][col] = 1
                                V[j_global * p + k_global][col] = 1
                                W[k_global * n + i_global][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
