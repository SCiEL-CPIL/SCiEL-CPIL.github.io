import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 3, 4, 5
    
    # Partition: i into [1,1,1], j into [2,2], k into [2,2,1]
    i_blocks = [(0, 1), (1, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 5)]
    
    rank = 0
    U = [[0] * 1 for _ in range(n * m)]  # Will be extended
    V = [[0] * 1 for _ in range(m * p)]
    W = [[0] * 1 for _ in range(p * n)]
    
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                # Each (1, j_size, k_size) block has rank 1
                r = rank
                rank += 1
                # Extend U, V, W
                for row in U:
                    row.append(0)
                for row in V:
                    row.append(0)
                for row in W:
                    row.append(0)
                
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        U[i * m + j][r] = 1
                        for k in range(k_start, k_end):
                            V[j * p + k][r] = 1
                            W[k * n + i][r] = 1
    
    # Verify
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        raise ValueError(f"Verification failed at i={i}, j={j}, k={k}: sum={s}")
    
    return {"U": U, "V": V, "W": W}
