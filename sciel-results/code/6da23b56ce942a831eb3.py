import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 4, 4
    rank = n * m * p  # 32
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct standard matrix multiplication tensor decomposition
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term (i, j, k) gets a unique rank index r
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
                if time.time() > deadline:
                    # In case of extreme slowness (unlikely for this simple loop),
                    # we would need to return a partial valid decomposition.
                    # However, this loop is O(32), so it completes instantly.
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
