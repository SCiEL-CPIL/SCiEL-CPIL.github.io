import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # 8 blocks, each rank 7
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)e
    # M6 = (c+d)(g+h)
    # M7 = (a-b)(e+f)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M1 + M3
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Blocks:
    # i0 is always 0 (since n=2, only one i block)
    # j0 in {0, 2}
    # k0 in {0, 2, 4, 6}
    
    block_count = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # i0 = 0
            i0 = 0
            # Process 7 terms
            for t in range(7):
                col = block_count * 7 + t
                # Update U, V, W based on term t
                # a = U[i0, j0], b = U[i0, j0+1], c = U[i0+1, j0], d = U[i0+1, j0+1]
                # e = V[j0, k0], f = V[j0, k0+1], g = V[j0+1, k0], h = V[j0+1, k0+1]
                
                if t == 0: # M1
                    U[i0*m + j0][col] += 1
                    U[(i0+1)*m + j0+1][col] += 1
                    V[j0*p + k0][col] += 1
                    V[(j0+1)*p + k0+1][col] += 1
                    W[k0*n + i0][col] += 1
                    W[(k0+1)*n + i0][col] += 1
                    W[k0*n + i0+1][col] += 1
                    W[(k0+1)*n + i0+1][col] += 1
                elif t == 1: # M2
                    U[(i0+1)*m + j0][col] += 1
                    U[(i0+1)*m + j0+1][col] += 1
                    V[(j0+1)*p + k0+1][col] += 1
                    W[k0*n + i0+1][col] += 1
                    W[(k0+1)*n + i0+1][col] += -1
                elif t == 2: # M3
                    U[i0*m + j0][col] += 1
                    V[j0*p + k0+1][col] += 1
                    V[(j0+1)*p + k0+1][col] += -1
                    W[(k0+1)*n + i0][col] += 1
                    W[(k0+1)*n + i0+1][col] += 1
                elif t == 3: # M4
                    U[(i0+1)*m + j0+1][col] += 1
                    V[(j0+1)*p + k0][col] += 1
                    V[j0*p + k0][col] += -1
                    W[k0*n + i0][col] += 1
                elif t == 4: # M5
                    U[i0*m + j0][col] += 1
                    U[i0*m + j0+1][col] += 1
                    V[j0*p + k0][col] += 1
                    W[k0*n + i0][col] += -1
                    W[k0*n + i0+1][col] += 1
                elif t == 5: # M6
                    U[(i0+1)*m + j0][col] += 1
                    U[(i0+1)*m + j0+1][col] += 1
                    V[(j0+1)*p + k0][col] += 1
                    V[(j0+1)*p + k0+1][col] += 1
                    W[(k0+1)*n + i0+1][col] += 1
                elif t == 6: # M7
                    U[i0*m + j0][col] += 1
                    U[i0*m + j0+1][col] += -1
                    V[j0*p + k0][col] += 1
                    V[j0*p + k0+1][col] += 1
                    W[k0*n + i0][col] += 1
            block_count += 1
            
    return {"U": U, "V": V, "W": W}
