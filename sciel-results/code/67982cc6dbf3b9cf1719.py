"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 8 x 36
    V = [[0]*rank for _ in range(m*p)]  # 20 x 36
    W = [[0]*rank for _ in range(p*n)]  # 10 x 36
    
    col = 0
    
    for half in range(2):  # split m=4 into 2+2
        for third in range(3):  # split p=5 into 2+2+1
            if third < 2:
                # 2x2 * 2x2, Strassen rank 7
                h = half
                t = third
                
                # M1 = (a+d)(e+h)
                U[0*4 + 2*h + 0][col] = 1
                U[1*4 + 2*h + 0][col] = 1
                V[(2*h+0)*5 + 2*t+0][col] = 1
                V[(2*h+1)*5 + 2*t+1][col] = 1
                W[(2*t+0)*2 + 0][col] = 1
                W[(2*t+1)*2 + 1][col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[1*4 + 2*h + 0][col] = 1
                U[1*4 + 2*h + 1][col] = 1
                V[(2*h+0)*5 + 2*t+0][col] = 1
                W[(2*t+0)*2 + 1][col] = 1
                W[(2*t+1)*2 + 1][col] = -1
                col += 1
                
                # M3 = a(f-h)
                U[0*4 + 2*h + 0][col] = 1
                V[(2*h+0)*5 + 2*t+1][col] = 1
                V[(2*h+1)*5 + 2*t+1][col] = -1
                W[(2*t+1)*2 + 0][col] = 1
                W[(2*t+1)*2 + 1][col] = 1
                col += 1
                
                # M4 = d(g-e)
                U[1*4 + 2*h + 1][col] = 1
                V[(2*h+1)*5 + 2*t+0][col] = 1
                V[(2*h+0)*5 + 2*t+0][col] = -1
                W[(2*t+0)*2 + 0][col] = 1
                W[(2*t+0)*2 + 1][col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[0*4 + 2*h + 0][col] = 1
                U[0*4 + 2*h + 1][col] = 1
                V[(2*h+1)*5 + 2*t+1][col] = 1
                W[(2*t+0)*2 + 0][col] = -1
                W[(2*t+1)*2 + 0][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[1*4 + 2*h + 0][col] = 1
                U[0*4 + 2*h + 0][col] = -1
                V[(2*h+0)*5 + 2*t+0][col] = 1
                V[(2*h+0)*5 + 2*t+1][col] = 1
                W[(2*t+1)*2 + 1][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[0*4 + 2*h + 1][col] = 1
                U[1*4 + 2*h + 1][col] = -1
                V[(2*h+1)*5 + 2*t+0][col] = 1
                V[(2*h+1)*5 + 2*t+1][col] = 1
                W[(2*t+0)*2 + 0][col] = 1
                col += 1
            else:
                # 2x2 * 2x1, schoolbook rank 4
                h = half
                t = third
                
                # Term 1: A[0,2h] * B[2h, 2t]
                U[0*4 + 2*h + 0][col] = 1
                V[(2*h+0)*5 + 2*t+0][col] = 1
                W[(2*t+0)*2 + 0][col] = 1
                col += 1
                
                # Term 2: A[0,2h+1] * B[2h+1, 2t]
                U[0*4 + 2*h + 1][col] = 1
                V[(2*h+1)*5 + 2*t+0][col] = 1
                W[(2*t+0)*2 + 0][col] = 1
                col += 1
                
                # Term 3: A[1,2h] * B[2h, 2t]
                U[1*4 + 2*h + 0][col] = 1
                V[(2*h+0)*5 + 2*t+0][col] = 1
                W[(2*t+0)*2 + 1][col] = 1
                col += 1
                
                # Term 4: A[1,2h+1] * B[2h+1, 2t]
                U[1*4 + 2*h + 1][col] = 1
                V[(2*h+1)*5 + 2*t+0][col] = 1
                W[(2*t+0)*2 + 1][col] = 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c]
    # should equal delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # More precisely: the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a=(i,j), b=(j',k), c=(k',i')
    # T should equal 1 if j=j', k=k', i=i'
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.outer(np.outer(U_np[:,r], V_np[:,r]), W_np[:,r])
    
    # Check against expected
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1
    
    if not np.allclose(T, expected):
        # Fallback to schoolbook
        rank_school = n*m*p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
